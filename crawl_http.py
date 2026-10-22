"""Bounded, same-origin, robots-aware HTTP transport. No automatic redirects."""
import email.utils
import math
import posixpath
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from urllib.parse import quote, unquote, urljoin, urlsplit, urlunsplit

from protego import Protego


class CrawlError(Exception):
    pass


class StopCrawl(CrawlError):
    """Host cooldown or request budget: stop the whole run, not just this page."""


def normalize(url, base='https://www.fnx.co.il'):
    try:
        p = urlsplit(urljoin(base + '/', url))
        origin = urlsplit(base)
        if p.scheme != 'https' or p.hostname != origin.hostname or p.port not in (None, 443) or p.username or p.password or p.query:
            return None
        path = unquote(p.path or '/')
        if '\\' in path or re.search(r'[\x00-\x1f\x7f]', path):
            return None
        path = posixpath.normpath('/' + path.lstrip('/')) + ('/' if path.endswith('/') and path != '/' else '')
        path = quote(path, safe="/-._~!$&'()*+,;=:@")
        return urlunsplit(('https', origin.hostname, path, '', ''))
    except (ValueError, TypeError):
        return None


def redirect_target(location, current, base):
    """Upgrade same-host HTTP Location headers without ever requesting HTTP."""
    try:
        target = urlsplit(urljoin(current, location))
        if (target.scheme == 'http' and target.hostname == urlsplit(base).hostname
                and target.port in (None, 80) and not target.username and not target.password):
            target = target._replace(scheme='https', netloc=urlsplit(base).netloc)
        return normalize(target.geturl(), base)
    except (ValueError, TypeError):
        return None


def retry_delay(value, now=None):
    if not value:
        return 0.0
    try:
        seconds = float(value)
    except ValueError:
        try:
            target = email.utils.parsedate_to_datetime(value)
            seconds = (target - (now or datetime.now(timezone.utc))).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return 0.0
    return max(0.0, seconds) if math.isfinite(seconds) else 0.0


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None


class Client:
    def __init__(self, config, opener=None, clock=time.monotonic, sleeper=time.sleep):
        self.c = config
        self.opener = opener or urllib.request.build_opener(NoRedirect())
        self.clock, self.sleep = clock, sleeper
        self.next_at = 0.0
        self.delay = max(1.0, config['delay_seconds'])
        self.requests = 0
        self.policy = None
        self.events = []

    def initialize(self):
        url = self.c['base_url'] + '/robots.txt'
        body, _, _ = self.fetch(url, bootstrap=True)
        text = body.decode('utf-8-sig', errors='replace')
        if '<html' in text.lower() or '<!doctype' in text.lower():
            raise CrawlError('robots returned HTML; policy not verified')
        self.policy = Protego.parse(text)
        delay = self.policy.crawl_delay(self.c['user_agent']) or 0
        rate = self.policy.request_rate(self.c['user_agent'])
        if self.policy.visit_time(self.c['user_agent']) or (rate and (rate.start_time or rate.end_time)):
            raise CrawlError('Time-window robots policy requires explicit scheduling support')
        self.delay = max(self.delay, delay, rate.seconds / rate.requests if rate and rate.requests else 0)
        self.next_at = max(self.next_at, self.clock() + self.delay)
        return text

    def fetch(self, url, bootstrap=False):
        url = normalize(url, self.c['base_url'])
        if not url:
            raise CrawlError('outside_scope_or_query')
        redirects = set()
        for _ in range(6):
            if url in redirects:
                raise CrawlError('redirect_loop')
            redirects.add(url)
            if bootstrap and url != self.c['base_url'] + '/robots.txt':
                raise CrawlError('robots_redirect_requires_review')
            if not bootstrap and any(x in urlsplit(url).path.lower().split('/') for x in ('login', 'signin', 'logout', 'personal-area', 'my-account', 'account')):
                raise CrawlError('private_or_action_path')
            if not bootstrap and (self.policy is None or not self.policy.can_fetch(url, self.c['user_agent'])):
                raise CrawlError('robots_denied')
            for attempt in range(self.c['max_retries'] + 1):
                if self.requests >= self.c['max_requests']:
                    raise StopCrawl('request_budget_reached')
                wait = max(0, self.next_at - self.clock())
                if wait > 60:
                    raise StopCrawl('host_cooldown_over_60_seconds; retry later')
                self.sleep(wait)
                self.requests += 1
                self.next_at = self.clock() + self.delay
                req = urllib.request.Request(url, headers={'User-Agent': self.c['user_agent'], 'Accept-Encoding': 'identity'})
                try:
                    with self.opener.open(req, timeout=self.c['timeout_seconds']) as response:
                        body = response.read(self.c['max_response_bytes'] + 1)
                        if len(body) > self.c['max_response_bytes']:
                            raise CrawlError('response_too_large')
                        content_type = response.headers.get('Content-Type', '')
                        self.events.append({'url': url, 'status': response.status, 'bytes': len(body)})
                        return body, content_type, url
                except urllib.error.HTTPError as error:
                    code, headers = error.code, error.headers
                    error.close()
                    self.events.append({'url': url, 'status': code})
                    if code in (301, 302, 303, 307, 308):
                        target = redirect_target(headers.get('Location', ''), url, self.c['base_url'])
                        if not target or not headers.get('Location'):
                            raise CrawlError('redirect_outside_scope_or_missing_location')
                        url = target
                        break
                    if code in (429, 500, 502, 503, 504):
                        cooldown = max(2 ** (attempt + 1), retry_delay(headers.get('Retry-After')))
                        self.next_at = max(self.next_at, self.clock() + cooldown)
                        if cooldown > 60:
                            raise StopCrawl('server_requested_long_cooldown; retry later')
                        if attempt < self.c['max_retries']:
                            continue
                    raise CrawlError(f'http_{code}') from error
                except (urllib.error.URLError, TimeoutError, OSError) as error:
                    self.events.append({'url': url, 'status': 'network_error', 'error': type(error).__name__})
                    if attempt == self.c['max_retries']:
                        raise CrawlError('network_failed: ' + str(error)) from error
                    self.next_at = max(self.next_at, self.clock() + 2 ** (attempt + 1))
            else:
                raise CrawlError('retry_exhausted')
        raise CrawlError('too_many_redirects')
