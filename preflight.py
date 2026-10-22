"""Read-only site preflight. Standard library only; no AI calls or credentials."""
import argparse
import json
import time
import urllib.error
import urllib.request
import urllib.robotparser
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

BASE = 'https://www.fnx.co.il'
AGENT = 'PhoenixRAGPrototype/0.1'
LIMIT = 8 * 1024 * 1024
SAMPLES = ['/apartment-insurance/', '/claims-customer-service/claims-appartment-insurance/damage-to-the-apartment/']


def in_scope(url):
    p = urlsplit(url)
    return p.scheme == 'https' and p.hostname == 'www.fnx.co.il' and p.port in (None, 443) and not p.username and not p.password


class ScopedRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not in_scope(newurl):
            raise ValueError('Redirect outside configured scope')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch(url):
    if not in_scope(url):
        raise ValueError('URL outside configured scope')
    req = urllib.request.Request(url, headers={'User-Agent': AGENT})
    with urllib.request.build_opener(ScopedRedirect()).open(req, timeout=25) as response:
        raw = response.read(LIMIT + 1)
        if len(raw) > LIMIT:
            raise ValueError('Response exceeds 8 MiB limit')
        return raw.decode(response.headers.get_content_charset() or 'utf-8', errors='replace'), response.headers.get('Content-Type', '')


class VisibleText(HTMLParser):
    """Diagnostic only: includes navigation; NOT the production extractor."""
    def __init__(self):
        super().__init__()
        self.hidden = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style', 'noscript'):
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'noscript'):
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data):
        if not self.hidden and data.strip():
            self.parts.append(data.strip())


def sitemap_urls(xml):
    root = ET.fromstring(xml)
    return root.tag.rsplit('}', 1)[-1], [node.text.strip() for node in root.iter() if node.tag.rsplit('}', 1)[-1] == 'loc' and node.text]


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    report = {'checked_at': datetime.now(timezone.utc).isoformat(), 'pages': [], 'errors': []}
    try:
        robots, _ = fetch(BASE + '/robots.txt')
        (output / 'robots.txt').write_text(robots, encoding='utf-8')
        policy = urllib.robotparser.RobotFileParser()
        policy.parse(robots.splitlines())
        report['robots'] = {'status': 'read', 'sitemaps': policy.site_maps() or [], 'crawl_delay': policy.crawl_delay(AGENT)}
    except Exception as error:
        report['errors'].append('robots could not be verified; page checks stopped: ' + str(error))
        (output / 'preflight.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        return report
    delay = max(1, policy.crawl_delay(AGENT) or 0)
    rate = policy.request_rate(AGENT)
    if rate and rate.requests > 0:
        delay = max(delay, rate.seconds / rate.requests)
    targets = policy.site_maps() or [BASE + '/sitemap.xml']
    report['sitemaps'] = []
    for url in targets[:5]:
        if not in_scope(url) or not policy.can_fetch(AGENT, url):
            report['sitemaps'].append({'url': url, 'status': 'skipped'})
            continue
        try:
            time.sleep(delay)
            xml, _ = fetch(url)
            kind, urls = sitemap_urls(xml)
            report['sitemaps'].append({'url': url, 'kind': kind, 'entries': len(urls), 'examples': urls[:8]})
            (output / ('sitemap-' + str(len(report['sitemaps'])) + '.xml')).write_text(xml, encoding='utf-8')
        except Exception as error:
            report['errors'].append(url + ': ' + str(error))
    for path in SAMPLES:
        url = BASE + path
        if not policy.can_fetch(AGENT, url):
            report['pages'].append({'url': url, 'allowed_by_robots': False})
            continue
        try:
            time.sleep(delay)
            html, content_type = fetch(url)
            parser = VisibleText()
            parser.feed(html)
            text = '\n'.join(parser.parts)
            report['pages'].append({'url': url, 'allowed_by_robots': True, 'content_type': content_type, 'html_chars': len(html), 'visible_text_chars': len(text), 'hebrew_chars': sum('\u0590' <= c <= '\u05ff' for c in text), 'text_preview': text[:600]})
            (output / ('sample-' + str(len(report['pages'])) + '.txt')).write_text(text, encoding='utf-8')
        except Exception as error:
            report['errors'].append(url + ': ' + str(error))
    (output / 'preflight.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('data/preflight'))
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(1 if result['errors'] else 0)
