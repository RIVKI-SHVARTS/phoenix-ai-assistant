"""General discovery + bounded crawl. Each run is a new, auditable snapshot."""
import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import uuid
import xml.etree.ElementTree as ET

from crawl_http import Client, CrawlError, StopCrawl, normalize
from extract_content import extract_page, VERSION


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def parse_sitemap(raw, limit):
    if raw.startswith(b'\x1f\x8b'):
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
            raw = stream.read(limit + 1)
    if len(raw) > limit or b'<!DOCTYPE' in raw.upper() or b'<!ENTITY' in raw.upper():
        raise CrawlError('unsafe_or_oversize_sitemap')
    root = ET.fromstring(raw)
    kind = root.tag.rsplit('}', 1)[-1]
    if kind not in ('urlset', 'sitemapindex'):
        raise CrawlError('unexpected_sitemap_root')
    # Only direct child loc fields; image/video extension URLs are not pages.
    urls = [loc.text.strip() for child in root for loc in child if loc.tag.rsplit('}', 1)[-1] == 'loc' and loc.text]
    return kind, urls


def classify(url):
    from urllib.parse import urlsplit
    path = urlsplit(url).path.lower()
    if any(x in path.split('/') for x in ('login', 'signin', 'logout', 'personal-area', 'my-account', 'account')):
        return 'private_or_action_path'
    suffix = Path(path).suffix
    if suffix == '.pdf':
        return 'pdf_deferred'
    if suffix and suffix not in ('.html', '.htm', '.aspx'):
        return 'non_html_resource'
    return 'page'


def load_config(path, max_pages=None):
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    if c['base_url'] != 'https://www.fnx.co.il':
        raise ValueError('This project is scoped to https://www.fnx.co.il')
    if max_pages is not None:
        c['max_pages'] = max_pages
    for key in ('max_pages', 'max_requests', 'max_sitemaps', 'max_discovered', 'timeout_seconds', 'max_response_bytes'):
        if not isinstance(c.get(key), int) or isinstance(c.get(key), bool) or c[key] <= 0:
            raise ValueError(key + ' must be a positive integer')
    if not 0 <= c['max_retries'] <= 5 or not 1 <= c['delay_seconds'] <= 60:
        raise ValueError('Invalid retries or delay')
    return c


def crawl(config, output_root, client=None, known_urls=(), discover_sitemaps=True, discover_links=True):
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    out = Path(output_root) / run_id
    out.mkdir(parents=True, exist_ok=False)
    (out / 'raw').mkdir()
    client = client or Client(config)
    manifest = {'run_id': run_id, 'started_at': utcnow(), 'extractor_version': VERSION, 'config': config,
                'status': 'running', 'sitemaps': [], 'failures': [], 'pages': [], 'discovery_truncated': False}
    known_urls = set(known_urls)
    queue, discovered, maps_seen, final_seen, hashes = deque(), {}, set(), set(known_urls), {}
    attempts = 0

    def discover(raw, origin):
        if raw in discovered:
            return
        if len(discovered) >= config['max_discovered']:
            manifest['discovery_truncated'] = True
            return
        url = normalize(raw, config['base_url'])
        if url in known_urls:
            return
        if url and url in discovered:
            return
        key = url or raw
        state = classify(url) if url else 'outside_scope_or_query'
        if state == 'page' and client.policy and not client.policy.can_fetch(url, config['user_agent']):
            state = 'robots_denied'
        discovered[key] = {'url': key, 'found_in': origin, 'status': 'queued' if state == 'page' else state}
        if state == 'page':
            queue.append(url)

    try:
        robots = client.initialize()
        (out / 'robots.txt').write_text(robots, encoding='utf-8')
        manifest['effective_delay_seconds'] = client.delay
        for seed in config['seeds']:
            discover(seed, 'seed')
        map_queue = deque(client.policy.sitemaps or []) if discover_sitemaps else deque()
        if discover_sitemaps and not map_queue:
            map_queue.append(config['base_url'] + '/sitemap.xml')
        while map_queue and len(maps_seen) < config['max_sitemaps']:
            raw_url = map_queue.popleft()
            url = normalize(raw_url, config['base_url'])
            if not url:
                manifest['sitemaps'].append({'url': raw_url, 'status': 'outside_scope_or_query'})
                continue
            if url in maps_seen:
                continue
            maps_seen.add(url)
            try:
                raw, _, final = client.fetch(url)
                kind, urls = parse_sitemap(raw, config['max_response_bytes'])
                (out / f'sitemap-{len(maps_seen)}.xml').write_bytes(gzip.decompress(raw) if raw.startswith(b'\x1f\x8b') else raw)
                manifest['sitemaps'].append({'url': final, 'status': 'read', 'kind': kind, 'entries': len(urls)})
                if kind == 'sitemapindex':
                    # Bound recursive discovery too; duplicate/cyclic maps are visited once.
                    for child in urls:
                        if len(map_queue) + len(maps_seen) < config['max_sitemaps']:
                            map_queue.append(child)
                        else:
                            manifest['discovery_truncated'] = True
                else:
                    for child in urls:
                        discover(child, final)
            except StopCrawl:
                raise
            except (CrawlError, ET.ParseError, OSError, EOFError) as error:
                manifest['failures'].append({'url': url, 'stage': 'sitemap', 'error': str(error)})
        if map_queue:
            manifest['discovery_truncated'] = True
        with (out / 'pages.jsonl').open('w', encoding='utf-8') as sink:
            while queue and attempts < config['max_pages']:
                url = queue.popleft()
                if url in final_seen:
                    discovered[url]['status'] = 'duplicate_final_url'
                    continue
                attempts += 1
                discovered[url]['status'] = 'fetching'
                try:
                    raw, content_type, final = client.fetch(url)
                    if final in final_seen:
                        discovered[url]['status'] = 'duplicate_final_url'
                        continue
                    final_seen.add(final)
                    if content_type.split(';')[0].lower().strip() not in ('text/html', 'application/xhtml+xml'):
                        discovered[url]['status'] = 'non_html_response'
                        continue
                    key = hashlib.sha256(final.encode()).hexdigest()[:20]
                    (out / 'raw' / f'{key}.html').write_bytes(raw)
                    # BeautifulSoup detects HTML encoding from original response bytes.
                    page = extract_page(raw, final)
                    links = page.pop('discovered_links')
                    if discover_links:
                        for link in links:
                            discover(link, final)
                    page.update({'requested_url': url, 'fetched_at': utcnow(), 'content_type': content_type,
                                 'raw_file': f'raw/{key}.html', 'raw_sha256': hashlib.sha256(raw).hexdigest()})
                    digest = page.get('content_hash')
                    if digest and digest in hashes:
                        page['duplicate_of'] = hashes[digest]
                        page['status'] = 'duplicate_content'
                    elif digest:
                        hashes[digest] = final
                    sink.write(json.dumps(page, ensure_ascii=False) + '\n')
                    sink.flush()
                    discovered[url]['status'] = page['status']
                    manifest['pages'].append({'url': final, 'status': page['status'], 'warnings': page['warnings'], 'quality': page.get('quality', {})})
                    print(f"{attempts}: {page['status']} {final}", flush=True)
                except StopCrawl:
                    discovered[url]['status'] = 'stopped'
                    raise
                except (CrawlError, ValueError, OSError) as error:
                    discovered[url]['status'] = 'failed'
                    manifest['failures'].append({'url': url, 'stage': 'page', 'error': str(error)})
        manifest['status'] = 'completed_with_errors' if manifest['failures'] else 'completed'
        manifest['stop_reason'] = 'max_pages' if queue else 'queue_exhausted'
    except StopCrawl as error:
        manifest['status'], manifest['stop_reason'] = 'stopped', str(error)
    except (CrawlError, ValueError, OSError) as error:
        manifest['status'], manifest['stop_reason'] = 'failed', str(error)
    except KeyboardInterrupt:
        manifest['status'], manifest['stop_reason'] = 'interrupted', 'user_interrupt'
    except Exception as error:
        manifest['status'], manifest['stop_reason'] = 'failed', 'unexpected_error: ' + type(error).__name__
        raise
    finally:
        manifest.update({'finished_at': utcnow(), 'page_attempts': attempts, 'http_requests': client.requests,
                         'discovered_count': len(discovered), 'remaining_queue': len(queue),
                         'discovery_status_counts': dict(Counter(x['status'] for x in discovered.values()))})
        (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        with (out / 'discovered.jsonl').open('w', encoding='utf-8') as f:
            for row in discovered.values():
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        (out / 'http-events.json').write_text(json.dumps(client.events, ensure_ascii=False, indent=2), encoding='utf-8')
    return out, manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='crawl_config.json')
    parser.add_argument('--max-pages', type=int)
    parser.add_argument('--output', type=Path, default=Path('data/crawls'))
    args = parser.parse_args()
    output, result = crawl(load_config(args.config, args.max_pages), args.output)
    print(json.dumps({'output': str(output), 'status': result['status'], 'pages': len(result['pages']), 'errors': len(result['failures'])}, ensure_ascii=False))
    raise SystemExit(0 if result['status'] == 'completed' else 1)
