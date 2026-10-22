"""Retry eligible failed HTML pages and combine snapshots. No AI/API calls."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

from crawler import crawl, load_config
from crawl_http import normalize
from ingest import prepare


def retryable(failure):
    reason = failure.get('error', '')
    return failure.get('stage') == 'page' and (
        reason == 'redirect_outside_scope_or_missing_location'
        or reason.startswith('network_failed:') or reason in {'http_500', 'http_502', 'http_503', 'http_504'})


def combine(original, recovery):
    """Retain source HTML in place; never rewrite an earlier snapshot."""
    destination = recovery.with_name(recovery.name + '-combined')
    destination.mkdir(exist_ok=False)
    by_url = {}
    for run in [original, recovery]:
        for line in (run / 'pages.jsonl').read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            raw = (run / row['raw_file']).resolve()
            if not raw.is_relative_to(run.resolve()) or not raw.is_file():
                raise ValueError('Missing or invalid original HTML path')
            row['raw_file'] = Path(os.path.relpath(raw, destination)).as_posix()
            previous = by_url.get(row['url'])
            if not previous or row.get('status') == 'extracted':
                by_url[row['url']] = row
    hashes = {}
    for row in by_url.values():
        if row.get('status') == 'extracted':
            digest = row.get('content_hash')
            if digest and digest in hashes:
                row.update(status='duplicate_content', duplicate_of=hashes[digest])
            elif digest:
                hashes[digest] = row['url']
    (destination / 'pages.jsonl').write_text(''.join(
        json.dumps(row, ensure_ascii=False) + '\n' for row in by_url.values()), encoding='utf-8')
    report = {'kind': 'combined_snapshots', 'sources': [str(original), str(recovery)],
              'page_status_counts': dict(Counter(row['status'] for row in by_url.values())),
              'coverage_complete': False, 'pdf_content_included': False,
              'raw_html_note': 'Keep both source folders: raw_file paths refer to saved source HTML.'}
    (destination / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original', type=Path)
    args = parser.parse_args()
    original = args.original
    manifest = json.loads((original / 'manifest.json').read_text(encoding='utf-8'))
    config = load_config(Path(__file__).with_name('crawl_config.json'))
    seeds = list(dict.fromkeys(f['url'] for f in manifest['failures'] if retryable(f)))
    if not seeds:
        raise SystemExit('No eligible failed pages to retry.')
    config.update(seeds=seeds, max_pages=len(seeds), max_requests=len(seeds) * 8 + 20)
    known = set()
    for line in (original / 'pages.jsonl').read_text(encoding='utf-8').splitlines():
        if line.strip():
            row = json.loads(line)
            known.update(filter(None, [normalize(row['url']), normalize(row.get('requested_url', row['url']))]))
    print(f'Retrying {len(seeds)} failed URLs only. Existing pages and 404s are not re-crawled. No AI/API calls.', flush=True)
    recovery, result = crawl(config, original.parent, known_urls=known,
                             discover_sitemaps=False, discover_links=False)
    print(json.dumps({'recovery': str(recovery), 'status': result['status'],
                      'new_pages': len(result['pages']), 'remaining_errors': len(result['failures'])}), flush=True)
    if not (recovery / 'pages.jsonl').exists():
        raise SystemExit('Recovery did not reach page processing. Keep the original snapshot.')
    combined = combine(original, recovery)
    _, estimate = prepare(combined / 'pages.jsonl')
    brief = {key: estimate[key] for key in ['source_pages', 'included_pages', 'chunks',
                                           'embedding_tokens', 'estimated_embedding_usd', 'model']}
    brief.update(combined_pages=str(combined / 'pages.jsonl'), dry_run=True)
    (combined / 'index-estimate.json').write_text(json.dumps(brief, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(brief, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
