"""Sequential paid checks of the running LOCAL app; save evidence, never grade truth."""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE = Path(__file__).resolve().parent


def request_json(path, payload=None):
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8')
    request = Request('http://127.0.0.1:5000' + path, data=body,
                      headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=310) as response:
        return json.load(response)


def fingerprint(cases_path):
    digest = hashlib.sha256()
    for path in [cases_path, BASE / 'answering.py', BASE / 'retrieval.py', BASE / 'app.py',
                 BASE / 'data/index/CURRENT.json']:
        digest.update(str(path.relative_to(BASE)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def structural_issues(result):
    """A quote match proves source membership only, NOT factual correctness."""
    if not isinstance(result, dict) or result.get('status') not in ('answered', 'insufficient', 'clarification', 'limited'):
        return ['Invalid response status']
    issues = []
    if not isinstance(result.get('answer'), str) or not result['answer'].strip():
        issues.append('Empty answer')
    if result['status'] == 'answered':
        sources = {s['source_id']: s for s in result.get('sources', [])}
        if not result.get('claims'):
            issues.append('Missing claims')
        for claim in result.get('claims', []):
            if not claim.get('evidence'):
                issues.append('Missing evidence')
            for evidence in claim.get('evidence', []):
                source = sources.get(evidence.get('source_id'), {})
                quote = evidence.get('quote')
                if not quote or quote not in source.get('text', ''):
                    issues.append('Evidence is not an exact source span')
    return issues


def save_report(report, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(output)
    esc = lambda value: html.escape(str(value), quote=True)
    blocks = []
    for row in report['results']:
        result = row.get('response', {})
        sources = ''.join('<details><summary>' + esc(s['source_id']) + ' · ' + esc(s['page_title'])
                          + '</summary><p>' + esc(s['page_url']) + '</p><pre>' + esc(s['text'])
                          + '</pre></details>' for s in result.get('sources', []))
        blocks.append('<article><h2>' + esc(row['id']) + ' — ' + esc(row['question'])
                      + '</h2><p><b>קריטריון לבדיקה:</b> ' + esc(row['manual_review'])
                      + '</p><p><b>מצב:</b> ' + esc(result.get('status', row.get('error', '')))
                      + '</p><pre>' + esc(result.get('answer', '')) + '</pre>'
                      + '<p><b>בדיקת מבנה בלבד:</b> ' + esc(row.get('structural_issues', []))
                      + '</p><p><b>בדיקת נכונות אנושית: טרם בוצעה.</b></p>' + sources + '</article>')
    output.with_suffix('.html').write_text(
        '<!doctype html><html lang="he" dir="rtl"><meta charset="utf-8"><title>בדיקות הפניקס</title>'
        '<style>body{font:18px Arial;max-width:1000px;margin:30px auto;padding:20px;background:#f4f6fa}'
        'article{background:white;padding:24px;margin:20px 0;border:1px solid #ccd}pre{white-space:pre-wrap;'
        'overflow-wrap:anywhere;font:inherit}summary{cursor:pointer}</style><h1>דוח בדיקות שיחה</h1>'
        '<p>זהו תיעוד תשובות בפועל, ללא ציון איכות אוטומטי. הצלחת בקשה והתאמת ציטוט אינן הוכחה לנכונות.</p>'
        '<p>הושלמו ' + str(len(report['results'])) + ' מתוך ' + str(report['total']) + ' בקשות.</p>'
        + ''.join(blocks) + '</html>', encoding='utf-8')


def run(cases, report, output, call=request_json):
    completed = {row['id']: row for row in report['results']}
    for case in cases:
        if case['id'] in completed:
            continue
        history = []
        if case.get('parent'):
            parent = completed.get(case['parent'])
            if not parent or not parent.get('response'):
                raise ValueError('Follow-up parent was not completed')
            history = (parent['history'] + [
                {'role': 'user', 'content': parent['question']},
                {'role': 'assistant', 'content': parent['response']['answer']}])[-6:]
        row = {**case, 'history': history, 'human_review': 'pending'}
        print(f"{len(completed)+1}/{len(cases)} {case['id']}: requesting", flush=True)
        started = time.monotonic()
        try:
            result = call('/api/chat', {'question': case['question'], 'history': history})
            row['response'] = result
            row['structural_issues'] = structural_issues(result)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
            # No raw transport bodies, credentials or provider errors in the report.
            row['error'] = type(error).__name__
            if isinstance(error, HTTPError):
                row['http_status'] = error.code
            row['seconds'] = round(time.monotonic() - started, 2)
            report['results'].append(row)
            save_report(report, output)
            print('Stopped. Saved report; no automatic retry. The server may still be processing.', flush=True)
            return 1
        row['seconds'] = round(time.monotonic() - started, 2)
        report['results'].append(row)
        completed[case['id']] = row
        save_report(report, output)
        if row['structural_issues']:
            print('Stopped: response structure needs review.', flush=True)
            return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Send paid model requests through the local app')
    parser.add_argument('--resume', action='store_true', help='Continue after prior successful requests, same code/index only')
    args = parser.parse_args()
    cases_path = BASE / 'evaluation/chat_cases.json'
    cases = json.loads(cases_path.read_text(encoding='utf-8'))
    if not args.execute:
        print(f'{len(cases)} questions. No requests sent. --execute uses paid API via the running local app.')
        return 0
    output = BASE / 'evaluation/chat-run.json'
    if output.exists() and not args.resume:
        parser.error('Report already exists. Use --resume, or preserve it elsewhere before a fresh run.')
    signature = fingerprint(cases_path)
    if args.resume:
        if not output.exists():
            parser.error('No report to resume')
        report = json.loads(output.read_text(encoding='utf-8'))
        if report['fingerprint'] != signature:
            parser.error('Code, cases or active index changed. Preserve the old report before a new run.')
        if any(row.get('error') or row.get('structural_issues') for row in report['results']):
            parser.error('A failed request needs review first; resume will not silently retry or skip it.')
    else:
        report = {'created_at': datetime.now(timezone.utc).isoformat(), 'fingerprint': signature,
                  'total': len(cases), 'human_review': 'pending', 'results': []}
    health = request_json('/api/health')
    if not health.get('chat_ready'):
        parser.error('Local chat is not ready. No paid requests sent.')
    report['health'] = health
    save_report(report, output)
    try:
        return run(cases, report, output)
    except KeyboardInterrupt:
        print('Stopped. Completed responses saved. Wait for any in-flight request before resuming.')
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
