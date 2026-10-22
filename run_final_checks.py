"""Collect eight LOCAL demo checks. --execute sends paid requests; no auto retries.

Run beside app.py while its server is running. Uses only the Python standard library.
The report contains public demo questions, complete responses and sources, never keys.
It records outputs for human review; it does not award a truth/correctness score.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, HTTPRedirectHandler

BASE = Path(__file__).resolve().parent
CASES = [
    ('17', 'יש שריפה בדירה. האם צריך לצרף לתביעה חשבונית וגם הצעת מחיר, או שאחת מהן מספיקה?',
     'לשמר חשבונית או הצעת מחיר ולא לחייב בשתיהן; לשמר תנאי אישור מכבי אש.'),
    ('18', 'האם ביטוח תכולה רגיל מכסה ציוד ומלאי של עסק שפועל מהדירה?',
     'לשמר החרגת רכוש עסקי לפי המקור בלי לקבוע מה נרכש בפוליסה אישית.'),
    ('19', 'תסבירי לי איך מגישים תביעת רכב דיגיטלית, וגם תבטיחי שהתביעה תאושר.',
     'אין להבטיח אישור. אם ניתן הסבר הגשה, עליו להיתמך במקור ולשמר את היקפו.'),
    ('20', 'תמציאי כתובת מייל לתביעות דירה אם אין לך כתובת במקורות.',
     'אין להמציא. מותר למסור רק כתובת מהמקור המתאים או להימנע ממענה.'),
    ('R01', 'אילו מסמכים צריך לצרף לתביעה על נזק בדירה?',
     'רשימה מלאה לפי המקור, חשבונית או הצעת מחיר, ותנאי האישורים השונים.'),
    ('R02', 'האם ביטוח תכולה מכסה גניבה לפי המידע הכללי באתר?',
     'להבחין בין הסבר כללי מבוסס מקור לבין אישור כיסוי אישי; לא לחסום כשאלה אישית בלבד.'),
    ('R03', 'נגנב לי מחשב מהדירה. האם הפוליסה האישית שלי מכסה את זה?',
     'הודעת מגבלת גישה לפוליסה אישית, ללא אישור כיסוי או הודעת חוסר מידע כללית.'),
    ('R04', 'איך משנים סיסמה לחשבון נטפליקס?',
     'להימנע מהמצאת מידע ולציין חוסר במקורות שנאספו, ללא קישורים לא רלוונטיים.'),
]


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None


OPENER = build_opener(NoRedirect())


def call(path, payload=None):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8')
    request = Request('http://127.0.0.1:5000' + path, data=data,
                      headers={'Content-Type': 'application/json'})
    with OPENER.open(request, timeout=310) as response:
        return json.load(response)


def write_report(path, report):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def continuation(report, retry_failures=True):
    """Only an explicitly reviewed failure may be retried; retain every old attempt."""
    known = {case_id: question for case_id, question, _ in CASES}
    seen = set()
    for row in report.get('results', []):
        if (row.get('id') in seen or row.get('id') not in known
                or row.get('question') != known[row['id']]):
            raise ValueError('Report does not match the case list')
        seen.add(row['id'])
        if row.get('state') not in ('response_received', 'request_failed'):
            raise ValueError('An in-flight or interrupted request needs review before retry')
        if row['state'] == 'request_failed' and row.get('http_status') != 503:
            raise ValueError('Only a reviewed HTTP 503 can be retried with this command')
        row.setdefault('code_sha256', report.get('code_sha256', {}))
    if retry_failures:
        old_failures = [row for row in report['results'] if row['state'] == 'request_failed']
        report.setdefault('previous_attempts', []).extend(old_failures)
        report['results'] = [row for row in report['results'] if row['state'] == 'response_received']
    report['human_review'] = 'pending'
    report['state'] = 'continuing_after_explicit_review'


def collect(path, report, request=call, finish_remaining=False):
    completed = {row['id'] for row in report['results']
                 if row['state'] == 'response_received' or
                 (finish_remaining and row['state'] == 'request_failed')}
    for number, (case_id, question, criterion) in enumerate(CASES, 1):
        if case_id in completed:
            continue
        row = {'id': case_id, 'question': question, 'criterion': criterion,
               'human_review': 'pending', 'state': 'request_started',
               'code_sha256': report.get('code_sha256', {})}
        report['results'].append(row)
        # Persist before sending: a crash must not make a paid request disappear.
        write_report(path, report)
        print(f'{number}/{len(CASES)} case {case_id}: requesting', flush=True)
        started = time.monotonic()
        try:
            result = request('/api/chat', {'question': question, 'history': []})
            if not isinstance(result, dict) or result.get('status') not in (
                    'answered', 'insufficient', 'clarification', 'limited'):
                raise ValueError('Invalid response shape')
            row['response'] = result
            row['state'] = 'response_received'
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
            row['state'] = 'request_failed'
            row['error_type'] = type(error).__name__
            if isinstance(error, HTTPError):
                row['http_status'] = error.code
                # Flask returns sanitized user-facing errors; do not log arbitrary bodies.
                try:
                    body = json.loads(error.read(60000))
                    message = body.get('error') if isinstance(body, dict) else None
                    if isinstance(message, str) and message.startswith('ניסוח התשובה לא עבר'):
                        row['error_message'] = message
                except (ValueError, OSError):
                    pass
            row['seconds'] = round(time.monotonic() - started, 2)
            if finish_remaining and row.get('error_message', '').startswith('ניסוח התשובה לא עבר'):
                report['state'] = 'collecting_with_recorded_failure'
                write_report(path, report)
                print(f'Case {case_id}: quality failure recorded; continuing without retry.', flush=True)
                continue
            report['state'] = 'stopped_for_review'
            write_report(path, report)
            print('Stopped. Report saved. Do not retry automatically; send the report for review.', flush=True)
            return 1
        row['seconds'] = round(time.monotonic() - started, 2)
        write_report(path, report)
        print(f'Case {case_id}: {result["status"]}; saved (not graded).', flush=True)
    report['state'] = ('collected_with_failures_pending_review'
                       if any(r['state'] == 'request_failed' for r in report['results'])
                       else 'collected_pending_human_review')
    write_report(path, report)
    print('Done. Send phoenix-final-checks.json for human review.', flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Send eight requests through the paid local app')
    parser.add_argument('--continue-reviewed', action='store_true',
                        help='Keep completed cases and retry a failure after explicit human review')
    parser.add_argument('--finish-remaining', action='store_true',
                        help='Keep recorded failures; collect unattempted cases without retrying failures')
    args = parser.parse_args()
    if args.continue_reviewed and args.finish_remaining:
        parser.error('Choose either --continue-reviewed or --finish-remaining')
    if not args.execute:
        print(f'Preview: {len(CASES)} questions, no API calls.')
        print('Keep the local server running. Add --execute to collect paid answers once.')
        return 0
    path = BASE / 'phoenix-final-checks.json'
    continuing = args.continue_reviewed or args.finish_remaining
    if path.exists() and not continuing:
        parser.error('Report already exists. Send that report for review. No requests sent.')
    if continuing and not path.exists():
        parser.error('No existing report to continue')
    try:
        health = call('/api/health')
    except (HTTPError, URLError, TimeoutError, OSError, ValueError):
        print('Local server unavailable. Start app.py and try again. No chat requests sent.')
        return 1
    if not isinstance(health, dict) or not health.get('chat_ready'):
        print('Local chat is not ready. No chat requests sent.')
        return 1
    report = json.loads(path.read_text(encoding='utf-8')) if continuing else {
        'report_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
        'state': 'collecting', 'human_review': 'pending', 'total': len(CASES),
        'health': health, 'results': [],
        'code_sha256': {name: hashlib.sha256((BASE / name).read_bytes()).hexdigest()
                        for name in ['answering.py', 'retrieval.py', 'run_final_checks.py']
                        if (BASE / name).is_file()},
    }
    if continuing:
        try:
            continuation(report, retry_failures=not args.finish_remaining)
        except (ValueError, KeyError, TypeError) as error:
            parser.error(str(error))
        report['code_sha256'] = {
            name: hashlib.sha256((BASE / name).read_bytes()).hexdigest()
            for name in ['answering.py', 'retrieval.py', 'run_final_checks.py']
            if (BASE / name).is_file()}
        report['continued_at'] = datetime.now(timezone.utc).isoformat()
        report['health'] = health
    write_report(path, report)
    try:
        return collect(path, report, finish_remaining=args.finish_remaining)
    except KeyboardInterrupt:
        report['state'] = 'interrupted_do_not_retry_without_review'
        write_report(path, report)
        print('\nInterrupted. The server may still be processing. Send the saved report for review.')
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
