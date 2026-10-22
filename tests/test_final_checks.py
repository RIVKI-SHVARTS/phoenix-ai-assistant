"""Paid-run continuation bookkeeping; all HTTP responses are simulated."""
import copy
import json
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from urllib.error import HTTPError, URLError
from run_final_checks import CASES, collect, continuation


class ContinuationTests(unittest.TestCase):
    def report(self):
        return {'code_sha256': {'answering.py': 'old-version'}, 'results': [
            {'id': case_id, 'question': question,
             'state': 'request_failed' if case_id == '19' else 'response_received',
             **({'http_status': 503} if case_id == '19' else {'response': {'answer': 'saved'}})}
            for case_id, question, _ in CASES[:3]]}

    def test_completed_answers_are_not_resent_and_failed_attempt_is_retained(self):
        report = self.report()
        original = copy.deepcopy(report)
        continuation(report)
        report['code_sha256'] = {'answering.py': 'new-version'}
        sent = []
        def request(path, payload):
            sent.append(payload['question'])
            return {'status': 'insufficient', 'answer': 'test', 'claims': [], 'sources': []}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'report.json'
            self.assertEqual(collect(path, report, request), 0)
            saved = json.loads(path.read_text())
        self.assertEqual(sent, [q for _, q, _ in CASES[2:]])
        self.assertEqual(len(saved['results']), 8)
        self.assertEqual(saved['previous_attempts'][0]['id'], '19')
        self.assertEqual(saved['previous_attempts'][0]['http_status'], 503)
        self.assertEqual(saved['results'][0]['response'], original['results'][0]['response'])
        self.assertEqual(saved['results'][0]['code_sha256']['answering.py'], 'old-version')
        self.assertEqual(saved['results'][2]['code_sha256']['answering.py'], 'new-version')

    def test_ambiguous_inflight_request_is_not_silently_retried(self):
        report = self.report()
        report['results'][-1]['state'] = 'request_started'
        with self.assertRaises(ValueError):
            continuation(report)

    def test_unmatched_question_is_not_silently_skipped(self):
        report = self.report()
        report['results'][0]['question'] = 'different question'
        with self.assertRaises(ValueError):
            continuation(report)

    def test_finish_keeps_failed_case_and_never_resends_it(self):
        report = self.report()
        failed = copy.deepcopy(report['results'][2])
        continuation(report, retry_failures=False)
        sent = []
        def request(path, payload):
            sent.append(payload['question'])
            return {'status': 'insufficient', 'answer': 'test'}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'report.json'
            self.assertEqual(collect(path, report, request, finish_remaining=True), 0)
            saved = json.loads(path.read_text())
        self.assertEqual(sent, [q for _, q, _ in CASES[3:]])
        self.assertEqual(len(saved['results']), 8)
        self.assertEqual(saved['results'][2]['state'], failed['state'])
        self.assertEqual(saved['results'][2]['http_status'], 503)
        self.assertEqual(saved['state'], 'collected_with_failures_pending_review')

    def test_quality_failure_is_recorded_and_next_case_runs_without_retry(self):
        report = self.report()
        continuation(report, retry_failures=False)
        sent = []
        def request(path, payload):
            sent.append(payload['question'])
            if len(sent) == 1:
                body = json.dumps({'error': 'ניסוח התשובה לא עבר את בדיקת האיכות.'}, ensure_ascii=False).encode()
                raise HTTPError('http://127.0.0.1:5000/api/chat', 503, 'quality', {}, BytesIO(body))
            return {'status': 'insufficient', 'answer': 'test'}
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(collect(Path(tmp) / 'report.json', report, request, finish_remaining=True), 0)
        self.assertEqual(len(sent), 5)
        self.assertEqual(report['results'][3]['state'], 'request_failed')
        self.assertEqual(report['results'][4]['state'], 'response_received')
        self.assertEqual(report['state'], 'collected_with_failures_pending_review')

    def test_network_failure_still_stops_finish_mode(self):
        report = self.report()
        continuation(report, retry_failures=False)
        sent = []
        def request(path, payload):
            sent.append(payload)
            raise URLError('connection failed')
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(collect(Path(tmp) / 'report.json', report, request, finish_remaining=True), 1)
        self.assertEqual(len(sent), 1)
        self.assertEqual(report['state'], 'stopped_for_review')
