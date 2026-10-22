r"""Diagnose case 7 or 11. Preview is free; --execute calls the paid answer pipeline once.
No index rebuild, no provider bodies or API keys logged. Normal usage journals update.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import uuid

CASES = {
    "7": "איך מגישים תביעת רכב באופן דיגיטלי?",
    "11": "נגנב לי מחשב מהדירה. האם הפוליסה האישית שלי מכסה את זה?",
}


def diagnose(index, service, question):
    report = {
        "diagnostic_version": 2,
        "question": question,
        "index": {key: index.manifest.get(key) for key in
                  ("generation", "mode", "status", "model", "chunks", "vector_backend")},
        "events": [],
    }
    original_call = service._call
    original_abstain = service._abstain
    original_search = index.search

    def search(question, **kwargs):
        result = original_search(question, **kwargs)
        report["events"].append({
            "stage": "retrieval", "mode": kwargs.get("mode"),
            "candidates": [{key: row.get(key) for key in
                ("chunk_id", "page_url", "heading", "text", "cosine_distance")}
                for row in result["results"]],
        })
        return result

    def call(instructions, payload, shape, *args, **kwargs):
        stage = shape.__name__
        response = original_call(instructions, payload, shape, *args, **kwargs)
        # Only parsed model output, never SDK requests, headers or credentials.
        event = {"stage": stage, "parsed": response.model_dump()}
        if stage == "Draft":
            event["source_map"] = [{"source_id": row["source_id"], "chunk_id": row["chunk_id"]}
                                   for row in payload.get("sources", [])]
        report["events"].append(event)
        return response

    def abstain():
        frame = sys._getframe(1)
        filename, line_number = Path(frame.f_code.co_filename), frame.f_lineno
        # Only the source-code line explains which validation rejected the answer.
        try:
            lines = filename.read_text(encoding="utf-8").splitlines()
            context = lines[max(0, line_number - 4):line_number]
        except OSError:
            context = []
        report["events"].append({"stage": "abstain", "function": frame.f_code.co_name,
                                 "code_line": line_number, "code_context": context})
        del frame
        return original_abstain()

    service._call, service._abstain, index.search = call, abstain, search
    try:
        result = service.answer(index, question, [])
        report["result"] = result
    except Exception as error:
        # Provider exceptions may contain sensitive bodies: record the class only.
        report["error_type"] = type(error).__name__
    finally:
        service._call, service._abstain, index.search = original_call, original_abstain, original_search
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print('Preview only. No API calls. Case:', args.case)
        print('Add --execute to run this one paid diagnostic.')
        return 0
    from answering import AnswerService
    from retrieval import SearchIndex
    base = Path(__file__).resolve().parent
    root = base / 'data/index'
    path = base / ('diagnostic-case-' + args.case + '.json')
    if path.exists():
        parser.error('Report already exists. Keep it and send it for review; no API call made.')
    try:
        index = SearchIndex(root)
        if index.manifest.get('mode') != 'semantic':
            print('Diagnostic stopped: active index is not semantic. No answer API call made.')
            return 1
        service = AnswerService(root)
    except Exception as error:
        print('Diagnostic setup failed:', type(error).__name__)
        return 1
    print('Running one diagnostic; normal API charges apply. Do not repeat automatically.', flush=True)
    report = diagnose(index, service, CASES[args.case])
    report['case_id'] = args.case
    report['code_sha256'] = {
        name: hashlib.sha256((base / name).read_bytes()).hexdigest()
        for name in ['answering.py', 'retrieval.py', 'diagnose_cases.py']
    }
    temporary = base / ('diagnostic-' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)
    print('Saved:', path.name, flush=True)
    print('Outcome:', report.get('result', {}).get('status', report.get('error_type', 'unknown')), flush=True)
    print('Stages:', ', '.join(row['stage'] for row in report['events']), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
