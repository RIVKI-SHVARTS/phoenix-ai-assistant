r"""One fixed public-source diagnostic. No API keys or provider error bodies logged.

Run next to app.py: .venv\Scripts\python diagnose_answer.py
This repeats the answer pipeline once and incurs normal API charges.
It does not rebuild or change the active index. Normal cache/usage journals update.
"""
import json
from pathlib import Path
import sys
import uuid

QUESTION = "אילו מסמכים צריך לצרף לתביעה על גניבת רכוש מהדירה?"


def diagnose(index, service):
    report = {
        "diagnostic_version": 1,
        "question": QUESTION,
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
        result = service.answer(index, QUESTION, [])
        report["result"] = result
    except Exception as error:
        # Provider exceptions may contain sensitive bodies: record the class only.
        report["error_type"] = type(error).__name__
    finally:
        service._call, service._abstain, index.search = original_call, original_abstain, original_search
    return report


def main():
    from answering import AnswerService
    from retrieval import SearchIndex
    base = Path(__file__).resolve().parent
    root = base / "data/index"
    try:
        index = SearchIndex(root)
        if index.manifest.get("mode") != "semantic":
            print("Diagnostic stopped: active index is not semantic. No answer API call made.")
            return 1
        service = AnswerService(root)
    except Exception as error:
        print("Diagnostic setup failed:", type(error).__name__)
        return 1
    print("Running one answer diagnostic; normal API charges apply. Do not repeat automatically.", flush=True)
    report = diagnose(index, service)
    path = base / "answer-diagnostic.json"
    temporary = base / ("answer-diagnostic-" + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    print("Saved: answer-diagnostic.json", flush=True)
    print("Outcome:", report.get("result", {}).get("status", report.get("error_type", "unknown")), flush=True)
    print("Stages:", ", ".join(row["stage"] for row in report["events"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
