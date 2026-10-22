"""Replay only the last rejected review, with a structured explanation.

Run next to app.py and answer-diagnostic.json. One paid model call; no retrieval
or draft generation, no index rebuild, no automatic retries.
"""
import ast
import inspect
import json
from pathlib import Path
import uuid

from answering import AnswerService, StrictModel
from retrieval import SearchIndex


class ExplainedReview(StrictModel):
    approved: bool
    reason: str
    unsupported_claims: list[str]
    missing_required_information: list[str]


def review_payload(report, index):
    events = report["events"]
    retrieved = next(e for e in events if e["stage"] == "retrieval")
    draft = next(e for e in events if e["stage"] == "Draft")
    verdict = next(e for e in events if e["stage"] == "Review")
    if verdict["parsed"]["approved"] is not False:
        raise ValueError("The last diagnostic was not a rejected review")
    if index.manifest.get("generation") != report["index"]["generation"]:
        raise ValueError("The active index changed since the diagnostic")
    by_chunk = {row["chunk_id"]: row for row in index.chunks}
    recorded = {row["chunk_id"]: row for row in retrieved["candidates"]}
    sources = []
    for mapping in draft["source_map"]:
        row = by_chunk[mapping["chunk_id"]]
        previous = recorded[mapping["chunk_id"]]
        if any(row[k] != previous[k] for k in ("text", "heading", "page_url")):
            raise ValueError("The source changed since the diagnostic")
        sources.append({"source_id": mapping["source_id"], **{k: row[k] for k in
            ("chunk_id", "page_url", "page_title", "heading", "text", "fetched_at")}})
    return {"question": report["question"], "resolved_question": report["question"],
            "sources": sources, "proposed": draft["parsed"]}


def review_instructions():
    # Reuse the installed review rules rather than silently changing the standard.
    tree = ast.parse(inspect.getsource(AnswerService))
    matches = [node.value for node in ast.walk(tree)
               if isinstance(node, ast.Constant) and isinstance(node.value, str)
               and node.value.startswith("You are a strict grounding reviewer.")]
    if len(matches) != 1:
        raise ValueError("Cannot identify the installed reviewer instructions")
    return matches[0] + "\nExplain your verdict in Hebrew in reason. Identify exact unsupported claims and missing required information in their respective arrays; use empty arrays when none. Distinguish an actual factual/scope error from optional stylistic improvements. Do not treat the previous verdict as evidence."


def main():
    root = Path(__file__).resolve().parent
    try:
        report = json.loads((root / "answer-diagnostic.json").read_text(encoding="utf-8-sig"))
        index = SearchIndex(root / "data/index")
        payload = review_payload(report, index)
        instructions = review_instructions()
        service = AnswerService(root / "data/index")
    except Exception as error:
        print("Setup failed:", type(error).__name__)
        print("Keep this script next to app.py and the latest answer-diagnostic.json.")
        return 1
    print("Running ONE review call; normal API charges apply. Do not repeat automatically.", flush=True)
    try:
        review = service._call(instructions, payload, ExplainedReview, 1200)
        output = {"diagnostic_version": 1, "question": report["question"],
                  "original_review": {"approved": False}, "explained_review": review.model_dump(),
                  "notice": "A new review call may disagree with the original; this is not the original review's hidden reasoning."}
    except Exception as error:
        output = {"diagnostic_version": 1, "error_type": type(error).__name__}
    destination = root / "review-diagnostic.json"
    temporary = root / ("review-diagnostic-" + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)
    print("Saved review-diagnostic.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
