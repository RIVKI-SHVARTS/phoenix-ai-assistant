"""Evidence retrieval evaluation, deliberately separate from answer quality."""

import argparse
import json
from pathlib import Path
import re
from retrieval import SearchIndex


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def evaluate(index, cases, mode="lexical", k=6):
    search = SearchIndex(index)
    rows = []
    for case in cases:
        r = search.search(case["question"], mode, limit=k)["results"]
        row = dict(
            case,
            candidates=[
                {
                    "chunk_id": c["chunk_id"],
                    "url": c["page_url"],
                    "heading": c["heading"],
                    "snippet": c["text"][:180],
                }
                for c in r
            ],
        )
        if case.get("expected_url"):
            matching = [c for c in r if c["page_url"] == case["expected_url"]]
            text = norm("\n".join(c["text"] for c in matching))
            row["source_hit"] = bool(matching)
            row["evidence_hit"] = bool(matching) and all(
                norm(e) in text for e in case["evidence_all"]
            )
            row["source_rank"] = next(
                (
                    i
                    for i, c in enumerate(r, 1)
                    if c["page_url"] == case["expected_url"]
                ),
                None,
            )
        else:
            row["candidate_returned"] = bool(r)
            row["answerability_evaluated"] = False
        rows.append(row)
    positives = [r for r in rows if "evidence_hit" in r]
    return {
        "mode": mode,
        "k": k,
        "answer_generation_tested": False,
        "semantic_quality_tested": mode != "lexical",
        "positive_cases": len(positives),
        "evidence_hits": sum(r["evidence_hit"] for r in positives),
        "source_hits": sum(r["source_hit"] for r in positives),
        "unsupported_cases": len(rows) - len(positives),
        "unsupported_with_candidates": sum(
            r.get("candidate_returned", False) for r in rows
        ),
        "results": rows,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=Path, default=Path("data/index"))
    p.add_argument(
        "--cases", type=Path, default=Path("evaluation/retrieval_cases.json")
    )
    p.add_argument(
        "--mode", choices=["lexical", "semantic", "hybrid"], default="lexical"
    )
    p.add_argument(
        "--output", type=Path, default=Path("evaluation/retrieval-results.json")
    )
    a = p.parse_args()
    r = evaluate(a.index, json.loads(a.cases.read_text(encoding="utf-8")), a.mode)
    a.output.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in r.items() if k != "results"}, ensure_ascii=False, indent=2
        )
    )
