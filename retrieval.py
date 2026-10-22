"""Lexical/semantic/hybrid retrieval. Returns evidence candidates, never an answer."""

import argparse
import hashlib
import json
from pathlib import Path
import re
from lexical_search import LexicalSearch


def merge_rankings(lexical, semantic, limit=6):
    scores = {}
    for ranking in (lexical, semantic):
        for rank, key in enumerate(ranking, 1):
            scores[key] = scores.get(key, 0) + 1 / (60 + rank)
    return sorted(scores, key=lambda k: (-scores[k], k))[:limit]


class SearchIndex:
    def __init__(self, root, provider=None):
        root = Path(root)
        gen = json.loads((root / "CURRENT.json").read_text(encoding="utf-8"))[
            "generation"
        ]
        if not re.fullmatch(r"[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}", gen):
            raise ValueError("Invalid index pointer")
        self.path = root / gen
        self.manifest = json.loads(
            (self.path / "manifest.json").read_text(encoding="utf-8")
        )
        if self.manifest["status"] != "ready":
            raise ValueError("Index not ready")
        digest = hashlib.sha256()
        with (self.path / "chunks.jsonl").open('rb') as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        if digest.hexdigest() != self.manifest["chunks_sha256"]:
            raise ValueError("Index hash mismatch")
        with (self.path / "chunks.jsonl").open(encoding='utf-8') as stream:
            self.chunks = [json.loads(line) for line in stream]
        self.by_id = {c["chunk_id"]: c for c in self.chunks}
        if len(self.by_id) != self.manifest["chunks"]:
            raise ValueError("Chunk count mismatch")
        self.lexical = LexicalSearch(self.chunks)
        self.provider = provider
        self.root = root
        self.store = None

    def search(self, question, mode="lexical", limit=6):
        if (
            not isinstance(question, str)
            or not question.strip()
            or len(question) > 2000
        ):
            raise ValueError("Question must contain 1–2000 characters")
        if mode not in ("lexical", "semantic", "hybrid") or not 1 <= limit <= 20:
            raise ValueError("Invalid search options")
        lexical = (
            self.lexical.search(question, min(30, max(limit, 12)))
            if mode != "semantic"
            else []
        )
        if mode == "lexical":
            return {
                "mode": "lexical",
                "semantic_verified": False,
                "answer_generated": False,
                "results": lexical[:limit],
            }
        if self.manifest["mode"] != "semantic":
            raise ValueError(
                "Only a lexical baseline exists. Build a semantic index first; no silent fallback."
            )
        if self.manifest.get("vector_backend") != "sqlite-numpy-v1":
            raise ValueError("Old vector backend; rebuild the semantic index with updated ingest.py")
        if self.provider is None:
            from embedding_provider import OpenAIEmbeddings, TokenBudget

            self.provider = OpenAIEmbeddings(
                self.manifest["model"],
                TokenBudget(self.root / "embedding-usage.sqlite3"),
            )
        if (
            self.provider.model != self.manifest["model"]
            or self.provider.dimensions != self.manifest["dimensions"]
        ):
            raise ValueError("Query/index embedding model mismatch")
        if self.store is None:
            from vector_store import VectorStore

            self.store = VectorStore(
                self.path / "vectors", self.provider.model, self.provider.dimensions
            )
        vector = self.provider.embed([question])[0]
        semantic = self.store.query(vector, min(30, max(limit, 12)))
        ids = [row[0] for row in semantic]
        if mode == "hybrid":
            ids = merge_rankings([r["chunk_id"] for r in lexical], ids, limit)
        distances = dict(semantic)
        results = [
            dict(
                self.by_id[key], retrieval_mode=mode, cosine_distance=distances.get(key)
            )
            for key in ids[:limit]
        ]
        return {
            "mode": mode,
            "semantic_verified": self.manifest["semantic_quality_verified"],
            "answer_generated": False,
            "notice": "Ranked candidates are not confidence or proof an answer exists.",
            "results": results,
        }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--index", type=Path, default=Path("data/index"))
    p.add_argument(
        "--mode", choices=["lexical", "semantic", "hybrid"], default="lexical"
    )
    p.add_argument("--limit", type=int, default=6)
    a = p.parse_args()
    try:
        print(
            json.dumps(
                SearchIndex(a.index).search(a.question, a.mode, a.limit),
                ensure_ascii=False,
                indent=2,
            )
        )
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        p.exit(1, str(e) + "\n")
