"""New index generation; publish the pointer only after successful validation."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid
from chunking import make_chunks
from embedding_provider import MODELS, PRICE_CHECKED, OpenAIEmbeddings, TokenBudget


def dump(path, data):
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def prepare(pages_file, model="text-embedding-3-large"):
    chunks, stats = make_chunks(pages_file)
    price = MODELS[model]["usd_per_million"]
    return chunks, dict(
        stats,
        model=model,
        estimated_embedding_usd=round(stats["embedding_tokens"] * price / 1000000, 8),
        price_checked=PRICE_CHECKED,
        price_usd_per_million=price,
    )


def build_index(
    pages_file, root, mode="lexical", model="text-embedding-3-large", provider=None
):
    if mode not in ("lexical", "semantic"):
        raise ValueError("Unknown index mode")
    chunks, stats = make_chunks(pages_file)
    if mode == "semantic":
        provider = provider or OpenAIEmbeddings(
            model, TokenBudget(Path(root) / "embedding-usage.sqlite3")
        )
        if provider.model != model:
            raise ValueError("Provider/model mismatch")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    generation = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    out = root / generation
    out.mkdir()
    manifest = dict(
        stats,
        generation=generation,
        mode=mode,
        status="building",
        created_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=hashlib.sha256(Path(pages_file).read_bytes()).hexdigest(),
        model=model if mode == "semantic" else None,
        dimensions=provider.dimensions if mode == "semantic" else None,
        semantic_quality_verified=False,
        vector_backend="sqlite-numpy-v1" if mode == "semantic" else None,
    )
    dump(out / "manifest.json", manifest)
    try:
        chunks_file = out / "chunks.jsonl"
        chunks_file.write_text(
            "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks),
            encoding="utf-8",
        )
        manifest["chunks_sha256"] = hashlib.sha256(chunks_file.read_bytes()).hexdigest()
        if mode == "semantic":
            from vector_store import VectorStore

            store = VectorStore(out / "vectors", model, provider.dimensions, create=True)
            for start in range(0, len(chunks), 32):
                batch = chunks[start : start + 32]
                print(f"Embedding/storing chunks {start + 1}-{start + len(batch)} of {len(chunks)}", flush=True)
                store.add(batch, provider.embed([c["embedding_text"] for c in batch]))
            if store.count() != len(chunks):
                raise RuntimeError("Stored chunk count mismatch")
        manifest["status"] = "ready"
        dump(out / "manifest.json", manifest)
        pointer = root / ("CURRENT." + uuid.uuid4().hex + ".tmp")
        dump(pointer, {"generation": generation})
        os.replace(pointer, root / "CURRENT.json")
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error_type"] = type(error).__name__
        dump(out / "manifest.json", manifest)
        raise
    return out, manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("pages_file", type=Path)
    p.add_argument("--mode", choices=["lexical", "semantic"], default="lexical")
    p.add_argument("--model", choices=list(MODELS), default="text-embedding-3-large")
    p.add_argument("--output", type=Path, default=Path("data/index"))
    p.add_argument(
        "--execute",
        action="store_true",
        help="Call embedding API in semantic mode; otherwise estimate only",
    )
    a = p.parse_args()
    if a.mode == "semantic" and not a.execute:
        _, stats = prepare(a.pages_file, a.model)
        print(json.dumps(dict(stats, dry_run=True), ensure_ascii=False, indent=2))
    else:
        try:
            out, m = build_index(a.pages_file, a.output, a.mode, a.model)
            print(
                json.dumps(
                    {
                        "index": str(out),
                        "mode": m["mode"],
                        "chunks": m["chunks"],
                        "status": m["status"],
                    },
                    ensure_ascii=False,
                )
            )
        except (RuntimeError, ValueError) as e:
            p.exit(1, str(e) + "\n")
