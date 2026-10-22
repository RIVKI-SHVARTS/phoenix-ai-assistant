"""OpenAI embeddings with explicit token budget and no hidden SDK retries."""

from contextlib import closing
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import uuid
from datetime import datetime, timezone

from chunking import token_count

MODELS = {
    "text-embedding-3-large": {"dimensions": 3072, "usd_per_million": 0.13},
    "text-embedding-3-small": {"dimensions": 1536, "usd_per_million": 0.02},
}
PRICE_CHECKED = "2026-10-15"


class BudgetExceeded(RuntimeError):
    pass


class TokenBudget:
    """Shared persistent ceiling for this app's embedding tokens (not all API spend)."""

    def __init__(self, path, limit=None):
        if limit is None:
            config = Path(path).parent / 'embedding-budget.json'
            limit = json.loads(config.read_text(encoding='utf-8'))['token_limit'] if config.exists() else 1000000
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("Token limit must be positive")
        self.path, self.limit = Path(path), limit
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS usage (id TEXT PRIMARY KEY, reserved INTEGER, actual INTEGER, status TEXT, created_at TEXT)"
            )

    def reserve(self, tokens):
        if not isinstance(tokens, int) or tokens <= 0:
            raise ValueError("Token reservation must be a positive integer")
        key = uuid.uuid4().hex
        with closing(sqlite3.connect(self.path, timeout=10)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            used = db.execute("SELECT COALESCE(SUM(reserved),0) FROM usage").fetchone()[
                0
            ]
            if used + tokens > self.limit:
                raise BudgetExceeded("Embedding token budget reached")
            db.execute(
                "INSERT INTO usage VALUES (?,?,NULL,?,?)",
                (key, tokens, "reserved", datetime.now(timezone.utc).isoformat()),
            )
        return key

    def complete(self, key, tokens):
        # Failed/ambiguous requests remain reserved: do not assume they were free.
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute(
                "UPDATE usage SET actual=?, reserved=MAX(reserved,?), status=? WHERE id=?",
                (tokens, tokens, "completed", key),
            )


class OpenAIEmbeddings:
    def __init__(self, model="text-embedding-3-large", budget=None, api_client=None):
        if model not in MODELS:
            raise ValueError(
                "Unsupported model; explicitly update model metadata first"
            )
        if budget is None:
            raise ValueError("A persistent token budget is required")
        self.model, self.dimensions, self.budget = (
            model,
            MODELS[model]["dimensions"],
            budget,
        )
        if api_client is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY is missing. Configure it locally; do not paste it into chat."
                )
            from openai import OpenAI

            api_client = OpenAI(
                base_url="https://api.openai.com/v1", max_retries=0, timeout=45.0
            )
        self.client = api_client
        self.cache = budget.path.parent / "embedding-cache.sqlite3"
        with closing(sqlite3.connect(self.cache)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector TEXT NOT NULL)")

    def _key(self, text):
        return hashlib.sha256(json.dumps([self.model, self.dimensions, text], ensure_ascii=False).encode("utf-8")).hexdigest()

    def _validate(self, vector):
        if (len(vector) != self.dimensions
                or any(not math.isfinite(x) for x in vector) or not any(vector)):
            raise RuntimeError("Invalid embedding vector")

    def embed(self, texts):
        if (
            not texts
            or len(texts) > 32
            or any(not isinstance(t, str) or not t.strip() for t in texts)
        ):
            raise ValueError("Expected 1–32 non-empty texts")
        counts = [token_count(t) for t in texts]
        if max(counts) > 8000 or sum(counts) > 100000:
            raise ValueError("Embedding input limit exceeded")
        keys = [self._key(text) for text in texts]
        found, missing = {}, {}
        with closing(sqlite3.connect(self.cache)) as db:
            for key, text in zip(keys, texts):
                row = db.execute("SELECT vector FROM embeddings WHERE key=?", (key,)).fetchone()
                if row:
                    vector = json.loads(row[0])
                    self._validate(vector)
                    found[key] = vector
                else:
                    missing[key] = text
        if missing:
            reservation = self.budget.reserve(sum(token_count(t) for t in missing.values()))
            response = self.client.embeddings.create(
                model=self.model, input=list(missing.values()), encoding_format="float"
            )
            data = sorted(response.data, key=lambda row: row.index)
            if [row.index for row in data] != list(range(len(missing))):
                raise RuntimeError("Embedding response indices do not match input")
            vectors = [row.embedding for row in data]
            for vector in vectors:
                self._validate(vector)
            # Commit successful API results before downstream vector insertion.
            with closing(sqlite3.connect(self.cache)) as db, db:
                db.executemany("INSERT OR REPLACE INTO embeddings VALUES (?,?)",
                    [(key, json.dumps(vector, allow_nan=False)) for key, vector in zip(missing, vectors)])
            found.update(zip(missing, vectors))
            self.budget.complete(reservation, response.usage.total_tokens)
        return [found[key] for key in keys]
