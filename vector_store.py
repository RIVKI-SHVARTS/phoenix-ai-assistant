"""Persistent exact cosine search using SQLite and NumPy; no model downloads."""

from contextlib import closing
import heapq
from pathlib import Path
import sqlite3
import numpy as np

BACKEND = "sqlite-numpy-v1"


class VectorStore:
    def __init__(self, path, model, dimensions, create=False):
        self.path = Path(path) / "vectors.sqlite3"
        self.dimensions = dimensions
        metadata = {"backend": BACKEND, "model": model, "dimensions": str(dimensions)}
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                raise ValueError("Vector database already exists")
            with closing(sqlite3.connect(self.path)) as db, db:
                db.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                db.executemany("INSERT INTO metadata VALUES (?,?)", metadata.items())
                db.execute("CREATE TABLE vectors (id TEXT PRIMARY KEY, vector BLOB NOT NULL)")
        elif not self.path.is_file():
            raise ValueError("Vector database missing; rebuild the semantic index with the updated ingest.py")
        with closing(sqlite3.connect(self.path)) as db:
            if dict(db.execute("SELECT key,value FROM metadata")) != metadata:
                raise ValueError("Embedding model/dimensions/backend mismatch")

    def _normalize(self, vector):
        vector = np.asarray(vector, dtype=np.float64)
        if vector.shape != (self.dimensions,) or not np.isfinite(vector).all():
            raise ValueError("Invalid embedding shape or values")
        scale = np.max(np.abs(vector))
        if scale == 0:
            raise ValueError("Zero embedding vector")
        vector = vector / scale
        return (vector / np.linalg.norm(vector)).astype("<f4")

    def add(self, chunks, embeddings):
        if len(chunks) != len(embeddings):
            raise ValueError("Embedding shape mismatch")
        rows = [(c["chunk_id"], self._normalize(v).tobytes())
                for c, v in zip(chunks, embeddings)]
        with closing(sqlite3.connect(self.path)) as db, db:
            db.executemany("INSERT INTO vectors VALUES (?,?)", rows)

    def count(self):
        with closing(sqlite3.connect(self.path)) as db:
            return db.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]

    def query(self, vector, limit):
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("Query limit must be positive")
        query = self._normalize(vector)
        best = []
        with closing(sqlite3.connect(self.path)) as db:
            cursor = db.execute("SELECT id,vector FROM vectors ORDER BY id")
            while rows := cursor.fetchmany(256):
                if any(len(blob) != self.dimensions * 4 for _, blob in rows):
                    raise ValueError("Corrupt stored vector")
                matrix = np.frombuffer(b"".join(blob for _, blob in rows), dtype="<f4").reshape(len(rows), self.dimensions)
                if not np.isfinite(matrix).all():
                    raise ValueError("Corrupt stored vector")
                # Bounded exact search without starting BLAS worker threads for
                # every small batch on a shared CPU. Keep the full candidate set.
                scores = np.einsum("ij,j->i", matrix, query, optimize=False)
                distances = np.clip(1 - scores, 0, 2)
                best = heapq.nsmallest(limit, best + [
                    (float(distance), key)
                    for (key, _), distance in zip(rows, distances)
                ])
        return [(key, distance) for distance, key in best]


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        v = [0.0] * 3072
        w = v.copy()
        v[0], w[1] = 1.0, 1.0
        store = VectorStore(folder, "self-test", 3072, create=True)
        store.add([{"chunk_id": "first"}, {"chunk_id": "second"}], [v, w])
        reopened = VectorStore(folder, "self-test", 3072)
        if reopened.count() != 2 or reopened.query(w, 1)[0][0] != "second":
            raise RuntimeError("Storage self-test failed")
    print("Storage self-test OK: 3072 dimensions, insert, reopen, search. No API calls.", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()
    self_test()
