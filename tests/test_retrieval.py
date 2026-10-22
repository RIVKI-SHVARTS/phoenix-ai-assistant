import json
import math
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from chunking import chunk_page, split_long, token_count, make_chunks
from embedding_provider import BudgetExceeded, OpenAIEmbeddings, TokenBudget
from ingest import build_index
from retrieval import SearchIndex
from vector_store import VectorStore


def page(text):
    return {
        "url": "https://www.fnx.co.il/test/",
        "title": "בדיקת דירה",
        "text": text,
        "fetched_at": "2026-10-15T08:00:00Z",
        "content_hash": "test",
        "status": "extracted",
    }


class TestChunking(unittest.TestCase):
    def test_long_hebrew_has_no_lost_or_replacement_characters(self):
        source = "שָׁלוֹם ביטוח דירה 🏠 עם מידע ארוך ומפורט ללא חיתוך אותיות. " * 100
        parts = list(split_long(source, 96))
        self.assertEqual("".join(parts), source)
        self.assertTrue(all(token_count(s) <= 96 for s in parts))
        self.assertNotIn("\ufffd", "".join(parts))

    def test_header_context_budget_and_stable_ids(self):
        p = page(
            "# ביטוח\n\n## תביעה\n\n"
            + ("פרטי מסמכים תביעה בבית " + "אישור משטרה ") * 200
        )
        a = chunk_page(p, max_tokens=200, overlap_tokens=30)
        self.assertGreater(len(a), 1)
        self.assertTrue(all(c["token_count"] <= 200 for c in a))
        self.assertTrue(all(c["heading"] == "ביטוח > תביעה" for c in a[1:]))
        self.assertEqual(
            [c["chunk_id"] for c in a],
            [c["chunk_id"] for c in chunk_page(p, max_tokens=200, overlap_tokens=30)],
        )

    def test_review_and_duplicate_pages_are_not_indexed(self):
        for status in ["needs_review", "duplicate_content"]:
            p = page("text")
            p["status"] = status
            self.assertEqual(chunk_page(p), [])

    def test_literal_special_tokens_are_treated_as_source_text(self):
        self.assertGreater(token_count("מקור <|endoftext|> סוף"), 0)


class TestBudgetAndProvider(unittest.TestCase):
    def test_budget_persists_and_failed_reservations_stay_charged(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "budget.db"
            budget = TokenBudget(path, limit=10)
            budget.reserve(7)
            with self.assertRaises(BudgetExceeded):
                TokenBudget(path, limit=10).reserve(4)

    def test_api_response_reordered_by_input_index(self):
        class API:
            def create(self, **kwargs):
                self.kwargs = kwargs
                v = [0.0] * 1536
                v[0] = 1.0
                w = [0.0] * 1536
                w[1] = 1.0
                return SimpleNamespace(
                    data=[
                        SimpleNamespace(index=1, embedding=w),
                        SimpleNamespace(index=0, embedding=v),
                    ],
                    usage=SimpleNamespace(total_tokens=10),
                )

        with tempfile.TemporaryDirectory() as d:
            api = API()
            provider = OpenAIEmbeddings(
                "text-embedding-3-small",
                TokenBudget(Path(d) / "budget.db"),
                SimpleNamespace(embeddings=api),
            )
            vectors = provider.embed(["מסמך", "תביעה"])
            self.assertEqual(vectors[0][0], 1)
            self.assertEqual(vectors[1][1], 1)
            self.assertEqual(api.kwargs["encoding_format"], "float")

    def test_wrong_shape_or_nan_is_rejected(self):
        for vector in [[1.0], [math.nan] * 1536]:
            api = SimpleNamespace(
                embeddings=SimpleNamespace(
                    create=lambda **kw: SimpleNamespace(
                        data=[SimpleNamespace(index=0, embedding=vector)],
                        usage=SimpleNamespace(total_tokens=1),
                    )
                )
            )
            with tempfile.TemporaryDirectory() as d:
                provider = OpenAIEmbeddings(
                    "text-embedding-3-small", TokenBudget(Path(d) / "b"), api
                )
                with self.assertRaises(RuntimeError):
                    provider.embed(["בדיקה"])

    def test_missing_key_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(
            "os.environ", {"OPENAI_API_KEY": ""}
        ):
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                OpenAIEmbeddings(budget=TokenBudget(Path(d) / "b"))


class TestVectorStorage(unittest.TestCase):
    def test_persistence_ranking_and_metadata_guard(self):
        # These are deliberately hand-written vectors, not real AI embeddings.
        with tempfile.TemporaryDirectory() as d:
            store = VectorStore(Path(d) / "chroma", "test-only", 2, create=True)
            chunks = [
                {
                    "chunk_id": i,
                    "page_url": "https://www.fnx.co.il/" + i,
                    "page_title": i,
                    "heading": "נושא",
                    "fetched_at": "2026",
                    "text": i,
                }
                for i in ["car", "home"]
            ]
            store.add(chunks, [[1.0, 0.0], [0.0, 1.0]])
            reopened = VectorStore(Path(d) / "chroma", "test-only", 2)
            self.assertEqual(reopened.query([0.0, 1.0], 1)[0][0], "home")
            with self.assertRaises(ValueError):
                VectorStore(Path(d) / "chroma", "other-model", 2)

    def test_failed_rebuild_keeps_previous_current_index(self):
        class BrokenProvider:
            model = "test-only"
            dimensions = 2

            def embed(self, texts):
                raise RuntimeError("simulated API error")

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            source = d / "pages.jsonl"
            source.write_text(
                json.dumps(
                    page("# כותרת\n\nמסמכי תביעה אישור משטרה"), ensure_ascii=False
                )
            )
            root = d / "index"
            build_index(source, root, "lexical")
            old = (root / "CURRENT.json").read_bytes()
            with self.assertRaises(RuntimeError):
                build_index(source, root, "semantic", "test-only", BrokenProvider())
            self.assertEqual((root / "CURRENT.json").read_bytes(), old)
            results = SearchIndex(root).search("אישור משטרה")["results"]
            self.assertTrue(results)

    def test_lexical_mode_cannot_masquerade_as_semantic(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            source = d / "pages.jsonl"
            source.write_text(json.dumps(page("# מסמכים\n\nתביעה על נזק בבית")))
            build_index(source, d / "index", "lexical")
            search = SearchIndex(d / "index")
            with self.assertRaises(ValueError):
                search.search("נזילה", mode="semantic")
            self.assertEqual(search.search("קוואזרים")["results"], [])

    def test_tampered_chunk_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            source = d / "p"
            source.write_text(json.dumps(page("# מידע\n\nמסמכי תביעה")))
            out, _ = build_index(source, d / "index", "lexical")
            with (out / "chunks.jsonl").open("a") as f:
                f.write(" ")
            with self.assertRaisesRegex(ValueError, "hash"):
                SearchIndex(d / "index")

class TestRecovery(unittest.TestCase):
    def test_cache_survives_storage_failure_and_retry(self):
        calls = []
        def create(**kwargs):
            calls.append(kwargs["input"])
            return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[1.0]+[0.0]*1535)
                for i, _ in enumerate(kwargs["input"])], usage=SimpleNamespace(total_tokens=20))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "index"
            source = Path(folder) / "pages.jsonl"
            source.write_text(json.dumps(page("# מידע\n\nמסמכי תביעה")), encoding="utf-8")
            def provider():
                return OpenAIEmbeddings("text-embedding-3-small", TokenBudget(root / "usage.db"),
                                        SimpleNamespace(embeddings=SimpleNamespace(create=create)))
            build_index(source, root, "lexical")
            old = (root / "CURRENT.json").read_bytes()
            with patch.object(VectorStore, "add", side_effect=RuntimeError("disk failure")):
                with self.assertRaises(RuntimeError):
                    build_index(source, root, "semantic", "text-embedding-3-small", provider())
            self.assertEqual((root / "CURRENT.json").read_bytes(), old)
            _, manifest = build_index(source, root, "semantic", "text-embedding-3-small", provider())
            self.assertEqual(manifest["status"], "ready")
            self.assertEqual(len(calls), 1)
            self.assertTrue(SearchIndex(root, provider()).search("מסמכים", mode="semantic")["results"])

    def test_partial_cache_duplicates_and_order(self):
        calls = []
        def create(**kwargs):
            calls.append(kwargs["input"])
            return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[float(len(t)), 1.0]+[0.0]*1534)
                for i,t in enumerate(kwargs["input"])], usage=SimpleNamespace(total_tokens=10))
        with tempfile.TemporaryDirectory() as folder:
            p = OpenAIEmbeddings("text-embedding-3-small", TokenBudget(Path(folder)/"usage"),
                                 SimpleNamespace(embeddings=SimpleNamespace(create=create)))
            first = p.embed(["א"])[0]
            results = p.embed(["ארוך", "א", "ארוך"])
            self.assertEqual(calls, [["א"], ["ארוך"]])
            self.assertEqual(results[1], first)
            self.assertEqual(results[0], results[2])

    def test_exact_search_spans_batches_and_failed_add_is_atomic(self):
        import numpy as np
        rng = np.random.default_rng(42)
        vectors = rng.normal(size=(600, 12))
        q = rng.normal(size=12)
        expected = np.argsort(-(vectors @ q)/(np.linalg.norm(vectors, axis=1)*np.linalg.norm(q)))[:10]
        with tempfile.TemporaryDirectory() as folder:
            store = VectorStore(folder, "test", 12, create=True)
            store.add([{"chunk_id": str(i)} for i in range(600)], vectors)
            self.assertEqual([key for key,_ in store.query(q,10)], [str(i) for i in expected])
            with self.assertRaises(sqlite3.IntegrityError):
                store.add([{"chunk_id":"new"}, {"chunk_id":"0"}], vectors[:2])
            self.assertEqual(store.count(),600)
            with self.assertRaises(ValueError):
                store.add([{"chunk_id":"bad"}], [[0.0]*12])
            self.assertEqual(store.count(),600)


if __name__ == "__main__":
    unittest.main()
