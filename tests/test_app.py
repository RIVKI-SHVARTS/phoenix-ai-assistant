"""HTTP integration tests using real saved lexical chunks, no external API."""

import json
from pathlib import Path
import tempfile
import unittest
from app import create_app

ROOT = Path(__file__).resolve().parents[1]


class AppTests(unittest.TestCase):
    def setUp(self):
        from ingest import build_index
        self.fixture = tempfile.TemporaryDirectory()
        self.addCleanup(self.fixture.cleanup)
        root = Path(self.fixture.name)
        build_index(ROOT / 'data/crawls/20261015T084034Z-225fef25/pages.jsonl', root)
        self.app = create_app(root)
        self.client = self.app.test_client()

    def test_ui_and_assets(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn('dir="rtl"', response.text)
        self.assertIn(
            "frame-ancestors 'none'", response.headers["Content-Security-Policy"]
        )
        for url in ["/static/app.js", "/static/style.css"]:
            with self.client.get(url) as asset:
                self.assertEqual(asset.status_code, 200)

    def test_real_search_returns_sources_without_claiming_answer(self):
        health = self.client.get("/api/health").json
        self.assertTrue(health["index_ready"])
        self.assertFalse(health["chat_ready"])
        r = self.client.post(
            "/api/search",
            json={"question": "אילו מסמכים צריך לצרף לתביעה על נזק לדירה?"},
        )
        self.assertEqual(r.status_code, 200)
        self.assertFalse(r.json["answer_generated"])
        self.assertEqual(r.json["mode"], "lexical")
        self.assertTrue(
            any("תמונות ממקום האירוע" in c["text"] for c in r.json["results"])
        )
        self.assertTrue(
            all(
                c["page_url"].startswith("https://www.fnx.co.il/")
                for c in r.json["results"]
            )
        )

    def test_input_rejected(self):
        for value in [
            None,
            [],
            {},
            {"question": None},
            {"question": 2},
            {"question": " "},
            {"question": "א" * 2001},
            {"question": "x", "mode": "semantic"},
        ]:
            r = self.client.post(
                "/api/search",
                data=json.dumps(value, ensure_ascii=False),
                content_type="application/json",
            )
            self.assertEqual(r.status_code, 400)
        self.assertEqual(self.client.post("/api/search", data="x").status_code, 415)
        self.assertEqual(
            self.client.post(
                "/api/search", data="x" * 61000, content_type="application/json"
            ).status_code,
            413,
        )

    def test_no_index_returns_clear_error(self):
        with tempfile.TemporaryDirectory() as d:
            client = create_app(d).test_client()
            self.assertEqual(client.get("/api/health").status_code, 503)
            self.assertEqual(
                client.post("/api/search", json={"question": "בדיקה"}).status_code, 503
            )

    def test_private_or_build_endpoints_do_not_exist(self):
        for path in [
                        "/api/crawl",
            "/api/ingest",
            "/data/index/CURRENT.json",
        ]:
            self.assertEqual(self.client.get(path).status_code, 404)

    def test_untrusted_host_rejected(self):
        self.assertEqual(
            self.client.get("/", headers={"Host": "untrusted.example"}).status_code, 400
        )


if __name__ == "__main__":
    unittest.main()
