"""Export and cloud bootstrap using synthetic vectors; no network or paid API."""
import base64
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from app import create_app
from cloud_start import prepare_index
import export_submission
from vector_store import VectorStore

ROOT = Path(__file__).resolve().parents[1]


class CloudExportTests(unittest.TestCase):
    def test_cloud_host_and_demo_password_gate(self):
        with patch.dict('os.environ', {'RENDER_EXTERNAL_HOSTNAME': 'phoenix-test.onrender.com',
                                       'DEMO_PASSWORD': 'demo-test-only'}):
            client = create_app().test_client()
            self.assertEqual(client.get('/healthz', base_url='https://phoenix-test.onrender.com').status_code, 200)
            self.assertEqual(client.get('/', base_url='https://phoenix-test.onrender.com').status_code, 401)
            auth = base64.b64encode(b'demo:demo-test-only').decode()
            self.assertEqual(client.get('/', base_url='https://phoenix-test.onrender.com',
                headers={'Authorization': 'Basic ' + auth}).status_code, 200)
            self.assertEqual(client.get('/', base_url='https://attacker.example',
                headers={'Authorization': 'Basic ' + auth}).status_code, 400)

    def test_allowlisted_export_and_runtime_reopen_preserve_existing_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            source = temp / 'source'
            shutil.copytree(ROOT, source, ignore=shutil.ignore_patterns(
                'data', '__pycache__', '.venv', '*.sqlite3', '*.zip'))
            sample = ROOT / 'data/crawls/20261015T084034Z-225fef25'
            shutil.copytree(sample, source / 'data/crawls' / sample.name)
            chunk = {'chunk_id': 'synthetic-one', 'page_url': 'https://www.fnx.co.il/test/',
                     'page_title': 'בדיקה', 'heading': 'בדיקה', 'text': 'טקסט מקור לבדיקה.',
                     'embedding_text': 'טקסט מקור לבדיקה.', 'token_count': 10,
                     'fetched_at': '2026-10-22T00:00:00Z'}
            generation = '20261022T000000Z-aaaaaaaa'
            root = source / 'data/index'
            folder = root / generation
            folder.mkdir(parents=True)
            raw = (json.dumps(chunk, ensure_ascii=False) + '\n').encode()
            (folder / 'chunks.jsonl').write_bytes(raw)
            manifest = {'generation': generation, 'mode': 'semantic', 'status': 'ready',
                        'chunks': 1, 'included_pages': 1, 'model': 'text-embedding-3-small',
                        'dimensions': 1536, 'vector_backend': 'sqlite-numpy-v1',
                        'chunks_sha256': hashlib.sha256(raw).hexdigest()}
            (folder / 'manifest.json').write_text(json.dumps(manifest))
            (root / 'CURRENT.json').write_text(json.dumps({'generation': generation}))
            vectors = VectorStore(folder / 'vectors', manifest['model'], 1536, create=True)
            vectors.add([chunk], [[1.0] + [0.0] * 1535])
            (source / '.env').write_text('DO_NOT_EXPORT')
            (source / 'private_key.py').write_text('DO_NOT_EXPORT')
            (root / 'answer-usage.sqlite3').write_text('LOCAL_USAGE')
            (root / 'embedding-cache.sqlite3').write_text('PRIVATE_CACHE')
            destination = temp / 'submission'
            with patch.object(export_submission, 'BASE', source):
                export_submission.export(destination)
            self.assertFalse((destination / '.env').exists())
            self.assertFalse((destination / 'private_key.py').exists())
            self.assertFalse((destination / 'data/index').exists())
            self.assertEqual((root / 'answer-usage.sqlite3').read_text(), 'LOCAL_USAGE')
            runtime = temp / 'runtime'
            prepare_index(destination / 'demo_index', runtime)
            self.assertEqual(VectorStore(runtime / generation / 'vectors', manifest['model'], 1536).count(), 1)
            (runtime / 'answer-usage.sqlite3').write_text('CLOUD_USAGE')
            prepare_index(destination / 'demo_index', runtime)
            self.assertEqual((runtime / 'answer-usage.sqlite3').read_text(), 'CLOUD_USAGE')
            with patch.object(export_submission, 'BASE', source):
                with self.assertRaises(ValueError):
                    export_submission.export(destination)
