"""Prepare a precomputed index without API calls, then launch one threaded worker."""
import json
import os
from pathlib import Path
import re
import shutil
import uuid
from index_package import unpack
from retrieval import SearchIndex
from vector_store import VectorStore

BASE = Path(__file__).resolve().parent


def prepare_index(source, destination):
    source, destination = Path(source), Path(destination)
    current = json.loads((source / 'CURRENT.json').read_text(encoding='utf-8'))
    generation = current['generation']
    if not re.fullmatch(r'\d{8}T\d{6}Z-[0-9a-f]{8}', generation):
        raise ValueError('Invalid generation')
    active = json.loads((destination / 'CURRENT.json').read_text(encoding='utf-8')) if (destination / 'CURRENT.json').exists() else None
    if active != current:
        destination.mkdir(parents=True, exist_ok=True)
        staging = destination / (generation + '.staging-' + uuid.uuid4().hex)
        shutil.copytree(source / generation, staging)
        unpack(staging)
        # Validate before switching away from any existing working generation.
        # SearchIndex validates the byte signature, IDs and recorded count.
        raw = (staging / 'chunks.jsonl').read_bytes()
        m = json.loads((staging / 'manifest.json').read_text(encoding='utf-8'))
        import hashlib
        if hashlib.sha256(raw).hexdigest() != m['chunks_sha256']:
            raise ValueError('Index hash mismatch')
        if m['status'] != 'ready' or m['mode'] != 'semantic' or m.get('vector_backend') != 'sqlite-numpy-v1':
            raise ValueError('Cloud demo requires a ready semantic index')
        rows = [json.loads(line) for line in raw.decode('utf-8').splitlines()]
        if len(rows) != m['chunks'] or len({r['chunk_id'] for r in rows}) != len(rows):
            raise ValueError('Invalid chunk count or duplicate IDs')
        if VectorStore(staging / 'vectors', m['model'], m['dimensions']).count() != len(rows):
            raise ValueError('Vector count mismatch')
        del rows, raw
        target = destination / generation
        if target.exists():
            shutil.rmtree(target)
        staging.replace(target)
        pointer = destination / 'CURRENT.tmp'
        pointer.write_text(json.dumps(current), encoding='utf-8')
        pointer.replace(destination / 'CURRENT.json')
    index = SearchIndex(destination)
    if index.manifest['mode'] != 'semantic' or index.manifest.get('vector_backend') != 'sqlite-numpy-v1':
        raise ValueError('Cloud demo requires a ready semantic index')
    store = VectorStore(index.path / 'vectors', index.manifest['model'], index.manifest['dimensions'])
    if store.count() != len(index.chunks):
        raise ValueError('Vector count mismatch')


if __name__ == '__main__':
    if not os.environ.get('OPENAI_API_KEY') or not os.environ.get('DEMO_PASSWORD'):
        raise SystemExit('Set OPENAI_API_KEY and DEMO_PASSWORD privately before deployment.')
    root = Path(os.environ.get('PHOENIX_INDEX_ROOT', 'runtime/index'))
    if not root.is_absolute():
        root = BASE / root
    os.environ['PHOENIX_INDEX_ROOT'] = str(root)
    prepare_index(BASE / 'demo_index', root)
    port = int(os.environ.get('PORT', '10000'))
    os.chdir(BASE)
    os.execvp('gunicorn', ['gunicorn', '--bind', f'0.0.0.0:{port}', '--workers', '1',
                          '--threads', '4', '--timeout', '330', 'app:app'])
