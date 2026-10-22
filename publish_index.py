"""Update the existing submission's public index and cloud loader; no API calls."""
import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
from urllib.parse import quote
import uuid
import re
from retrieval import SearchIndex
from vector_store import VectorStore
from index_package import pack

BASE = Path(__file__).resolve().parent


def publish(destination):
    destination = Path(destination)
    if not (destination / 'app.py').is_file():
        raise ValueError('Choose the existing phoenix-submission folder.')
    index = SearchIndex(BASE / 'data/index')
    m = index.manifest
    if m['mode'] != 'semantic' or m.get('vector_backend') != 'sqlite-numpy-v1':
        raise ValueError('A ready semantic index is required.')
    store = VectorStore(index.path / 'vectors', m['model'], m['dimensions'])
    if store.count() != len(index.chunks):
        raise ValueError('Vector count mismatch')
    root = destination / 'demo_index'
    root.mkdir(exist_ok=True)
    output = root / index.path.name
    if output.exists():
        raise ValueError('This generation is already exported; no files changed.')
    # These are generated copies from an earlier failed export, never source data.
    for leftover in root.glob(index.path.name + '.staging-*'):
        suffix = leftover.name.removeprefix(index.path.name + '.staging-')
        if re.fullmatch(r'[0-9a-f]{32}', suffix) and leftover.is_dir() and not leftover.is_symlink():
            shutil.rmtree(leftover)
    ignore = destination / '.gitignore'
    rules = ignore.read_text(encoding='utf-8') if ignore.exists() else ''
    if '/demo_index/*.staging-*/' not in rules.splitlines():
        ignore.write_text(rules.rstrip('\n') + '\n/demo_index/*.staging-*/\n', encoding='utf-8')
    staging = root / (index.path.name + '.staging-' + uuid.uuid4().hex)
    (staging / 'vectors').mkdir(parents=True)
    for name in ['chunks.jsonl', 'manifest.json']:
        shutil.copy2(index.path / name, staging / name)
    uri = 'file:' + quote(store.path.resolve().as_posix(), safe='/:') + '?mode=ro'
    with closing(sqlite3.connect(uri, uri=True)) as source:
        with closing(sqlite3.connect(staging / 'vectors/vectors.sqlite3')) as target:
            source.backup(target)
    if hashlib.sha256((staging / 'chunks.jsonl').read_bytes()).hexdigest() != m['chunks_sha256']:
        raise ValueError('Index byte integrity check failed')
    if VectorStore(staging / 'vectors', m['model'], m['dimensions']).count() != len(index.chunks):
        raise ValueError('Snapshot vector count mismatch')
    pack(staging)
    staging.replace(output)
    # Only allowlisted source files. Never copy .env, usage ledgers or caches.
    for name in ['cloud_start.py', 'index_package.py', 'publish_index.py', 'build_expanded.py', 'export_submission.py',
                 'embedding_provider.py', 'retrieval.py', 'crawler.py', 'crawl_http.py', 'recover_crawl.py']:
        shutil.copy2(BASE / name, destination / name)
    for name in ['static/app.js', 'static/style.css', 'templates/index.html']:
        shutil.copy2(BASE / name, destination / name)
    if (BASE / 'static/phoenix-logo.png').is_file():
        shutil.copy2(BASE / 'static/phoenix-logo.png', destination / 'static/phoenix-logo.png')
    if (BASE / 'static/happy-bot.png').is_file():
        shutil.copy2(BASE / 'static/happy-bot.png', destination / 'static/happy-bot.png')
    attrs = destination / '.gitattributes'
    text = attrs.read_text(encoding='utf-8') if attrs.exists() else ''
    if 'demo_index/** -text' not in text.splitlines():
        attrs.write_text(text.rstrip('\n') + '\ndemo_index/** -text\n', encoding='utf-8')
    (destination / 'INDEX_COVERAGE.json').write_text(json.dumps({
        'generation': index.path.name, 'pages': m['included_pages'], 'chunks': m['chunks'],
        'model': m['model'], 'pdf_content_included': False, 'coverage_verified_complete': False,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    pointer = root / 'CURRENT.tmp'
    pointer.write_text(json.dumps({'generation': index.path.name}), encoding='utf-8')
    pointer.replace(root / 'CURRENT.json')
    print(json.dumps({'submission': str(destination.resolve()), 'pages': m['included_pages'],
                      'chunks': m['chunks'], 'status': 'exported', 'api_calls': 0}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=BASE.parent / 'phoenix-submission')
    args = parser.parse_args()
    publish(args.output)
