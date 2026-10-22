"""Export an allowlisted public project and its existing semantic index. No API calls."""
import json
import re
from pathlib import Path
import shutil
import sqlite3
from urllib.parse import quote
from retrieval import SearchIndex
from vector_store import VectorStore

BASE = Path(__file__).resolve().parent


def export(destination):
    destination = Path(destination)
    if destination.exists():
        raise ValueError('Destination already exists. Preserve it or choose a new destination.')
    index = SearchIndex(BASE / 'data/index')
    m = index.manifest
    if m['mode'] != 'semantic' or m.get('vector_backend') != 'sqlite-numpy-v1':
        raise ValueError('A ready semantic index must be active before export.')
    store = VectorStore(index.path / 'vectors', m['model'], m['dimensions'])
    if store.count() != len(index.chunks):
        raise ValueError('Vector count does not match chunks.')
    destination.mkdir(parents=True)
    # No recursive copy of the user's working directory, secrets, usage ledgers or caches.
    names = [
        'answering.py', 'app.py', 'audit_crawl.py', 'chunking.py', 'crawl_http.py',
        'crawler.py', 'diagnose_answer.py', 'diagnose_cases.py', 'diagnose_review.py',
        'embedding_provider.py', 'evaluate_retrieval.py', 'extract_content.py',
        'ingest.py', 'lexical_search.py', 'preflight.py', 'retrieval.py', 'vector_store.py',
        'cloud_start.py', 'export_submission.py', 'run_chat_checks.py',
        'run_final_checks.py', 'finish_remaining_checks.py',
        'requirements.txt', 'requirements-crawler.txt', 'requirements-retrieval.txt',
        'requirements-cloud.txt', 'README.md', 'SUBMISSION_HE.md', 'DEPLOY_HE.md',
        'render.yaml', '.gitignore', 'crawl_config.json',
        'static/app.js', 'static/style.css', 'templates/index.html',
        'tests/test_answering.py', 'tests/test_app.py', 'tests/test_crawler.py',
        'tests/test_preflight.py', 'tests/test_retrieval.py', 'tests/test_final_checks.py',
        'tests/test_cloud_export.py', 'tests/chat-ui.test.cjs', 'tests/source-links.test.cjs',
        'tests/fixtures/theft-diagnostic.json', 'tests/fixtures/mixed-request-diagnostic.json',
        'tests/fixtures/digital-claim-sources.json',
    ]
    for name in names:
        source_path = BASE / name
        # Refuse a likely hardcoded secret without printing the matching value.
        if re.search(r'sk-[A-Za-z0-9_-]{20,}', source_path.read_text(encoding='utf-8')):
            raise ValueError('Possible hardcoded API key in an export file; export stopped.')
        target_path = destination / name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
    (destination / 'evaluation').mkdir()
    for name in ['chat_cases.json', 'retrieval_cases.json']:
        shutil.copy2(BASE / 'evaluation' / name, destination / 'evaluation' / name)
    # Public source fixture is retained so crawl/retrieval checks remain reproducible.
    sample = BASE / 'data/crawls/20261015T084034Z-225fef25'
    shutil.copytree(sample, destination / 'data/crawls' / sample.name)
    out = destination / 'demo_index' / index.path.name
    (out / 'vectors').mkdir(parents=True)
    for name in ['chunks.jsonl', 'manifest.json']:
        shutil.copy2(index.path / name, out / name)
    # Consistent SQLite snapshot, including committed rows still in WAL if present.
    source_uri = 'file:' + quote(store.path.resolve().as_posix(), safe='/:') + '?mode=ro'
    with sqlite3.connect(source_uri, uri=True) as source:
        with sqlite3.connect(out / 'vectors/vectors.sqlite3') as target:
            source.backup(target)
    (out.parent / 'CURRENT.json').write_text(
        json.dumps({'generation': index.path.name}), encoding='utf-8')
    (destination / 'INDEX_COVERAGE.json').write_text(json.dumps({
        'generation': index.path.name,
        'pages': len({chunk['page_url'] for chunk in index.chunks}),
        'chunks': len(index.chunks),
        'model': m['model'],
        'pdf_content_included': False,
        'coverage_verified_complete': False,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    saved = VectorStore(out / 'vectors', m['model'], m['dimensions'])
    if saved.count() != len(index.chunks):
        raise ValueError('Export verification failed')
    print('Export complete: ' + str(destination.resolve()))
    print('Includes public source snapshots and vectors. No API key, caches or usage ledgers.')
    return destination


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=BASE.parent / 'phoenix-submission')
    args = parser.parse_args()
    try:
        export(args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, str(error) + '\n')
