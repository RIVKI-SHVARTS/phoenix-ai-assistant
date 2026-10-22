"""Estimate first; explicitly reserve a bounded additional build allowance."""
import argparse
import json
import os
from pathlib import Path
import sqlite3
from contextlib import closing
from embedding_provider import OpenAIEmbeddings, TokenBudget, MODELS
from ingest import prepare, build_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pages_file', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--max-new-tokens', type=int, default=6500000)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 10000000:
        parser.error('max-new-tokens must be between 1 and 10000000')
    _, stats = prepare(args.pages_file)
    print(json.dumps({k: stats[k] for k in ['included_pages', 'chunks', 'embedding_tokens',
          'estimated_embedding_usd', 'model']}, ensure_ascii=False), flush=True)
    if not args.execute:
        print('Estimate only. No API calls. Add --execute to build.', flush=True)
        return
    if stats['embedding_tokens'] > args.max_new_tokens:
        raise SystemExit('Estimate exceeds the explicit new-token limit. No API calls sent.')
    if not os.environ.get('OPENAI_API_KEY'):
        raise SystemExit('OPENAI_API_KEY missing in this terminal. No API calls sent.')
    root = Path(__file__).resolve().parent / 'data/index'
    budget = TokenBudget(root / 'embedding-usage.sqlite3')
    with closing(sqlite3.connect(budget.path)) as db:
        used = db.execute('SELECT COALESCE(SUM(reserved),0) FROM usage').fetchone()[0]
    limit = used + args.max_new_tokens
    # Retain all existing usage; never delete or reset the persistent ledger.
    config = root / 'embedding-budget.json'
    temporary = root / 'embedding-budget.tmp'
    temporary.write_text(json.dumps({'token_limit': limit,
        'additional_build_allowance': args.max_new_tokens}), encoding='utf-8')
    temporary.replace(config)
    provider = OpenAIEmbeddings(stats['model'], TokenBudget(budget.path, limit=limit))
    maximum = args.max_new_tokens * MODELS[stats['model']]['usd_per_million'] / 1000000
    print(f'Building with at most {args.max_new_tokens} additional embedding tokens '
          f'(estimated-price ceiling USD {maximum:.3f}). No automatic API retries.', flush=True)
    output, manifest = build_index(args.pages_file, root, mode='semantic',
                                   model=stats['model'], provider=provider)
    print(json.dumps({'index': str(output), 'status': manifest['status'],
                      'pages': manifest['included_pages'], 'chunks': manifest['chunks']}), flush=True)


if __name__ == '__main__':
    main()
