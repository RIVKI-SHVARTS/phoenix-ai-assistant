"""Split public index files into Git-sized parts and verify reconstruction."""
import hashlib
import json
from pathlib import Path

PART_BYTES = 48 * 1024 * 1024


def pack(folder, part_bytes=PART_BYTES):
    folder = Path(folder)
    records = []
    for relative in ['chunks.jsonl', 'vectors/vectors.sqlite3']:
        path = folder / relative
        if path.stat().st_size <= part_bytes:
            continue
        digest = hashlib.sha256()
        parts = []
        with path.open('rb') as source:
            while data := source.read(part_bytes):
                target = path.with_name(path.name + f'.part{len(parts):04d}')
                target.write_bytes(data)
                parts.append(target.relative_to(folder).as_posix())
                digest.update(data)
        records.append({'file': relative, 'parts': parts, 'sha256': digest.hexdigest(),
                        'bytes': path.stat().st_size})
        path.unlink()
    if records:
        (folder / 'index-parts.json').write_text(json.dumps(records, indent=2), encoding='utf-8')


def unpack(folder):
    folder = Path(folder)
    manifest = folder / 'index-parts.json'
    if not manifest.exists():
        return
    for record in json.loads(manifest.read_text(encoding='utf-8')):
        target = (folder / record['file']).resolve()
        parts = [(folder / p).resolve() for p in record['parts']]
        if not target.is_relative_to(folder.resolve()) or any(
                not p.is_relative_to(folder.resolve()) or p == target for p in parts):
            raise ValueError('Invalid index part path')
        if target.exists():
            raise ValueError('Index destination already exists')
        temporary = target.with_name(target.name + '.assembling')
        digest, size = hashlib.sha256(), 0
        with temporary.open('wb') as output:
            for part in parts:
                with part.open('rb') as source:
                    while data := source.read(1024 * 1024):
                        output.write(data); digest.update(data); size += len(data)
        if digest.hexdigest() != record['sha256'] or size != record['bytes']:
            temporary.unlink()
            raise ValueError('Index part integrity check failed')
        temporary.replace(target)
        for part in parts:
            part.unlink()
    manifest.unlink()
