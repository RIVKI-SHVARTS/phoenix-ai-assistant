"""Token-bounded, source-preserving chunks with Hebrew-safe string boundaries."""

import hashlib
import json
import re
from pathlib import Path
import tiktoken

VERSION = "headings-paragraphs-v1"
ENCODING = "cl100k_base"


def tokenizer():
    return tiktoken.get_encoding(ENCODING)


def token_count(text):
    return len(tokenizer().encode(text, disallowed_special=()))


def split_long(text, limit):
    """Split Unicode strings, not token bytes, so Hebrew cannot be corrupted."""
    if limit < 16:
        raise ValueError("Token budget too small")
    while text:
        if token_count(text) <= limit:
            yield text
            return
        lo, hi = 1, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if token_count(text[:mid]) <= limit:
                lo = mid
            else:
                hi = mid - 1
        end = lo
        boundary = text.rfind(" ", 0, end)
        if boundary > end // 2:
            end = boundary
        piece = text[:end]
        if not piece or token_count(piece) > limit:
            raise ValueError("Unable to split safely")
        yield piece
        text = text[end:]


def sections(markdown):
    stack, lines = [], []
    for line in markdown.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if match:
            if lines or stack:
                yield [x[1] for x in stack], "\n".join(lines).strip()
            depth, title = len(match[1]), match[2]
            stack = [x for x in stack if x[0] < depth] + [(depth, title)]
            lines = []
        else:
            lines.append(line)
    if lines or stack:
        yield [x[1] for x in stack], "\n".join(lines).strip()


def chunk_page(page, max_tokens=650, overlap_tokens=80):
    if page.get("status") != "extracted":
        return []
    if not 128 <= max_tokens <= 2000 or not 0 <= overlap_tokens < max_tokens // 2:
        raise ValueError("Invalid chunk limits")
    output, seen = [], set()
    for section_index, (heading_path, content) in enumerate(sections(page["text"])):
        heading = " > ".join(heading_path)
        prefix = page["title"] + "\n" + heading + "\n\n"
        budget = max_tokens - token_count(prefix) - 8
        if budget < 32:
            raise ValueError("Heading too long; requires review instead of truncation")
        content = content or (heading_path[-1] if heading_path else "")
        if not content.strip():
            continue
        units = []
        for paragraph in re.split(r"\n\s*\n", content):
            if paragraph.strip():
                units.extend(
                    piece for piece in split_long(paragraph, budget) if piece.strip()
                )
        groups, current = [], []
        for unit in units:
            candidate = "\n\n".join(current + [unit])
            if current and token_count(prefix + candidate) > max_tokens:
                groups.append(current)
                overlap = (
                    [current[-1]] if token_count(current[-1]) <= overlap_tokens else []
                )
                current = (
                    overlap
                    if token_count(prefix + "\n\n".join(overlap + [unit])) <= max_tokens
                    else []
                )
            current.append(unit)
        if current:
            groups.append(current)
        for group in groups:
            text = "\n\n".join(group)
            embedded = prefix + text
            count = token_count(embedded)
            if count > max_tokens:
                raise ValueError("Chunk exceeds model-independent project limit")
            digest = hashlib.sha256(
                (page["url"] + "\n" + heading + "\n" + text).encode()
            ).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            output.append(
                {
                    "chunk_id": digest,
                    "page_url": page["url"],
                    "page_title": page["title"],
                    "heading": heading,
                    "section_index": section_index,
                    "text": text,
                    "embedding_text": embedded,
                    "token_count": count,
                    "fetched_at": page["fetched_at"],
                    "page_hash": page["content_hash"],
                    "chunking_version": VERSION,
                }
            )
    return output


def make_chunks(pages_file, max_tokens=650, overlap_tokens=80):
    pages = [
        json.loads(line)
        for line in Path(pages_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    chunks = [c for p in pages for c in chunk_page(p, max_tokens, overlap_tokens)]
    if not chunks:
        raise ValueError("No extracted pages to index")
    return chunks, {
        "source_pages": len(pages),
        "included_pages": len(set(c["page_url"] for c in chunks)),
        "excluded_pages": [
            {"url": p["url"], "status": p.get("status")}
            for p in pages
            if p.get("status") != "extracted"
        ],
        "chunks": len(chunks),
        "embedding_tokens": sum(c["token_count"] for c in chunks),
        "max_chunk_tokens": max(c["token_count"] for c in chunks),
        "chunking_version": VERSION,
        "max_tokens": max_tokens,
        "overlap_tokens": overlap_tokens,
        "encoding": ENCODING,
    }
