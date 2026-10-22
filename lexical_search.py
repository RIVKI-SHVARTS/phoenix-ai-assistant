"""Explicit word-search baseline. This module does not understand semantic meaning."""

import re
import unicodedata
from rank_bm25 import BM25L

STOP = set(
    "איך כיצד מה האם אני לי את של על עם או זה זו יש אפשר כדי שלי צריך ניתן וגם כל הוא היא הם הן".split()
)


def terms(text):
    # Do not let URL path names dominate Hebrew relevance.
    text = re.sub(r"\[([^\]]+)\]\([^\n]*?\)", r"\1", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = "".join(
        c
        for c in unicodedata.normalize("NFKD", text.lower())
        if not unicodedata.combining(c)
    )
    words = re.findall(r"[a-z0-9]+|[\u05d0-\u05ea]+", text)
    out = []
    for word in words:
        if word in STOP:
            continue
        out.append(word)
        # Supplement exact Hebrew forms; no claim of morphological analysis.
        if len(word) > 3 and word[0] in "והבכלמש":
            root = word[1:]
            if root not in STOP:
                out.append(root)
    return out


class LexicalSearch:
    def __init__(self, chunks):
        self.chunks = chunks
        self.model = (
            BM25L([terms(c["embedding_text"]) for c in chunks]) if chunks else None
        )

    def search(self, question, limit=6):
        if (
            not isinstance(question, str)
            or not question.strip()
            or len(question) > 2000
        ):
            raise ValueError("Question must contain 1–2000 characters")
        if not 1 <= limit <= 30:
            raise ValueError("Invalid result limit")
        query = terms(question)
        if not query or self.model is None:
            return []
        scores = self.model.get_scores(query)
        ranked = sorted(
            range(len(scores)),
            key=lambda i: (-float(scores[i]), self.chunks[i]["chunk_id"]),
        )
        return [
            dict(
                self.chunks[i],
                retrieval_score=float(scores[i]),
                retrieval_mode="lexical",
            )
            for i in ranked
            if scores[i] > 0
        ][:limit]
