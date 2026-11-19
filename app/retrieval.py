"""BM25 retrieval over the documents a principal is allowed to see.

The index is built from the caller's visible documents on every query. That
keeps term statistics from other tenants out of the scores (no IDF side
channel) and makes the permission boundary obvious.
shortcut: rebuilds per query, fine for hundreds of docs; cache per (tenant, clearance) when the corpus grows.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from app.store import Document

_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i in is it its of on or our "
    "the this to was we what when which who why will with you your me my".split()
)
K1, B = 1.5, 0.75


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        if tok in _STOPWORDS:
            continue
        tokens.append(tok)
        if "-" in tok:  # LAP-14-STD also matches "lap" and "std"
            tokens.extend(p for p in tok.split("-") if p not in _STOPWORDS)
    return tokens


@dataclass(frozen=True)
class Hit:
    doc: Document
    score: float


def search(docs: list[Document], query: str, k: int) -> list[Hit]:
    q_terms = set(tokenize(query))
    if not docs or not q_terms:
        return []
    bags = [Counter(tokenize(f"{d.title} {d.text}")) for d in docs]
    avg_len = sum(sum(b.values()) for b in bags) / len(bags)
    df = Counter(t for bag in bags for t in set(bag))
    n = len(docs)
    hits = []
    for doc, bag in zip(docs, bags, strict=True):
        length = sum(bag.values())
        score = 0.0
        for term in q_terms & bag.keys():
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            tf = bag[term]
            score += idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * length / avg_len))
        if score > 0:
            hits.append(Hit(doc, round(score, 4)))
    hits.sort(key=lambda h: (-h.score, h.doc.id))
    return hits[:k]
