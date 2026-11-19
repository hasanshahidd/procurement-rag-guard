"""Hybrid retrieval (BM25 plus embeddings) over the documents a principal may see.

Both rankers only ever receive the caller's visible documents. The BM25 index
is built from that subset on every query, so term statistics from other
tenants never shape the scores (no IDF side channel). Dense scores are plain
cosine similarities, which do not depend on other documents at all.
shortcut: BM25 rebuilds per query, fine for hundreds of docs; cache per (tenant, clearance) when the corpus grows.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from app.embeddings import Embedder, EmbeddingUnavailableError, cosine
from app.store import Document

RRF_K = 60  # standard reciprocal rank fusion constant

_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i in is it its of on or our "
    "the this to was we what when which who why will with you your me my".split()
)
K1, B = 1.5, 0.75


def tokenize(text: str, parts: bool = True) -> list[str]:
    tokens: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        if tok in _STOPWORDS:
            continue
        tokens.append(tok)
        if parts and "-" in tok:  # LAP-14-STD also matches "lap" and "std"
            tokens.extend(p for p in tok.split("-") if p not in _STOPWORDS)
    return tokens


@dataclass(frozen=True)
class Hit:
    doc: Document
    score: float
    bm25: float | None = None
    dense: float | None = None


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
            hits.append(Hit(doc, round(score, 4), bm25=round(score, 4)))
    hits.sort(key=lambda h: (-h.score, h.doc.id))
    return hits[:k]


def hybrid_search(
    docs: list[Document], query: str, k: int, embedder: Embedder, min_bm25: float, min_dense: float
) -> tuple[list[Hit], str]:
    """Fuse BM25 and dense rankings with reciprocal rank fusion.

    A document is a candidate when either ranker finds it relevant enough
    (BM25 >= min_bm25 or cosine >= min_dense). Returns the hits and a short note
    for the agent trace. Falls back to BM25 alone if embeddings are unavailable.
    """
    lexical = {h.doc.id: h.bm25 for h in search(docs, query, len(docs))}
    try:
        q_vec = embedder.embed_query(query)
        d_vecs = embedder.embed_documents([f"{d.title}\n{d.text}" for d in docs])
    except EmbeddingUnavailableError as exc:
        hits = [h for h in search(docs, query, k) if (h.bm25 or 0) >= min_bm25]
        return hits, f"BM25 only, embeddings unavailable ({exc})"
    dense = {d.id: round(cosine(q_vec, v), 4) for d, v in zip(docs, d_vecs, strict=True)}
    eligible = [d for d in docs if lexical.get(d.id, 0) >= min_bm25 or dense[d.id] >= min_dense]
    by_lexical = sorted(eligible, key=lambda d: (-lexical.get(d.id, 0), d.id))
    by_dense = sorted(eligible, key=lambda d: (-dense[d.id], d.id))
    fused: dict[str, float] = {d.id: 0.0 for d in eligible}
    for ranking in (by_lexical, by_dense):
        for rank, d in enumerate(ranking, start=1):
            fused[d.id] += 1 / (RRF_K + rank)
    hits = [Hit(d, round(fused[d.id], 5), bm25=lexical.get(d.id), dense=dense[d.id]) for d in eligible]
    hits.sort(key=lambda h: (-h.score, h.doc.id))
    return hits[:k], f"hybrid BM25 + {embedder.name}"
