"""Does retrieval find the records that answer the question?

Scores the two retrieval modes the API can run (keyword BM25 and the default
hybrid BM25 + embeddings) against `data/retrieval_eval.json`, a set of queries
labelled with the records that actually answer them. Each query is run as the
principal who asks it, so retrieval only ever sees that caller's visible
records, exactly as the pipeline does.

Metrics, macro-averaged over queries:
  recall@k  fraction of a query's gold records found in the top k
  MRR       mean reciprocal rank of the first gold record (0 if none in top k)
  hit@1     the top result is a gold record

Hybrid needs the Ollama embedding model; keyword mode does not. If embeddings
are unavailable the hybrid column is skipped and the report says so.

Usage (from the repo root):
    python -m repro.retrieval_eval
    python -m repro.retrieval_eval --out reports/retrieval_eval.md
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.config import ROOT, Settings
from app.embeddings import EmbeddingUnavailableError, OllamaEmbedder
from app.retrieval import hybrid_search, search
from app.store import Store

EVAL_PATH = ROOT / "data" / "retrieval_eval.json"


def recall_at_k(gold: list[str], ranked: list[str], k: int) -> float:
    top = ranked[:k]
    return len([g for g in gold if g in top]) / len(gold)


def reciprocal_rank(gold: list[str], ranked: list[str], k: int) -> float:
    for rank, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in gold:
            return 1 / rank
    return 0.0


def hit_at_1(gold: list[str], ranked: list[str]) -> float:
    return 1.0 if ranked and ranked[0] in gold else 0.0


@dataclass(frozen=True)
class QueryResult:
    id: str
    gold: list[str]
    ranked: list[str]
    recall: float
    rr: float
    hit1: float


def rank_keyword(store: Store, api_key: str, question: str, k: int, min_bm25: float) -> list[str]:
    principal = store.authenticate(api_key)
    assert principal is not None, api_key
    visible = store.visible_docs(principal)
    return [h.doc.id for h in search(visible, question, k) if h.score >= min_bm25]


def rank_hybrid(store: Store, embedder: OllamaEmbedder, api_key: str, question: str, k: int, s: Settings) -> list[str]:
    principal = store.authenticate(api_key)
    assert principal is not None, api_key
    visible = store.visible_docs(principal)
    hits, _ = hybrid_search(visible, question, k, embedder, s.min_retrieval_score, s.min_dense_score)
    return [h.doc.id for h in hits]


def score(queries: list[dict], ranker, k: int) -> list[QueryResult]:
    results = []
    for q in queries:
        ranked = ranker(q["api_key"], q["question"])
        results.append(
            QueryResult(
                id=q["id"],
                gold=q["gold"],
                ranked=ranked,
                recall=recall_at_k(q["gold"], ranked, k),
                rr=reciprocal_rank(q["gold"], ranked, k),
                hit1=hit_at_1(q["gold"], ranked),
            )
        )
    return results


def means(results: list[QueryResult]) -> dict[str, float]:
    n = len(results)
    return {
        "recall": sum(r.recall for r in results) / n,
        "mrr": sum(r.rr for r in results) / n,
        "hit1": sum(r.hit1 for r in results) / n,
    }


def render(columns: dict[str, list[QueryResult]], k: int, note: str) -> str:
    names = list(columns)
    lines = [
        f"# Retrieval quality on `data/retrieval_eval.json` (k={k})",
        "",
        note,
        "",
        "| Metric | " + " | ".join(names) + " |",
        "|---|" + "|".join(["---"] * len(names)) + "|",
    ]
    rows = [("recall", f"Recall@{k}"), ("mrr", "MRR"), ("hit1", "Hit@1")]
    agg = {name: means(res) for name, res in columns.items()}
    for key, label in rows:
        lines.append(f"| {label} | " + " | ".join(f"{agg[name][key]:.3f}" for name in names) + " |")
    lines += ["", "Per query (first gold record's rank, or miss):", ""]
    lines.append("| Query | Gold | " + " | ".join(names) + " |")
    lines.append("|---|---|" + "|".join(["---"] * len(names)) + "|")
    for i, q in enumerate(next(iter(columns.values()))):

        def cell(name: str, idx: int = i) -> str:
            r = columns[name][idx]
            ranks = [str(r.ranked.index(g) + 1) for g in r.gold if g in r.ranked]
            return "rank " + ", ".join(ranks) if ranks else "miss"

        lines.append(f"| {q.id} | {', '.join(q.gold)} | " + " | ".join(cell(name) for name in names) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, help="write the report here as well as to stdout")
    args = ap.parse_args(argv)

    s = Settings()
    store = Store.from_dir(s.data_dir)
    spec = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    queries, k = spec["queries"], spec["k"]

    columns: dict[str, list[QueryResult]] = {
        "Keyword (BM25)": score(queries, lambda a, q: rank_keyword(store, a, q, k, s.min_retrieval_score), k)
    }
    detail = "Keyword mode is BM25 only; hybrid adds the embedding ranker fused by RRF."
    try:
        embedder = OllamaEmbedder(s.ollama_url, s.embed_model)
        columns["Hybrid (BM25 + embeddings)"] = score(queries, lambda a, q: rank_hybrid(store, embedder, a, q, k, s), k)
        detail += f" Embeddings: {embedder.name}."
    except EmbeddingUnavailableError as exc:
        detail += f" Hybrid skipped: embeddings unavailable ({exc})."

    note = (
        f"Run {datetime.now(UTC).date()}. {len(queries)} queries, each asked as its own principal so "
        f"retrieval sees only that caller's visible records. {detail} The query set was committed before "
        "any score was computed."
    )
    report = render(columns, k, note)
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
