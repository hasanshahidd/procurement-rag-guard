# Retrieval quality on `data/retrieval_eval.json` (k=4)

Run 2026-11-20. 12 queries, each asked as its own principal so retrieval sees only that caller's visible records. Keyword mode is BM25 only; hybrid adds the embedding ranker fused by RRF. Embeddings: ollama:nomic-embed-text. The query set was committed before any score was computed.

| Metric | Keyword (BM25) | Hybrid (BM25 + embeddings) |
|---|---|---|
| Recall@4 | 1.000 | 1.000 |
| MRR | 0.875 | 0.875 |
| Hit@1 | 0.750 | 0.750 |

Per query (first gold record's rank, or miss):

| Query | Gold | Keyword (BM25) | Hybrid (BM25 + embeddings) |
|---|---|---|---|
| q01 | TCK-H-101 | rank 1 | rank 1 |
| q02 | PRC-H-002 | rank 2 | rank 2 |
| q03 | KB-001 | rank 2 | rank 2 |
| q04 | KB-003 | rank 1 | rank 1 |
| q05 | TCK-H-102, LOG-H-201 | rank 1, 2 | rank 1, 4 |
| q06 | KB-005 | rank 1 | rank 1 |
| q07 | KB-004 | rank 1 | rank 1 |
| q08 | TCK-H-101 | rank 1 | rank 1 |
| q09 | KB-001 | rank 2 | rank 2 |
| q10 | PRC-R-001 | rank 1 | rank 1 |
| q11 | TCK-R-101 | rank 1 | rank 1 |
| q12 | LOG-R-201 | rank 1 | rank 1 |
