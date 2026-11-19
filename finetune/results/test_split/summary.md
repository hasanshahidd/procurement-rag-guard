# LoRA demo: results on `finetune/data/test.jsonl`

Summary written 2026-11-19. Every number comes from the metrics files in this folder.
Hardware: 11th Gen Intel(R) Core(TM) i5-1145G7 @ 2.60GHz, 8 logical CPUs, none (CPU only).
Versions: Python 3.11.9, torch 2.15.0+cpu, transformers 5.23.0, peft 0.21.3, tokenizers 0.23.3 (Windows-10-10.0.26200-SP0).
Evaluation set: `finetune/data/test.jsonl` (35 rows, sha256 `91c833e16f2e021e...`), the same file for every column.
Base model revision (transformers columns): `7ae557604adf67be50417f59c2c2f167def9a775`, pinned in finetune/common.py. Columns written before the pin loaded the same snapshot, the only one in the local Hugging Face cache.
Decoding for Baseline (no adapter), Run 2, selected (2 epochs, weak classes x2): transformers, `Qwen/Qwen2.5-0.5B-Instruct` at revision `7ae557604adf`, greedy (do_sample=False, same call as app/llm.py, so the checkpoint's repetition_penalty=1.1 applies), max_new_tokens=256, fp32 on CPU, 4 torch threads.
Decoding for Served model, qwen2.5:1.5b via Ollama, no fine-tuning: Ollama 0.40.16 serving `qwen2.5:1.5b` (1.5B, Q4_K_M), /api/chat with format=json and options {"temperature": 0, "seed": 7, "num_predict": 400, "num_ctx": 8192} (the app/llm.py options plus num_ctx).

This split was generated after run 2 with its own disjoint pool of names, SKUs and injection wording, and played no part in choosing the adapter. Only the baseline, the served model and the selected adapter (run 2) were scored on it, once each, so these are the unbiased numbers.

| Metric | Baseline (no adapter) | Served model, qwen2.5:1.5b via Ollama, no fine-tuning | Run 2, selected (2 epochs, weak classes x2) |
|---|---|---|---|
| JSON valid rate (parse_model_output accepts) | 0.971 | 0.971 | 1.000 |
| Status accuracy | 0.571 | 0.600 | 0.400 |
| Status macro F1 (3 labels) | 0.247 | 0.412 | 0.297 |
| Status precision: insufficient_evidence | n/a | 0.600 | 0.269 |
| Status recall: insufficient_evidence | 0.000 | 0.429 | 1.000 |
| Status precision: conflict | 0.000 | 0.000 | n/a |
| Status recall: conflict | 0.000 | 0.000 | 0.000 |
| Status precision: answered | 0.606 | 0.643 | 0.778 |
| Status recall: answered | 0.952 | 0.857 | 0.333 |
| Citation exact match (set equality) | 0.171 | 0.514 | 0.314 |
| Citation precision (micro) | 0.000 | 0.650 | 1.000 |
| Citation recall (micro) | 0.000 | 0.929 | 0.214 |
| Citation F1 (micro) | 0.000 | 0.765 | 0.353 |
| Injection resisted rate (n=7) | 0.857 | 0.857 | 1.000 |
| Injection answer grounded rate (n=7) | 0.000 | 1.000 | 0.429 |
| Status accuracy: answered_single | 0.857 | 0.714 | 0.143 |
| Status accuracy: answered_multi | 1.000 | 0.857 | 0.429 |
| Status accuracy: insufficient_evidence | 0.000 | 0.429 | 1.000 |
| Status accuracy: conflict | 0.000 | 0.000 | 0.000 |
| Status accuracy: injection | 1.000 | 1.000 | 0.429 |
| Eval wall time (s) | 1061.8 | 7691.6 | 604.7 |

## Status confusion matrices (rows: gold status, columns: predicted status)

Baseline (no adapter):

| gold \ predicted | answered | insufficient_evidence | conflict | invalid |
|---|---|---|---|---|
| answered | 20 | 0 | 1 | 0 |
| insufficient_evidence | 6 | 0 | 0 | 1 |
| conflict | 7 | 0 | 0 | 0 |

Served model, qwen2.5:1.5b via Ollama, no fine-tuning:

| gold \ predicted | answered | insufficient_evidence | conflict | invalid |
|---|---|---|---|---|
| answered | 18 | 2 | 1 | 0 |
| insufficient_evidence | 3 | 3 | 0 | 1 |
| conflict | 7 | 0 | 0 | 0 |

Run 2, selected (2 epochs, weak classes x2):

| gold \ predicted | answered | insufficient_evidence | conflict | invalid |
|---|---|---|---|---|
| answered | 7 | 14 | 0 | 0 |
| insufficient_evidence | 0 | 7 | 0 | 0 |
| conflict | 2 | 5 | 0 | 0 |

## Training

| Column | Epochs | lr | Grad accum | Oversample | Examples per epoch | Optimizer steps | Final step loss | Train wall time |
|---|---|---|---|---|---|---|---|---|
| Run 2, selected (2 epochs, weak classes x2) | 2 | 0.0002 | 2 | insufficient_evidence x2, conflict x2 | 126 | 126 | 0.0233 | 57.2 min |

## What improved and what did not

Run 2, selected (2 epochs, weak classes x2) compared with Served model, qwen2.5:1.5b via Ollama, no fine-tuning:
- Higher: JSON valid rate (parse_model_output accepts) (0.971 -> 1.000); Status recall: insufficient_evidence (0.429 -> 1.000); Status precision: answered (0.643 -> 0.778); Citation precision (micro) (0.650 -> 1.000); Injection resisted rate (0.857 -> 1.000); Status accuracy: insufficient_evidence (0.429 -> 1.000)
- Lower: Status accuracy (0.600 -> 0.400); Status macro F1 (3 labels) (0.412 -> 0.297); Status precision: insufficient_evidence (0.600 -> 0.269); Status recall: answered (0.857 -> 0.333); Citation exact match (set equality) (0.514 -> 0.314); Citation recall (micro) (0.929 -> 0.214); Citation F1 (micro) (0.765 -> 0.353); Injection answer grounded rate (1.000 -> 0.429); Status accuracy: answered_single (0.714 -> 0.143); Status accuracy: answered_multi (0.857 -> 0.429); Status accuracy: injection (1.000 -> 0.429)
- Unchanged: Status recall: conflict (0.000 -> 0.000); Status accuracy: conflict (0.000 -> 0.000)
- Not comparable (n/a): Status precision: conflict (0.000 -> n/a)

Run 2, selected (2 epochs, weak classes x2) compared with Baseline (no adapter):
- Higher: JSON valid rate (parse_model_output accepts) (0.971 -> 1.000); Status macro F1 (3 labels) (0.247 -> 0.297); Status recall: insufficient_evidence (0.000 -> 1.000); Status precision: answered (0.606 -> 0.778); Citation exact match (set equality) (0.171 -> 0.314); Citation precision (micro) (0.000 -> 1.000); Citation recall (micro) (0.000 -> 0.214); Citation F1 (micro) (0.000 -> 0.353); Injection resisted rate (0.857 -> 1.000); Injection answer grounded rate (0.000 -> 0.429); Status accuracy: insufficient_evidence (0.000 -> 1.000)
- Lower: Status accuracy (0.571 -> 0.400); Status recall: answered (0.952 -> 0.333); Status accuracy: answered_single (0.857 -> 0.143); Status accuracy: answered_multi (1.000 -> 0.429); Status accuracy: injection (1.000 -> 0.429)
- Unchanged: Status recall: conflict (0.000 -> 0.000); Status accuracy: conflict (0.000 -> 0.000)
- Not comparable (n/a): Status precision: insufficient_evidence (n/a -> 0.269); Status precision: conflict (0.000 -> n/a)

## Commands

Commands as recorded, run from the repo root. Rescored columns list the command that generated their replies (an earlier evaluate.py that ran the base model and the adapter in one call) and the rescoring command.

```
python -m finetune.build_dataset --seed 13
python -m finetune.train_lora --epochs 2 --lr 2e-4 --grad-accum 2 --threads 4 --oversample insufficient_evidence=2,conflict=2
python -m finetune.evaluate --heldout finetune/data/test.jsonl --adapter none --name baseline --label "Baseline (no adapter)" --out-dir finetune/results/test_split --threads 4
python -m finetune.evaluate --heldout finetune/data/test.jsonl --ollama-model qwen2.5:1.5b --name served --label "Served model, qwen2.5:1.5b via Ollama, no fine-tuning" --out-dir finetune/results/test_split
python -m finetune.evaluate --heldout finetune/data/test.jsonl --adapter finetune/outputs/qwen2.5-0.5b-procurement-lora --name selected --label "Run 2, selected (2 epochs, weak classes x2)" --out-dir finetune/results/test_split --threads 4
python -m finetune.evaluate --summary baseline served selected --out-dir finetune/results/test_split --note "This split was generated after run 2 with its own disjoint pool of names, SKUs and injection wording, and played no part in choosing the adapter. Only the baseline, the served model and the selected adapter (run 2) were scored on it, once each, so these are the unbiased numbers."
```

The set is small and synthetic (invented companies, SKUs and injection wording that never appear in train), so these numbers are indicative only. One row moves an overall rate by 0.029, the injection rates by 0.143 and the conflict recall by 0.143 (7 gold rows). Each column is a single run on a shared CPU. Status macro F1 is computed here from the per-label precision and recall in each metrics file, counting an undefined precision (label never predicted) as 0.
