# LoRA demo: results on `finetune/data/heldout.jsonl`

Summary written 2026-11-19. Every number comes from the metrics files in this folder.
Hardware: 11th Gen Intel(R) Core(TM) i5-1145G7 @ 2.60GHz, 8 logical CPUs, none (CPU only).
Versions: Python 3.11.9, torch 2.15.0+cpu, transformers 5.23.0, peft 0.21.3, tokenizers 0.23.3 (Windows-10-10.0.26200-SP0).
Evaluation set: `finetune/data/heldout.jsonl` (35 rows, sha256 `e9718bc671ce5711...`), the same file for every column.
Base model revision (transformers columns): `7ae557604adf67be50417f59c2c2f167def9a775`, pinned in finetune/common.py. Columns written before the pin loaded the same snapshot, the only one in the local Hugging Face cache.
Decoding for Baseline (no adapter), Run 1 (1 epoch), Run 2 (2 epochs, weak classes x2), Run 3 (2 epochs, natural mix): transformers, `Qwen/Qwen2.5-0.5B-Instruct` at revision `7ae557604adf`, greedy (do_sample=False, same call as app/llm.py, so the checkpoint's repetition_penalty=1.1 applies), max_new_tokens=256, fp32 on CPU, 4 torch threads.
Decoding for Served model, qwen2.5:1.5b via Ollama, no fine-tuning: Ollama 0.40.16 serving `qwen2.5:1.5b` (1.5B, Q4_K_M), /api/chat with format=json and options {"temperature": 0, "seed": 7, "num_predict": 400, "num_ctx": 8192} (the app/llm.py options plus num_ctx).
Rescored columns (Baseline (no adapter), Run 1 (1 epoch)): the current scorer was run on replies saved by the earlier generation run (no new generation). The held-out file and decoding settings are unchanged.

The served model column is what the API uses by default (LLM_BACKEND=ollama with qwen2.5:1.5b, 4-bit quantised, Ollama JSON mode). The LoRA was trained on the 0.5B model because one run fits in under an hour on this 4-core CPU; the adapter is served with LLM_BACKEND=transformers and LORA_ADAPTER=finetune/outputs/qwen2.5-0.5b-procurement-lora. Ollama 0.40.16 refused to import the adapter (Error: LoRA adapters are no longer supported), so the adapter has no Ollama column.

| Metric | Baseline (no adapter) (rescored) | Served model, qwen2.5:1.5b via Ollama, no fine-tuning | Run 1 (1 epoch) (rescored) | Run 2 (2 epochs, weak classes x2) | Run 3 (2 epochs, natural mix) |
|---|---|---|---|---|---|
| JSON valid rate (parse_model_output accepts) | 0.914 | 1.000 | 1.000 | 1.000 | 1.000 |
| Status accuracy | 0.543 | 0.657 | 0.600 | 0.457 | 0.600 |
| Status macro F1 (3 labels) | 0.244 | 0.500 | 0.250 | 0.455 | 0.250 |
| Status precision: insufficient_evidence | n/a | 0.600 | n/a | 0.280 | n/a |
| Status recall: insufficient_evidence | 0.000 | 0.429 | 0.000 | 1.000 | 0.000 |
| Status precision: conflict | 0.000 | 0.500 | n/a | 1.000 | n/a |
| Status recall: conflict | 0.000 | 0.143 | 0.000 | 0.286 | 0.000 |
| Status precision: answered | 0.613 | 0.679 | 0.600 | 0.875 | 0.600 |
| Status recall: answered | 0.905 | 0.905 | 1.000 | 0.333 | 1.000 |
| Citation exact match (set equality) | 0.086 | 0.457 | 0.343 | 0.429 | 0.457 |
| Citation precision (micro) | 0.000 | 0.610 | 0.679 | 1.000 | 0.829 |
| Citation recall (micro) | 0.000 | 0.857 | 0.452 | 0.286 | 0.690 |
| Citation F1 (micro) | 0.000 | 0.713 | 0.543 | 0.444 | 0.753 |
| Injection resisted rate (n=7) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Injection answer grounded rate (n=7) | 0.000 | 1.000 | 0.429 | 0.286 | 1.000 |
| Status accuracy: answered_single | 1.000 | 0.714 | 1.000 | 0.571 | 1.000 |
| Status accuracy: answered_multi | 0.714 | 1.000 | 1.000 | 0.143 | 1.000 |
| Status accuracy: insufficient_evidence | 0.000 | 0.429 | 0.000 | 1.000 | 0.000 |
| Status accuracy: conflict | 0.000 | 0.143 | 0.000 | 0.286 | 0.000 |
| Status accuracy: injection | 1.000 | 1.000 | 1.000 | 0.286 | 1.000 |
| Eval wall time (s) | 1137.8 | 362.0 | 593.6 | 347.7 | 754.3 |

Injection resisted is 1.000 for every column, the baseline included, so it does not separate the models on this set.

## Selection

Selection rule (decided before run 3 was evaluated): the standard adapter at `finetune/outputs/qwen2.5-0.5b-procurement-lora/` is the trained run with the highest status macro F1 on this set; ties are broken by status accuracy, then citation F1.

| Run | Status macro F1 | Status accuracy | Citation F1 |
|---|---|---|---|
| Run 2 (2 epochs, weak classes x2) | 0.455 | 0.457 | 0.444 |
| Run 3 (2 epochs, natural mix) | 0.250 | 0.600 | 0.753 |
| Run 1 (1 epoch) | 0.250 | 0.600 | 0.543 |

Selected: Run 2 (2 epochs, weak classes x2).

Because this set was used to choose between runs, it acts as a validation set and the chosen run's numbers here are optimistic. An unbiased estimate needs a split that played no part in the choice: see `finetune/data/test.jsonl` and `finetune/results/test_split/summary.md`.

## Status confusion matrices (rows: gold status, columns: predicted status)

Baseline (no adapter) (rescored):

| gold \ predicted | answered | insufficient_evidence | conflict | invalid |
|---|---|---|---|---|
| answered | 19 | 0 | 0 | 2 |
| insufficient_evidence | 5 | 0 | 1 | 1 |
| conflict | 7 | 0 | 0 | 0 |

Served model, qwen2.5:1.5b via Ollama, no fine-tuning:

| gold \ predicted | answered | insufficient_evidence | conflict | invalid |
|---|---|---|---|---|
| answered | 19 | 2 | 0 | 0 |
| insufficient_evidence | 3 | 3 | 1 | 0 |
| conflict | 6 | 0 | 1 | 0 |

Run 1 (1 epoch) (rescored):

| gold \ predicted | answered | insufficient_evidence | conflict | invalid |
|---|---|---|---|---|
| answered | 21 | 0 | 0 | 0 |
| insufficient_evidence | 7 | 0 | 0 | 0 |
| conflict | 7 | 0 | 0 | 0 |

Run 2 (2 epochs, weak classes x2):

| gold \ predicted | answered | insufficient_evidence | conflict | invalid |
|---|---|---|---|---|
| answered | 7 | 14 | 0 | 0 |
| insufficient_evidence | 0 | 7 | 0 | 0 |
| conflict | 1 | 4 | 2 | 0 |

Run 3 (2 epochs, natural mix):

| gold \ predicted | answered | insufficient_evidence | conflict | invalid |
|---|---|---|---|---|
| answered | 21 | 0 | 0 | 0 |
| insufficient_evidence | 7 | 0 | 0 | 0 |
| conflict | 7 | 0 | 0 | 0 |

## Training

| Column | Epochs | lr | Grad accum | Oversample | Examples per epoch | Optimizer steps | Final step loss | Train wall time |
|---|---|---|---|---|---|---|---|---|
| Run 1 (1 epoch) | 1 | 0.0003 | 2 | none | 90 | 45 | 0.1951 | 27.3 min |
| Run 2 (2 epochs, weak classes x2) | 2 | 0.0002 | 2 | insufficient_evidence x2, conflict x2 | 126 | 126 | 0.0233 | 57.2 min |
| Run 3 (2 epochs, natural mix) | 2 | 0.0002 | 2 | none | 90 | 90 | 0.192 | 41.8 min |

## What improved and what did not

Run 2 (2 epochs, weak classes x2) compared with Run 3 (2 epochs, natural mix):
- Higher: Status macro F1 (3 labels) (0.250 -> 0.455); Status recall: insufficient_evidence (0.000 -> 1.000); Status recall: conflict (0.000 -> 0.286); Status precision: answered (0.600 -> 0.875); Citation precision (micro) (0.829 -> 1.000); Status accuracy: insufficient_evidence (0.000 -> 1.000); Status accuracy: conflict (0.000 -> 0.286)
- Lower: Status accuracy (0.600 -> 0.457); Status recall: answered (1.000 -> 0.333); Citation exact match (set equality) (0.457 -> 0.429); Citation recall (micro) (0.690 -> 0.286); Citation F1 (micro) (0.753 -> 0.444); Injection answer grounded rate (1.000 -> 0.286); Status accuracy: answered_single (1.000 -> 0.571); Status accuracy: answered_multi (1.000 -> 0.143); Status accuracy: injection (1.000 -> 0.286)
- Unchanged: JSON valid rate (parse_model_output accepts) (1.000 -> 1.000); Injection resisted rate (1.000 -> 1.000)
- Not comparable (n/a): Status precision: insufficient_evidence (n/a -> 0.280); Status precision: conflict (n/a -> 1.000)

Run 2 (2 epochs, weak classes x2) compared with Run 1 (1 epoch):
- Higher: Status macro F1 (3 labels) (0.250 -> 0.455); Status recall: insufficient_evidence (0.000 -> 1.000); Status recall: conflict (0.000 -> 0.286); Status precision: answered (0.600 -> 0.875); Citation exact match (set equality) (0.343 -> 0.429); Citation precision (micro) (0.679 -> 1.000); Status accuracy: insufficient_evidence (0.000 -> 1.000); Status accuracy: conflict (0.000 -> 0.286)
- Lower: Status accuracy (0.600 -> 0.457); Status recall: answered (1.000 -> 0.333); Citation recall (micro) (0.452 -> 0.286); Citation F1 (micro) (0.543 -> 0.444); Injection answer grounded rate (0.429 -> 0.286); Status accuracy: answered_single (1.000 -> 0.571); Status accuracy: answered_multi (1.000 -> 0.143); Status accuracy: injection (1.000 -> 0.286)
- Unchanged: JSON valid rate (parse_model_output accepts) (1.000 -> 1.000); Injection resisted rate (1.000 -> 1.000)
- Not comparable (n/a): Status precision: insufficient_evidence (n/a -> 0.280); Status precision: conflict (n/a -> 1.000)

Run 2 (2 epochs, weak classes x2) compared with Served model, qwen2.5:1.5b via Ollama, no fine-tuning:
- Higher: Status recall: insufficient_evidence (0.429 -> 1.000); Status precision: conflict (0.500 -> 1.000); Status recall: conflict (0.143 -> 0.286); Status precision: answered (0.679 -> 0.875); Citation precision (micro) (0.610 -> 1.000); Status accuracy: insufficient_evidence (0.429 -> 1.000); Status accuracy: conflict (0.143 -> 0.286)
- Lower: Status accuracy (0.657 -> 0.457); Status macro F1 (3 labels) (0.500 -> 0.455); Status precision: insufficient_evidence (0.600 -> 0.280); Status recall: answered (0.905 -> 0.333); Citation exact match (set equality) (0.457 -> 0.429); Citation recall (micro) (0.857 -> 0.286); Citation F1 (micro) (0.713 -> 0.444); Injection answer grounded rate (1.000 -> 0.286); Status accuracy: answered_single (0.714 -> 0.571); Status accuracy: answered_multi (1.000 -> 0.143); Status accuracy: injection (1.000 -> 0.286)
- Unchanged: JSON valid rate (parse_model_output accepts) (1.000 -> 1.000); Injection resisted rate (1.000 -> 1.000)

Run 2 (2 epochs, weak classes x2) compared with Baseline (no adapter):
- Higher: JSON valid rate (parse_model_output accepts) (0.914 -> 1.000); Status macro F1 (3 labels) (0.244 -> 0.455); Status recall: insufficient_evidence (0.000 -> 1.000); Status precision: conflict (0.000 -> 1.000); Status recall: conflict (0.000 -> 0.286); Status precision: answered (0.613 -> 0.875); Citation exact match (set equality) (0.086 -> 0.429); Citation precision (micro) (0.000 -> 1.000); Citation recall (micro) (0.000 -> 0.286); Citation F1 (micro) (0.000 -> 0.444); Injection answer grounded rate (0.000 -> 0.286); Status accuracy: insufficient_evidence (0.000 -> 1.000); Status accuracy: conflict (0.000 -> 0.286)
- Lower: Status accuracy (0.543 -> 0.457); Status recall: answered (0.905 -> 0.333); Status accuracy: answered_single (1.000 -> 0.571); Status accuracy: answered_multi (0.714 -> 0.143); Status accuracy: injection (1.000 -> 0.286)
- Unchanged: Injection resisted rate (1.000 -> 1.000)
- Not comparable (n/a): Status precision: insufficient_evidence (n/a -> 0.280)

## Commands

Commands as recorded, run from the repo root. Rescored columns list the command that generated their replies (an earlier evaluate.py that ran the base model and the adapter in one call) and the rescoring command.

```
python -m finetune.build_dataset --seed 13
python -m finetune.train_lora --epochs 1 --lr 3e-4 --grad-accum 2 --threads 4
python -m finetune.train_lora --epochs 2 --lr 2e-4 --grad-accum 2 --threads 4 --oversample insufficient_evidence=2,conflict=2
python -m finetune.train_lora --epochs 2 --lr 2e-4 --grad-accum 2 --threads 4 --output-dir finetune/outputs/run3_2epochs/qwen2.5-0.5b-procurement-lora
python -m finetune.evaluate --threads 4
python -m finetune.evaluate --ollama-model qwen2.5:1.5b --name served --label "Served model, qwen2.5:1.5b via Ollama, no fine-tuning"
python -m finetune.evaluate --name run2 --label "Run 2 (2 epochs, weak classes x2)" --threads 4
python -m finetune.evaluate --adapter finetune/outputs/run3_2epochs/qwen2.5-0.5b-procurement-lora --name run3 --label "Run 3 (2 epochs, natural mix)" --threads 4
python -m finetune.evaluate --rescore finetune/results/run1_1epoch/predictions_baseline.jsonl --adapter none --name baseline --label "Baseline (no adapter)"
python -m finetune.evaluate --rescore finetune/results/run1_1epoch/predictions_adapted.jsonl --adapter finetune/outputs/run1_1epoch/qwen2.5-0.5b-procurement-lora --name run1 --label "Run 1 (1 epoch)"
python -m finetune.evaluate --summary baseline served run1 run2 run3 --note "The served model column is what the API uses by default (LLM_BACKEND=ollama with qwen2.5:1.5b, 4-bit quantised, Ollama JSON mode). The LoRA was trained on the 0.5B model because one run fits in under an hour on this 4-core CPU; the adapter is served with LLM_BACKEND=transformers and LORA_ADAPTER=finetune/outputs/qwen2.5-0.5b-procurement-lora. Ollama 0.40.16 refused to import the adapter (Error: LoRA adapters are no longer supported), so the adapter has no Ollama column."
```

The set is small and synthetic (invented companies, SKUs and injection wording that never appear in train), so these numbers are indicative only. One row moves an overall rate by 0.029, the injection rates by 0.143 and the conflict recall by 0.143 (7 gold rows). Each column is a single run on a shared CPU. Status macro F1 is computed here from the per-label precision and recall in each metrics file, counting an undefined precision (label never predicted) as 0.
