# LoRA demo: held-out results

Date: 2026-11-19. Every number below comes from the run recorded in this folder.
Hardware: 11th Gen Intel(R) Core(TM) i5-1145G7 @ 2.60GHz, 8 logical CPUs, none (CPU only), torch threads 4.
Versions: Python 3.11.9, torch 2.15.0+cpu, transformers 5.23.0, peft 0.21.3, tokenizers 0.23.3 (Windows-10-10.0.26200-SP0).
Held-out set: `finetune/data/heldout.jsonl` (35 rows, sha256 `e9718bc671ce5711...`).
Decoding: greedy (do_sample=False, same call as app/llm.py, so the checkpoint's repetition_penalty=1.1 applies), max_new_tokens=256, fp32 on CPU.

| Metric | Baseline (Qwen2.5-0.5B-Instruct, no adapter) | Adapted (+ LoRA) |
|---|---|---|
| JSON valid rate (parse_model_output accepts) | 0.914 | 1.000 |
| Status accuracy | 0.543 | 0.600 |
| Citation exact match (set equality) | 0.086 | 0.343 |
| Citation precision (micro) | 0.000 | 0.679 |
| Citation recall (micro) | 0.000 | 0.452 |
| Citation F1 (micro) | 0.000 | 0.543 |
| Injection resisted rate (n=7) | 1.000 | 1.000 |
| Status accuracy: answered_single | 1.000 | 1.000 |
| Status accuracy: answered_multi | 0.714 | 1.000 |
| Status accuracy: insufficient_evidence | 0.000 | 0.000 |
| Status accuracy: conflict | 0.000 | 0.000 |
| Status accuracy: injection | 1.000 | 1.000 |
| Eval wall time (s) | 1137.8 | 593.6 |

Commands (run from the repo root):

```
python -m finetune.build_dataset --seed 13
python -m finetune.train_lora --epochs 1 --lr 3e-4 --grad-accum 2 --threads 4
python -m finetune.evaluate --threads 4
```

Training: 90 examples, 45 optimizer steps, final step loss 0.1951, wall time 27.3 min.

The held-out set is small and synthetic (invented companies, SKUs and injection wording that never appear in train), so these numbers are indicative only. A difference of one example is 0.029 on the overall rates and 0.143 on the injection rate.
