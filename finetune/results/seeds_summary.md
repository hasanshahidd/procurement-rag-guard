# LoRA seeds: results on `finetune/data/heldout.jsonl`

Summary written 2026-11-20 from the metrics files in this folder. Every seed uses the same data and settings; sd is the sample standard deviation over 3 seeds.

| Metric | Baseline (no adapter) (rescored) | Served model, qwen2.5:1.5b via Ollama, no fine-tuning | Run 2 (2 epochs, weak classes x2) | Run 4, seed 13 (300 rows, 1 epoch) | Run 4, seed 14 (300 rows, 1 epoch) | Run 4, seed 15 (300 rows, 1 epoch) | Mean ± sd over seeds |
|---|---|---|---|---|---|---|---|
| JSON valid rate (parse_model_output accepts) | 0.914 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 ± 0.000 |
| Status accuracy | 0.543 | 0.657 | 0.457 | 0.486 | 0.771 | 0.600 | 0.619 ± 0.143 |
| Status macro F1 (3 labels) | 0.244 | 0.500 | 0.455 | 0.460 | 0.588 | 0.250 | 0.433 ± 0.171 |
| Status precision: insufficient_evidence | n/a | 0.600 | 0.280 | 0.318 | 1.000 | n/a | 0.659 ± 0.482 |
| Status recall: insufficient_evidence | 0.000 | 0.429 | 1.000 | 1.000 | 0.857 | 0.000 | 0.619 ± 0.541 |
| Status precision: conflict | 0.000 | 0.500 | 1.000 | 0.500 | n/a | n/a | 0.500 ± 0.000 |
| Status recall: conflict | 0.000 | 0.143 | 0.286 | 0.286 | 0.000 | 0.000 | 0.095 ± 0.165 |
| Status precision: answered | 0.613 | 0.679 | 0.875 | 0.889 | 0.724 | 0.600 | 0.738 ± 0.145 |
| Status recall: answered | 0.905 | 0.905 | 0.333 | 0.381 | 1.000 | 1.000 | 0.794 ± 0.357 |
| Citation exact match (set equality) | 0.086 | 0.457 | 0.429 | 0.457 | 0.600 | 0.514 | 0.524 ± 0.072 |
| Citation precision (micro) | 0.000 | 0.610 | 1.000 | 0.882 | 0.909 | 0.786 | 0.859 ± 0.065 |
| Citation recall (micro) | 0.000 | 0.857 | 0.286 | 0.357 | 0.714 | 0.786 | 0.619 ± 0.230 |
| Citation F1 (micro) | 0.000 | 0.713 | 0.444 | 0.508 | 0.800 | 0.786 | 0.698 ± 0.165 |
| Injection resisted rate | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 ± 0.000 |
| Injection answer grounded rate | 0.000 | 1.000 | 0.286 | 0.429 | 0.857 | 1.000 | 0.762 ± 0.297 |
| Status accuracy: answered_single | 1.000 | 0.714 | 0.571 | 0.571 | 1.000 | 1.000 | 0.857 ± 0.248 |
| Status accuracy: answered_multi | 0.714 | 1.000 | 0.143 | 0.143 | 1.000 | 1.000 | 0.714 ± 0.495 |
| Status accuracy: insufficient_evidence | 0.000 | 0.429 | 1.000 | 1.000 | 0.857 | 0.000 | 0.619 ± 0.541 |
| Status accuracy: conflict | 0.000 | 0.143 | 0.286 | 0.286 | 0.000 | 0.000 | 0.095 ± 0.165 |
| Status accuracy: injection | 1.000 | 1.000 | 0.286 | 0.429 | 1.000 | 1.000 | 0.810 ± 0.330 |

## Standard adapter rule

Mean status macro F1 over seeds 0.433 against Run 2 (2 epochs, weak classes x2) 0.455: Not higher, so under the rule fixed before training Run 2 (2 epochs, weak classes x2) stays the standard adapter.

## Command

```
python -m finetune.evaluate --seed-summary run4_s13 run4_s14 run4_s15 --refs baseline served run2 --beat run2
```
