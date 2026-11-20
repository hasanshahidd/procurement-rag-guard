# LoRA seeds: results on `finetune/data/test.jsonl`

Summary written 2026-11-20 from the metrics files in this folder. Every seed uses the same data and settings; sd is the sample standard deviation over 3 seeds.

| Metric | Baseline (no adapter) | Served model, qwen2.5:1.5b via Ollama, no fine-tuning | Run 2, selected (2 epochs, weak classes x2) | Run 4, seed 13 (300 rows, 1 epoch) | Run 4, seed 14 (300 rows, 1 epoch) | Run 4, seed 15 (300 rows, 1 epoch) | Mean ± sd over seeds |
|---|---|---|---|---|---|---|---|
| JSON valid rate (parse_model_output accepts) | 0.971 | 0.971 | 1.000 | 0.971 | 1.000 | 1.000 | 0.990 ± 0.017 |
| Status accuracy | 0.571 | 0.600 | 0.400 | 0.457 | 0.600 | 0.600 | 0.552 ± 0.083 |
| Status macro F1 (3 labels) | 0.247 | 0.412 | 0.297 | 0.405 | 0.250 | 0.250 | 0.302 ± 0.089 |
| Status precision: insufficient_evidence | n/a | 0.600 | 0.269 | 0.318 | n/a | n/a | 0.318 ± 0.000 |
| Status recall: insufficient_evidence | 0.000 | 0.429 | 1.000 | 1.000 | 0.000 | 0.000 | 0.333 ± 0.577 |
| Status precision: conflict | 0.000 | 0.000 | n/a | 0.333 | n/a | n/a | 0.333 ± 0.000 |
| Status recall: conflict | 0.000 | 0.000 | 0.000 | 0.143 | 0.000 | 0.000 | 0.048 ± 0.083 |
| Status precision: answered | 0.606 | 0.643 | 0.778 | 0.889 | 0.600 | 0.600 | 0.696 ± 0.167 |
| Status recall: answered | 0.952 | 0.857 | 0.333 | 0.381 | 1.000 | 1.000 | 0.794 ± 0.357 |
| Citation exact match (set equality) | 0.171 | 0.514 | 0.314 | 0.314 | 0.429 | 0.457 | 0.400 ± 0.076 |
| Citation precision (micro) | 0.000 | 0.650 | 1.000 | 0.857 | 0.744 | 0.732 | 0.778 ± 0.069 |
| Citation recall (micro) | 0.000 | 0.929 | 0.214 | 0.286 | 0.690 | 0.714 | 0.563 ± 0.240 |
| Citation F1 (micro) | 0.000 | 0.765 | 0.353 | 0.429 | 0.716 | 0.723 | 0.623 ± 0.168 |
| Injection resisted rate | 0.857 | 0.857 | 1.000 | 0.857 | 1.000 | 1.000 | 0.952 ± 0.083 |
| Injection answer grounded rate | 0.000 | 1.000 | 0.429 | 0.429 | 1.000 | 1.000 | 0.810 ± 0.330 |
| Status accuracy: answered_single | 0.857 | 0.714 | 0.143 | 0.143 | 1.000 | 1.000 | 0.714 ± 0.495 |
| Status accuracy: answered_multi | 1.000 | 0.857 | 0.429 | 0.571 | 1.000 | 1.000 | 0.857 ± 0.248 |
| Status accuracy: insufficient_evidence | 0.000 | 0.429 | 1.000 | 1.000 | 0.000 | 0.000 | 0.333 ± 0.577 |
| Status accuracy: conflict | 0.000 | 0.000 | 0.000 | 0.143 | 0.000 | 0.000 | 0.048 ± 0.083 |
| Status accuracy: injection | 1.000 | 1.000 | 0.429 | 0.429 | 1.000 | 1.000 | 0.810 ± 0.330 |

## Command

```
python -m finetune.evaluate --seed-summary run4_s13 run4_s14 run4_s15 --refs baseline served selected --out-dir finetune/results/test_split
```
