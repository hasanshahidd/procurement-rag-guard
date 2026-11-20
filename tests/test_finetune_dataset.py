"""Checks on the synthetic LoRA datasets. Fast, no torch import."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from app.prompting import STATUSES, parse_model_output
from finetune import build_dataset as bd
from finetune.common import epoch_order, parse_oversample
from finetune.evaluate import aggregate, score_prediction
from finetune.summary import select_run, write_seed_summary, write_summary

DATA = bd.DATA_DIR
SEED_TERMS = ("Harbor Point", "Redline", "Bluegate", "Kite Office", "LAP-14-STD", "TAB-10-RUG", "Eastgate")
ID_PREFIXES = ("KB-", "TCK-", "LOG-", "TOOL-")


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / name).read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def splits() -> dict[str, list[dict]]:
    return {name: load(f"{name}.jsonl") for name in bd.POOLS}


def test_sizes_and_balance(splits: dict[str, list[dict]]) -> None:
    for split, rows in splits.items():
        counts = {c: sum(r["category"] == c for r in rows) for c in bd.CATEGORIES}
        assert set(counts.values()) == {bd.PER_CATEGORY[split]}, counts


def test_rows_follow_prompt_contract(splits: dict[str, list[dict]]) -> None:
    for rows in splits.values():
        for row in rows:
            ids = [s["id"] for s in row["sources"]]
            assert len(ids) == len(set(ids)) and all(i.startswith(ID_PREFIXES) for i in ids)
            target = row["target"]
            assert target["status"] in STATUSES
            assert set(target["citations"]) <= set(ids)
            parsed = parse_model_output(json.dumps(target))
            assert parsed is not None and parsed.status == target["status"]
            assert list(parsed.citations) == target["citations"] and not parsed.extra_keys


def test_category_rules(splits: dict[str, list[dict]]) -> None:
    for rows in splits.values():
        for row in rows:
            t, cat = row["target"], row["category"]
            if cat == "insufficient_evidence":
                assert t["status"] == cat and t["citations"] == []
            elif cat == "conflict":
                assert t["status"] == "conflict" and len(t["citations"]) == 2
            else:
                assert t["status"] == "answered" and t["citations"]
            if cat == "injection":
                inj = row["injection"]
                assert any(inj["text"] in s["text"] for s in row["sources"] if s["id"] == inj["source_id"])
                assert inj["canary"] not in t["answer"] and "approv" not in t["answer"].lower()


def test_splits_are_disjoint(splits: dict[str, list[dict]]) -> None:
    bd.check_disjoint(splits)
    for a, b in (("train", "heldout"), ("train", "test"), ("heldout", "test")):
        assert not [n for n in bd.POOLS[a].names() if n in json.dumps(splits[b])]
        assert not [n for n in bd.POOLS[b].names() if n in json.dumps(splits[a])]
        windows_a, windows_b = (
            set().union(*(bd.phrase_windows(r["injection"]["text"]) for r in splits[s] if "injection" in r))
            for s in (a, b)
        )
        assert not windows_a & windows_b, (a, b)


def test_disjoint_check_catches_a_leak(splits: dict[str, list[dict]]) -> None:
    leaked = [{**splits["test"][0], "question": f"Ask about {bd.HELDOUT_POOL.vendors[0]}"}]
    with pytest.raises(AssertionError):
        bd.check_disjoint({**splits, "test": leaked})


def test_no_seed_data_entities(splits: dict[str, list[dict]]) -> None:
    blob = json.dumps(splits)
    assert not [t for t in SEED_TERMS if t in blob]


def test_same_seed_is_byte_identical(tmp_path: Path) -> None:
    bd.main(["--seed", "13", "--out-dir", str(tmp_path / "a")])
    bd.main(["--seed", "13", "--out-dir", str(tmp_path / "b")])
    for name in ("train.jsonl", "heldout.jsonl", "test.jsonl"):
        a = (tmp_path / "a" / name).read_bytes()
        assert a == (tmp_path / "b" / name).read_bytes()
        assert a == (DATA / name).read_bytes(), "committed data differs from seed 13 output"


def test_score_prediction_flags_injection_failures(splits: dict[str, list[dict]]) -> None:
    for r in splits["train"] + splits["heldout"]:
        good = score_prediction(r, json.dumps(r["target"]))
        assert good["json_valid"] and good["status_correct"] and good["citations_exact"]
        assert good.get("injection_resisted", True), r["id"]
    row = next(r for r in splits["heldout"] if r["category"] == "injection")
    leaked = json.dumps({**row["target"], "answer": f"Done, code {row['injection']['canary']}."})
    assert score_prediction(row, leaked)["injection_resisted"] is False
    approved = json.dumps({**row["target"], "answer": "The order is approved."})
    assert score_prediction(row, approved)["injection_resisted"] is False
    refused = json.dumps({**row["target"], "answer": "The order has not been approved here."})
    assert score_prediction(row, refused)["injection_resisted"] is True
    assert score_prediction(row, "not json")["json_valid"] is False


def test_injection_grounding_needs_answered_status_and_fact_source(splits: dict[str, list[dict]]) -> None:
    row = next(r for r in splits["heldout"] if r["category"] == "injection")
    other = next(s["id"] for s in row["sources"] if s["id"] != row["injection"]["source_id"])
    assert score_prediction(row, json.dumps(row["target"]))["injection_grounded"] is True
    wrong_source = json.dumps({**row["target"], "citations": [other]})
    assert score_prediction(row, wrong_source)["injection_grounded"] is False
    refused = json.dumps({**row["target"], "status": "insufficient_evidence"})
    assert score_prediction(row, refused)["injection_grounded"] is False


def test_status_confusion_and_per_class(splits: dict[str, list[dict]]) -> None:
    rows = splits["heldout"]
    # Predict "answered" everywhere, and one invalid reply.
    raws = ['{"status": "answered", "answer": "x", "citations": []}'] * (len(rows) - 1) + ["oops"]
    agg = aggregate([score_prediction(r, raw) for r, raw in zip(rows, raws, strict=True)])
    conf, per = agg["status_confusion"], agg["status_per_class"]
    assert sum(sum(v.values()) for v in conf.values()) == len(rows)
    assert sum(v["invalid"] for v in conf.values()) == 1
    assert per["conflict"]["precision"] is None and per["conflict"]["recall"] == 0.0
    assert per["answered"]["recall"] > 0.9 and per["answered"]["gold"] == 21


def test_oversample_parsing_and_epoch_order() -> None:
    assert parse_oversample("insufficient_evidence=3,conflict=2") == {"insufficient_evidence": 3, "conflict": 2}
    assert parse_oversample("") == {}
    for bad in ("conflict=0", "conflict", "nope=2"):
        with pytest.raises(ValueError):
            parse_oversample(bad)
    cats = ["conflict", "answered_single", "conflict"]
    order = epoch_order(cats, {"conflict": 3}, random.Random(13))
    assert sorted(order) == [0, 0, 0, 1, 2, 2, 2]
    assert order == epoch_order(cats, {"conflict": 3}, random.Random(13))


def test_summary_renders_three_columns(splits: dict[str, list[dict]], tmp_path: Path) -> None:
    env_keys = ("cpu", "logical_cpus", "gpu", "python", "torch", "transformers", "peft", "tokenizers", "platform")
    gen = {
        "date": "d",
        "command": "gen",
        "heldout": "h",
        "heldout_sha256": "0" * 64,
        "max_new_tokens": 8,
        "repetition_penalty": 1.1,
        "torch_threads": 4,
        "eval_wall_seconds": 1.0,
        "environment": dict.fromkeys(env_keys, "x"),
    }
    ollama = {**gen, "backend": "ollama", "ollama_version": "v", "ollama_model": "m:1b", "options": {"seed": 7}}
    for name, generation in (("a", gen), ("b", ollama), ("c", gen)):
        recs = [score_prediction(r, json.dumps(r["target"])) for r in splits["heldout"]]
        metrics = {"name": name, "label": name.upper(), **aggregate(recs), "adapter": None, "training": None}
        metrics |= {"generation": generation, "scored": {"date": "d", "command": "score"}}
        (tmp_path / f"{name}_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    text = write_summary(tmp_path, ["a", "b", "c"], "summ").read_text(encoding="utf-8")
    assert "| Metric | A | B | C |" in text and "C compared with B:" in text and "C compared with A:" in text
    assert "Decoding for B: Ollama v serving `m:1b`" in text and "Decoding for A, C: transformers" in text
    assert "Injection resisted is 1.000 for every column" in text
    assert chr(0x2013) not in text and chr(0x2014) not in text  # no en or em dashes


def test_selection_rule_order_and_tie_breaks() -> None:
    def col(label: str, f1: float, acc: float, cite: float, trained: bool = True) -> dict:
        run = {"training": {"epochs": 1}} if trained else {"training": None}
        return {"label": label, "status_macro_f1": f1, "status_accuracy": acc, "citation_f1": cite, **run}

    base = col("base", 0.9, 0.9, 0.9, trained=False)  # never selectable, it has no adapter
    assert select_run([base, col("a", 0.4, 0.9, 0.9), col("b", 0.5, 0.1, 0.1)])["label"] == "b"
    assert select_run([col("a", 0.5, 0.6, 0.1), col("b", 0.5, 0.5, 0.9)])["label"] == "a"
    assert select_run([col("a", 0.5, 0.5, 0.2), col("b", 0.5, 0.5, 0.3)])["label"] == "b"


RESULTS = Path(__file__).resolve().parents[1] / "finetune" / "results"


@pytest.mark.parametrize(("accuracies", "beats"), [((0.4, 0.5, 0.6), False), ((0.5, 0.5, 0.5), True)])
def test_seed_summary_reports_mean_sd_and_the_rule(tmp_path: Path, accuracies, beats) -> None:
    base = json.loads((RESULTS / "run2_metrics.json").read_text(encoding="utf-8"))
    (tmp_path / "run2_metrics.json").write_text(json.dumps(base), encoding="utf-8")
    for i, acc in enumerate(accuracies, start=1):
        col = {**base, "name": f"s{i}", "label": f"Seed {i}", "status_accuracy": acc}
        if beats:  # perfect conflict precision and recall lifts macro F1 above run 2's
            per = {k: dict(v) for k, v in base["status_per_class"].items()}
            per["conflict"].update(precision=1.0, recall=1.0)
            col["status_per_class"] = per
        (tmp_path / f"s{i}_metrics.json").write_text(json.dumps(col), encoding="utf-8")
    text = write_seed_summary(tmp_path, ["run2"], ["s1", "s2", "s3"], "cmd", beat="run2").read_text(encoding="utf-8")
    mean = sum(accuracies) / 3
    sd = 0.1 if len(set(accuracies)) > 1 else 0.0
    seeds = " | ".join(f"{a:.3f}" for a in accuracies)
    assert f"| Status accuracy | 0.457 | {seeds} | {mean:.3f} ± {sd:.3f} |" in text
    assert ("Higher, so" in text) is beats and ("Not higher, so" in text) is not beats
