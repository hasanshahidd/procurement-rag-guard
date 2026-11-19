"""Checks on the synthetic LoRA datasets. Fast, no torch import."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.prompting import STATUSES, parse_model_output
from finetune import build_dataset as bd
from finetune.evaluate import score_prediction

DATA = bd.DATA_DIR
SEED_TERMS = ("Harbor Point", "Redline", "Bluegate", "Kite Office", "LAP-14-STD", "TAB-10-RUG", "Eastgate")
ID_PREFIXES = ("KB-", "TCK-", "LOG-", "TOOL-")


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / name).read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def splits() -> dict[str, list[dict]]:
    return {"train": load("train.jsonl"), "heldout": load("heldout.jsonl")}


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
    bd.check_disjoint(splits["train"], splits["heldout"])
    train_blob, heldout_blob = (json.dumps(splits[s]) for s in ("train", "heldout"))
    assert not [n for n in bd.TRAIN_POOL.names() if n in heldout_blob]
    assert not [n for n in bd.HELDOUT_POOL.names() if n in train_blob]
    train_payloads = {r["injection"]["text"] for r in splits["train"] if r["category"] == "injection"}
    heldout_windows = set().union(
        *(bd.phrase_windows(r["injection"]["text"]) for r in splits["heldout"] if "injection" in r)
    )
    assert not heldout_windows & set().union(*(bd.phrase_windows(p) for p in train_payloads))


def test_no_seed_data_entities(splits: dict[str, list[dict]]) -> None:
    blob = json.dumps(splits)
    assert not [t for t in SEED_TERMS if t in blob]


def test_same_seed_is_byte_identical(tmp_path: Path) -> None:
    bd.main(["--seed", "13", "--out-dir", str(tmp_path / "a")])
    bd.main(["--seed", "13", "--out-dir", str(tmp_path / "b")])
    for name in ("train.jsonl", "heldout.jsonl"):
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
