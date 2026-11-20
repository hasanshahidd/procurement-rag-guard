"""Fast checks on the retrieval eval: metric maths and the labelled set. No model needed."""

from __future__ import annotations

import json

from app.config import Settings
from app.store import Store
from repro import retrieval_eval as rev


def test_metric_functions_on_known_rankings() -> None:
    gold = ["A", "B"]
    ranked = ["X", "A", "Y", "B"]
    assert rev.recall_at_k(gold, ranked, 4) == 1.0
    assert rev.recall_at_k(gold, ranked, 2) == 0.5  # only A is in the top 2
    assert rev.reciprocal_rank(gold, ranked, 4) == 0.5  # first gold at rank 2
    assert rev.hit_at_1(gold, ranked) == 0.0
    assert rev.hit_at_1(gold, ["A", "B"]) == 1.0
    assert rev.reciprocal_rank(gold, ["X", "Y"], 4) == 0.0  # no gold found
    assert rev.recall_at_k(gold, [], 4) == 0.0


def test_labelled_set_gold_is_visible_current_and_answers_hold() -> None:
    store = Store.from_dir(Settings().data_dir)
    superseded = {old for d in store._snapshot() for old in d.supersedes}
    spec = json.loads(rev.EVAL_PATH.read_text(encoding="utf-8"))
    ids = [q["id"] for q in spec["queries"]]
    assert len(ids) == len(set(ids))
    for q in spec["queries"]:
        principal = store.authenticate(q["api_key"])
        assert principal is not None, q["api_key"]
        visible = {d.id for d in store.visible_docs(principal)}
        assert q["gold"], q["id"]
        for g in q["gold"]:
            assert g in visible, (q["id"], g)  # answerable by this caller
            assert g not in superseded, (q["id"], g)  # gold is never an outdated record


def test_keyword_ranker_finds_obvious_answers() -> None:
    store = Store.from_dir(Settings().data_dir)
    k = json.loads(rev.EVAL_PATH.read_text(encoding="utf-8"))["k"]
    ranked = rev.rank_keyword(store, "demo-harbor-buyer", "status of purchase request PR-1042?", k, 1.0)
    assert "TCK-H-101" in ranked
