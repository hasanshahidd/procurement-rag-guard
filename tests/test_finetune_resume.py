"""Checkpoint and resume of the LoRA training loop, on a toy model (no model download).

An interrupted run, resumed from its last checkpoint, must end with the same
loss log and the same weights as a run that was never interrupted.
"""

from __future__ import annotations

import argparse
import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")  # train_lora imports the whole ML stack (requirements-ml.txt)
pytest.importorskip("transformers")
from finetune import train_lora  # noqa: E402

CATEGORIES = ["answered_single", "answered_multi", "insufficient_evidence", "conflict", "injection"]


def toy_loss(model, ex: dict):
    return ((model(ex["x"]) - ex["y"]) ** 2).mean()


def examples() -> list[dict]:
    g = torch.Generator().manual_seed(0)
    return [
        {"x": torch.randn(1, 4, generator=g), "y": torch.randn(1, 1, generator=g), "category": c}
        for c in CATEGORIES * 4
    ]  # 20 examples, 10 steps per epoch with grad_accum 2


def toy_model():
    torch.manual_seed(0)
    # Dropout draws from the torch RNG, so a resume is only exact if that state is restored too.
    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(0.3), torch.nn.Linear(8, 1))


def make_args(tmp_path, **overrides) -> argparse.Namespace:
    data = tmp_path / "train.jsonl"
    data.write_text("toy\n", encoding="utf-8")
    base = {
        "train": data,
        "epochs": 2,
        "lr": 0.05,
        "max_len": 1024,
        "limit": 0,
        "grad_accum": 2,
        "oversample": {},
        "rank": 8,
        "alpha": 16,
        "dropout": 0.0,
        "seed": 13,
        "checkpoint_every": 3,
    }
    return argparse.Namespace(**{**base, **overrides})


@pytest.mark.parametrize(
    ("every", "crash_on_call", "resumed_at"),
    [
        (3, 15, 6),  # crash in step 8, mid-epoch
        (5, 23, 10),  # last checkpoint is the end of epoch 1
        (3, 27, 12),  # crash in epoch 2, which has its own shuffled order
    ],
)
def test_interrupted_run_resumes_to_the_same_result(tmp_path, monkeypatch, every, crash_on_call, resumed_at):
    args = make_args(tmp_path, checkpoint_every=every)
    monkeypatch.setattr(train_lora, "reply_loss", toy_loss)
    reference = toy_model()
    full = train_lora.train(reference, examples(), args, None)

    calls = {"n": 0}

    def crashing_loss(model, ex):
        calls["n"] += 1
        if calls["n"] == crash_on_call:
            raise KeyboardInterrupt  # what a closed session or Ctrl+C does to the process
        return toy_loss(model, ex)

    ckpt = tmp_path / "checkpoint.pt"
    monkeypatch.setattr(train_lora, "reply_loss", crashing_loss)
    with pytest.raises(KeyboardInterrupt):
        train_lora.train(toy_model(), examples(), args, ckpt)
    assert ckpt.exists()

    monkeypatch.setattr(train_lora, "reply_loss", toy_loss)
    restarted = toy_model()  # a new process starts from the base weights again
    resumed = train_lora.train(restarted, examples(), args, ckpt)

    assert resumed["resumed_at_steps"] == [resumed_at]
    assert [e["loss"] for e in resumed["log"]] == [e["loss"] for e in full["log"]]
    for got, want in zip(restarted.parameters(), reference.parameters(), strict=True):
        assert torch.allclose(got, want)


def test_checkpoint_from_other_settings_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(train_lora, "reply_loss", toy_loss)
    ckpt = tmp_path / "checkpoint.pt"
    args = make_args(tmp_path, epochs=1, checkpoint_every=2)
    model = toy_model()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    position = {"signature": train_lora.checkpoint_signature(args), "epoch": 1, "start": 2, "order": [], "log": []}
    train_lora.save_checkpoint(ckpt, model, opt, sched, random.Random(1), {**position, "resumed": [], "seconds": 0})
    with pytest.raises(SystemExit, match="different settings"):
        train_lora.train(toy_model(), examples(), make_args(tmp_path, epochs=1, lr=0.01, checkpoint_every=2), ckpt)
