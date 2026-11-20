"""LoRA fine-tune of Qwen2.5-0.5B-Instruct on the synthetic procurement set.

Plain PyTorch loop on CPU. Loss is computed on the assistant reply tokens only;
the prompt is built with app.prompting.build_messages, so the adapter sees the
same system prompt and source rendering as the API.

Commands of the recorded runs (all with --grad-accum 2 --threads 4):

    python -m finetune.train_lora --epochs 1 --lr 3e-4                                         (run 1)
    python -m finetune.train_lora --epochs 2 --lr 2e-4 --oversample insufficient_evidence=2,conflict=2  (run 2)
    python -m finetune.train_lora --epochs 2 --lr 2e-4 --output-dir <run 3 folder>               (run 3)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.optim.lr_scheduler import LambdaLR

from finetune.common import (
    ADAPTER_DIR,
    DATA_DIR,
    DEFAULT_THREADS,
    MODEL_ID,
    MODEL_REVISION,
    environment,
    epoch_order,
    load_jsonl,
    load_model,
    parse_oversample,
    rel,
    sha256,
    to_messages,
)

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--train", type=Path, default=DATA_DIR / "train.jsonl")
    p.add_argument("--output-dir", type=Path, default=ADAPTER_DIR)
    # Defaults reproduce run 1. Run 2 passes its settings explicitly (recorded in run_info.json).
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-len", type=int, default=1024, help="examples longer than this are skipped, not cut")
    p.add_argument("--limit", type=int, default=0, help="use only the first N training rows (0 = all)")
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument(
        "--oversample",
        type=parse_oversample,
        default={},
        help="repeat rows of a category within each epoch, e.g. insufficient_evidence=3,conflict=3",
    )
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="save a resumable checkpoint every N optimizer steps (0 = never); rerun the same command to resume",
    )
    return p.parse_args(argv)


# Settings that must match for a checkpoint to be resumed; anything else would mix two different runs.
RESUME_KEYS = ("epochs", "lr", "max_len", "limit", "grad_accum", "oversample", "rank", "alpha", "dropout", "seed")


def checkpoint_signature(args: argparse.Namespace) -> dict:
    return {"train_sha256": sha256(args.train), **{k: getattr(args, k) for k in RESUME_KEYS}}


def save_checkpoint(path: Path, model, opt, sched, rng: random.Random, position: dict) -> None:
    """Everything needed to continue exactly: trainable weights, optimizer, schedule, data order and RNG state."""
    state = {
        **position,
        "trainable": {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad},
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "rng": rng.getstate(),
        "torch_rng": torch.get_rng_state(),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)  # atomic, so a kill during the save never leaves a half-written checkpoint


def load_checkpoint(path: Path, model, opt, sched, rng: random.Random, signature: dict) -> dict | None:
    if not path.exists():
        return None
    state = torch.load(path, weights_only=False)  # our own file, written by save_checkpoint
    if state["signature"] != signature:
        raise SystemExit(f"{path} was written with different settings or data; delete it to start this run fresh")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in state["trainable"]:
                param.copy_(state["trainable"][name])
    opt.load_state_dict(state["opt"])
    sched.load_state_dict(state["sched"])
    rng.setstate(state["rng"])
    torch.set_rng_state(state["torch_rng"])
    return state


def encode(tok, row: dict) -> tuple[list[int], int]:
    """Token ids for prompt + reply, and how many trailing ids are the reply (incl. end-of-turn)."""
    messages = to_messages(row)
    reply = {"role": "assistant", "content": json.dumps(row["target"])}
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template([*messages, reply], tokenize=False)
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tok(full, add_special_tokens=False)["input_ids"]
    if not full.startswith(prompt) or full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(f"{row['id']}: prompt tokens are not a prefix of the full chat")
    return full_ids, len(full_ids) - len(prompt_ids)


def build_examples(tok, rows: list[dict], max_len: int) -> tuple[list[dict], list[str], int]:
    examples, skipped, longest = [], [], 0
    for row in rows:
        ids, n_reply = encode(tok, row)
        longest = max(longest, len(ids))
        if len(ids) > max_len:
            skipped.append(row["id"])
            continue
        examples.append({"input_ids": torch.tensor([ids]), "n_reply": n_reply, "category": row["category"]})
    return examples, skipped, longest


def reply_loss(model, ex: dict) -> torch.Tensor:
    """Mean cross-entropy over the reply tokens only (prompt tokens are masked out).

    Logits are computed only for the last n_reply + 1 positions, which skips the
    152k-vocab output layer for the prompt; that layer dominates CPU time otherwise.
    """
    ids, n = ex["input_ids"], ex["n_reply"]
    logits = model(input_ids=ids, logits_to_keep=n + 1).logits[0, :-1]
    return F.cross_entropy(logits, ids[0, -n:])


def train(model, examples: list[dict], args: argparse.Namespace, ckpt: Path | None = None) -> dict:
    """Returns the loss log, the steps it resumed at, and the training seconds summed over every session."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    categories = [ex["category"] for ex in examples]
    per_epoch = sum(args.oversample.get(c, 1) for c in categories)
    total = math.ceil(per_epoch / args.grad_accum) * args.epochs
    sched = LambdaLR(opt, lambda s: max(0.0, 1 - s / total))  # linear decay to zero
    rng, log, resumed, before = random.Random(args.seed), [], [], 0.0
    signature = checkpoint_signature(args)
    state = load_checkpoint(ckpt, model, opt, sched, rng, signature) if ckpt else None
    first_epoch, first_start = 1, 0
    if state:
        first_epoch, first_start, log, before = state["epoch"], state["start"], state["log"], state["seconds"]
        resumed = [*state["resumed"], len(log)]
        print(f"resumed from {ckpt} at step {len(log)}/{total}", flush=True)
    session_t0 = time.perf_counter()
    model.train()
    for epoch in range(first_epoch, args.epochs + 1):
        resuming = state is not None and epoch == first_epoch
        order = state["order"] if resuming else epoch_order(categories, args.oversample, rng)
        for start in range(first_start if resuming else 0, len(order), args.grad_accum):
            batch, t0, step_loss = order[start : start + args.grad_accum], time.perf_counter(), 0.0
            for idx in batch:
                loss = reply_loss(model, examples[idx]) / len(batch)
                loss.backward()
                step_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            entry = {
                "step": len(log) + 1,
                "epoch": epoch,
                "loss": round(step_loss, 4),
                "seconds": round(time.perf_counter() - t0, 2),
            }
            log.append(entry)
            print(
                f"step {entry['step']}/{total} epoch {epoch} loss {entry['loss']:.4f} ({entry['seconds']}s)", flush=True
            )
            if ckpt and args.checkpoint_every and len(log) % args.checkpoint_every == 0 and len(log) < total:
                position = {
                    "signature": signature,
                    "epoch": epoch,
                    "start": start + args.grad_accum,
                    "order": order,
                    "log": log,
                    "resumed": resumed,
                    "seconds": before + time.perf_counter() - session_t0,
                }
                save_checkpoint(ckpt, model, opt, sched, rng, position)
    return {"log": log, "resumed_at_steps": resumed, "seconds": before + time.perf_counter() - session_t0}


def write_model_card(output_dir: Path) -> None:
    """Replace peft's placeholder model card with a pointer to the real documentation."""
    text = (
        f"# LoRA adapter for {MODEL_ID}\n\n"
        "Trained by `finetune/train_lora.py` on synthetic data only. Settings, versions and the loss "
        "log are in `../run_info.json`; results and commands are in `finetune/README.md`.\n"
    )
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)  # also fixes the LoRA A-matrix init below
    rows = load_jsonl(args.train)[: args.limit or None]
    tok, model = load_model()
    model.config.use_cache = False
    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    examples, skipped, longest = build_examples(tok, rows, args.max_len)
    print(
        f"{len(examples)} examples ({len(skipped)} skipped, longest {longest} tokens), {trainable:,} trainable params",
        flush=True,
    )
    ckpt = args.output_dir.parent / "checkpoint.pt"
    result = train(model, examples, args, ckpt if args.checkpoint_every else None)
    log, wall = result["log"], result["seconds"]
    model.save_pretrained(args.output_dir)
    write_model_card(args.output_dir)
    ckpt.unlink(missing_ok=True)  # the finished adapter replaces it
    info = {
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": "python -m finetune.train_lora " + " ".join(sys.argv[1:] if argv is None else argv),
        "base_model": MODEL_ID,
        "base_model_revision": MODEL_REVISION,
        "environment": {**environment(), "torch_threads": torch.get_num_threads()},
        "hyperparameters": {k: rel(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "lora_target_modules": TARGET_MODULES,
        "trainable_params": trainable,
        "dataset": {"path": rel(args.train), "sha256": sha256(args.train), "rows_used": len(rows)},
        "examples_trained": len(examples),
        "examples_per_epoch_after_oversampling": sum(args.oversample.get(e["category"], 1) for e in examples),
        "skipped_too_long": skipped,
        "longest_example_tokens": longest,
        "optimizer_steps": len(log),
        "final_loss": log[-1]["loss"] if log else None,
        "train_wall_seconds": round(wall, 1),
        "resumed_at_steps": result["resumed_at_steps"],  # empty when the run was never interrupted
        "loss_log": log,
    }
    run_info = args.output_dir.parent / "run_info.json"
    run_info.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"saved adapter to {args.output_dir} and {run_info} ({wall / 60:.1f} min)")


if __name__ == "__main__":
    main()
