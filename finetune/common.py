"""Paths and helpers shared by train_lora and evaluate. Heavy imports stay inside functions."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
from importlib.metadata import version
from pathlib import Path

from app.prompting import ContextSource, build_messages
from finetune.build_dataset import CATEGORIES

ROOT = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
# Hugging Face commit of MODEL_ID. Runs 1 to 3 loaded this snapshot (the only one in the local cache).
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DATA_DIR = ROOT / "data"
ADAPTER_DIR = ROOT / "outputs" / "qwen2.5-0.5b-procurement-lora"
RESULTS_DIR = ROOT / "results"
# shortcut: assumes 2 hardware threads per core (true for the i5-1145G7), pass --threads otherwise
DEFAULT_THREADS = max(1, (os.cpu_count() or 2) // 2)


def rel(path: Path) -> str:
    """Repo-relative path for run_info.json and summaries. Raw console logs keep what the libraries printed."""
    try:
        return path.resolve().relative_to(ROOT.parent).as_posix()
    except ValueError:
        return str(path)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def to_messages(row: dict) -> list[dict[str, str]]:
    """The exact chat the API sends for this question and these sources."""
    return build_messages(row["question"], [ContextSource(**s) for s in row["sources"]])


def parse_oversample(spec: str) -> dict[str, int]:
    """'insufficient_evidence=3,conflict=3' -> {category: copies per epoch}."""
    out: dict[str, int] = {}
    for part in filter(None, spec.split(",")):
        name, _, copies = part.partition("=")
        if name not in CATEGORIES or not copies.isdigit() or int(copies) < 1:
            raise ValueError(f"bad --oversample entry '{part}', expected <category>=<copies>")
        out[name] = int(copies)
    return out


def epoch_order(categories: list[str], oversample: dict[str, int], rng: random.Random) -> list[int]:
    """Example indices for one epoch, each repeated per its category's copies, then shuffled."""
    idx = [i for i, cat in enumerate(categories) for _ in range(oversample.get(cat, 1))]
    return rng.sample(idx, len(idx))  # with no repeats this draws exactly what run 1 drew


def cpu_name() -> str:
    try:
        import winreg

        key = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as k:
            return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
    except (ImportError, OSError):
        return platform.processor() or "unknown"


def environment() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": cpu_name(),
        "logical_cpus": os.cpu_count(),
        "gpu": "none (CPU only)",
        **{lib: version(lib) for lib in ("torch", "transformers", "peft", "tokenizers")},
    }


def load_model(model_id: str = MODEL_ID) -> tuple:
    """Tokenizer and fp32 model on CPU (bf16 is slow on CPUs without native bf16 support)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=MODEL_REVISION, dtype=torch.float32)
    return tok, model
