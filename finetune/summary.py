"""Write results/summary.md from the per-column metrics files written by evaluate.py."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from app.prompting import STATUSES
from finetune.build_dataset import CATEGORIES
from finetune.common import MODEL_ID, MODEL_REVISION

INVALID = "invalid"  # predicted status when parse_model_output rejects the reply
# (row title, key path into a metrics file). Every metric here is "higher is better".
METRICS: list[tuple[str, tuple[str, ...]]] = [
    ("JSON valid rate (parse_model_output accepts)", ("json_valid_rate",)),
    ("Status accuracy", ("status_accuracy",)),
    ("Status macro F1 (3 labels)", ("status_macro_f1",)),
    *(
        (f"Status {k}: {s}", ("status_per_class", s, k))
        for s in ("insufficient_evidence", "conflict", "answered")
        for k in ("precision", "recall")
    ),
    ("Citation exact match (set equality)", ("citation_exact_match",)),
    ("Citation precision (micro)", ("citation_precision",)),
    ("Citation recall (micro)", ("citation_recall",)),
    ("Citation F1 (micro)", ("citation_f1",)),
    ("Injection resisted rate", ("injection_resisted_rate",)),
    ("Injection answer grounded rate", ("injection_answer_grounded_rate",)),
    *((f"Status accuracy: {c}", ("status_accuracy_by_category", c)) for c in CATEGORIES),
]
SELECTION_KEYS = ("status_macro_f1", "status_accuracy", "citation_f1")
SELECTION_RULE = (
    "Selection rule (decided before run 3 was evaluated): the standard adapter at "
    "`finetune/outputs/qwen2.5-0.5b-procurement-lora/` is the trained run with the highest status macro F1 "
    "on this set; ties are broken by status accuracy, then citation F1."
)
VALIDATION_CAVEAT = (
    "Because this set was used to choose between runs, it acts as a validation set and the chosen run's "
    "numbers here are optimistic. An unbiased estimate needs a split that played no part in the choice: "
    "see `finetune/data/test.jsonl` and `finetune/results/test_split/summary.md`."
)


def with_macro_f1(col: dict) -> dict:
    """Mean per-label F1 from the stored precision and recall; an undefined precision counts as 0."""
    f1s = []
    for v in col["status_per_class"].values():
        p, r = v["precision"] or 0.0, v["recall"] or 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return {**col, "status_macro_f1": round(sum(f1s) / len(f1s), 3)}


def select_run(cols: list[dict]) -> dict:
    """The trained column that wins under SELECTION_RULE."""
    trained = [c for c in cols if c.get("training")]
    return max(trained, key=lambda c: tuple(c[k] for k in SELECTION_KEYS))


def selection(cols: list[dict]) -> list[str]:
    trained = [c for c in cols if c.get("training")]
    if len(trained) < 2:
        return []
    ranked = sorted(trained, key=lambda c: tuple(c[k] for k in SELECTION_KEYS), reverse=True)
    lines = ["## Selection", "", SELECTION_RULE, "", "| Run | Status macro F1 | Status accuracy | Citation F1 |"]
    lines.append("|---|---|---|---|")
    lines += [f"| {c['label']} | " + " | ".join(fmt(c[k]) for k in SELECTION_KEYS) + " |" for c in ranked]
    return [*lines, "", f"Selected: {select_run(cols)['label']}.", "", VALIDATION_CAVEAT, ""]


def pick(metrics: dict, path: tuple[str, ...]) -> float | None:
    for key in path:
        metrics = metrics[key]
    return metrics


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def column_title(col: dict) -> str:
    return col["label"] + (" (rescored)" if col["generation"].get("rescored_from") else "")


def decoding(g: dict) -> str:
    if g.get("backend") == "ollama":
        d = g.get("ollama_details", {})
        return (
            f"Ollama {g['ollama_version']} serving `{g['ollama_model']}` ({d.get('parameter_size')}, "
            f"{d.get('quantization_level')}), /api/chat with format=json and options {json.dumps(g['options'])} "
            "(the app/llm.py options plus num_ctx)"
        )
    return (
        f"transformers, `{MODEL_ID}` at revision `{MODEL_REVISION[:12]}`, greedy (do_sample=False, same call as "
        f"app/llm.py, so the checkpoint's repetition_penalty={g['repetition_penalty']} applies), "
        f"max_new_tokens={g['max_new_tokens']}, fp32 on CPU, {g['torch_threads']} torch threads"
    )


def header(cols: list[dict]) -> list[str]:
    gens = [c["generation"] for c in cols]
    if len({g["heldout_sha256"] for g in gens}) != 1:
        raise ValueError("columns were scored on different files; they are not comparable")
    env, g = gens[-1]["environment"], gens[-1]
    lines = [
        f"# LoRA demo: results on `{g['heldout']}`",
        "",
        f"Summary written {date.today().isoformat()}. Every number comes from the metrics files in this folder.",
        f"Hardware: {env['cpu']}, {env['logical_cpus']} logical CPUs, {env['gpu']}.",
        f"Versions: Python {env['python']}, torch {env['torch']}, transformers {env['transformers']}, "
        f"peft {env['peft']}, tokenizers {env['tokenizers']} ({env['platform']}).",
        f"Evaluation set: `{g['heldout']}` ({cols[-1]['n']} rows, sha256 `{g['heldout_sha256'][:16]}...`), "
        "the same file for every column.",
        f"Base model revision (transformers columns): `{MODEL_REVISION}`, pinned in finetune/common.py. Columns "
        "written before the pin loaded the same snapshot, the only one in the local Hugging Face cache.",
    ]
    by_decoding: dict[str, list[str]] = {}
    for c in cols:
        by_decoding.setdefault(decoding(c["generation"]), []).append(c["label"])
    lines += [f"Decoding for {', '.join(labels)}: {desc}." for desc, labels in by_decoding.items()]
    if len({json.dumps(x["environment"], sort_keys=True) for x in gens}) > 1:
        lines.append("Library versions or hardware differ between columns; see each metrics file.")
    rescored = [c for c in cols if c["generation"].get("rescored_from")]
    if rescored:
        names = ", ".join(c["label"] for c in rescored)
        lines.append(
            f"Rescored columns ({names}): the current scorer was run on replies saved by the earlier "
            "generation run (no new generation). The held-out file and decoding settings are unchanged."
        )
    return [*lines, ""]


def main_table(cols: list[dict]) -> list[str]:
    n_inj = cols[-1]["injection_n"]
    lines = ["| Metric | " + " | ".join(column_title(c) for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for title, path in METRICS:
        title = f"{title} (n={n_inj})" if path[0].startswith("injection") else title
        lines.append(f"| {title} | " + " | ".join(fmt(pick(c, path)) for c in cols) + " |")
    lines.append("| Eval wall time (s) | " + " | ".join(str(c["generation"]["eval_wall_seconds"]) for c in cols) + " |")
    resisted = {c["injection_resisted_rate"] for c in cols}
    if len(resisted) == 1:
        lines += [
            "",
            f"Injection resisted is {fmt(resisted.pop())} for every column, the baseline included, so it does "
            "not separate the models on this set.",
        ]
    return [*lines, ""]


def confusion_tables(cols: list[dict]) -> list[str]:
    lines = ["## Status confusion matrices (rows: gold status, columns: predicted status)", ""]
    for c in cols:
        lines += [f"{column_title(c)}:", "", "| gold \\ predicted | " + " | ".join((*STATUSES, INVALID)) + " |"]
        lines.append("|---" * (len(STATUSES) + 2) + "|")
        for gold in STATUSES:
            row = c["status_confusion"][gold]
            lines.append(f"| {gold} | " + " | ".join(str(row[p]) for p in (*STATUSES, INVALID)) + " |")
        lines.append("")
    return lines


def training_table(cols: list[dict]) -> list[str]:
    trained = [c for c in cols if c.get("training")]
    if not trained:
        return []
    lines = [
        "## Training",
        "",
        "| Column | Epochs | lr | Grad accum | Oversample | Examples per epoch | Optimizer steps "
        "| Final step loss | Train wall time |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in trained:
        t = c["training"]
        over = ", ".join(f"{k} x{v}" for k, v in t["oversample"].items()) or "none"
        lines.append(
            f"| {c['label']} | {t['epochs']} | {t['lr']} | {t['grad_accum']} | {over} | {t['examples_per_epoch']} "
            f"| {t['optimizer_steps']} | {t['final_loss']} | {t['train_wall_seconds'] / 60:.1f} min |"
        )
    return [*lines, ""]


def commands(cols: list[dict], summary_cmd: str) -> list[str]:
    cmds = ["python -m finetune.build_dataset --seed 13"]
    cmds += [c["training"]["command"] for c in cols if c.get("training")]
    cmds += [c["generation"]["command"] for c in cols]
    cmds += [c["scored"]["command"] for c in cols if c["generation"].get("rescored_from")]
    cmds.append(summary_cmd)
    unique = list(dict.fromkeys(cmds))
    note = (
        "Commands as recorded, run from the repo root. Rescored columns list the command that generated "
        "their replies (an earlier evaluate.py that ran the base model and the adapter in one call) and "
        "the rescoring command."
    )
    return ["## Commands", "", note, "", "```", *unique, "```", ""]


def direction(before: float | None, after: float | None) -> str:
    if before is None or after is None:
        return "Not comparable (n/a)"
    if after == before:
        return "Unchanged"
    return "Higher" if after > before else "Lower"


def comparison(cols: list[dict]) -> list[str]:
    """Plain lists of which metrics are higher, lower or equal for the selected run against every other column."""
    last = select_run(cols) if any(c.get("training") for c in cols) else cols[-1]
    lines = ["## What improved and what did not", ""]
    for ref in reversed([c for c in cols if c is not last]):
        groups: dict[str, list[str]] = {"Higher": [], "Lower": [], "Unchanged": [], "Not comparable (n/a)": []}
        for title, path in METRICS:
            a, b = pick(ref, path), pick(last, path)
            groups[direction(a, b)].append(f"{title} ({fmt(a)} -> {fmt(b)})")
        lines.append(f"{last['label']} compared with {ref['label']}:")
        lines += [f"- {k}: {'; '.join(v)}" for k, v in groups.items() if v]
        lines.append("")
    return lines


def write_summary(out_dir: Path, names: list[str], summary_cmd: str, note: str = "") -> Path:
    cols = [with_macro_f1(json.loads((out_dir / f"{n}_metrics.json").read_text(encoding="utf-8"))) for n in names]
    n, n_inj = cols[-1]["n"], max(cols[-1]["injection_n"], 1)
    n_conflict = max(cols[-1]["status_per_class"]["conflict"]["gold"], 1)
    caveat = (
        "The set is small and synthetic (invented companies, SKUs and injection wording that never "
        f"appear in train), so these numbers are indicative only. One row moves an overall rate by {1 / n:.3f}, "
        f"the injection rates by {1 / n_inj:.3f} and the conflict recall by {1 / n_conflict:.3f} "
        f"({n_conflict} gold rows). Each column is a single run on a shared CPU. Status macro F1 is computed "
        "here from the per-label precision and recall in each metrics file, counting an undefined precision "
        "(label never predicted) as 0."
    )
    lines = [
        *header(cols),
        *([note, ""] if note else []),
        *main_table(cols),
        *selection(cols),
        *confusion_tables(cols),
        *training_table(cols),
        *comparison(cols),
        *commands(cols, summary_cmd),
        caveat,
        "",
    ]
    path = out_dir / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
