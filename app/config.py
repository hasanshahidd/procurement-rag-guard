"""Runtime settings, read from environment variables and checked at startup."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_CHOICES = {"llm_backend": ("ollama", "transformers"), "retrieval_mode": ("hybrid", "bm25")}


@dataclass(frozen=True)
class Settings:
    data_dir: Path = ROOT / "data" / "seed"
    audit_path: Path = ROOT / "logs" / "audit.jsonl"
    llm_backend: str = "ollama"  # ollama | transformers
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:1.5b"
    hf_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    hf_revision: str = "7ae557604adf67be50417f59c2c2f167def9a775"  # same snapshot the LoRA runs used
    lora_adapter: str | None = None
    llm_timeout_s: float = 120.0
    min_retrieval_score: float = 1.0
    retrieval_mode: str = "hybrid"  # hybrid | bm25
    embed_model: str = "nomic-embed-text"
    min_dense_score: float = 0.6  # measured: on-topic questions 0.70 to 0.80, off-topic at most 0.51
    db_path: Path | None = None
    rate_limit_per_min: int = 60

    def __post_init__(self) -> None:
        # Fail at startup, not by silently falling back to another mode.
        for name, allowed in _CHOICES.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name.upper()} must be one of {', '.join(allowed)}, got '{getattr(self, name)}'")

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ.get
        return cls(
            data_dir=Path(env("DATA_DIR", str(cls.data_dir))),
            audit_path=Path(env("AUDIT_LOG", str(cls.audit_path))),
            llm_backend=env("LLM_BACKEND", cls.llm_backend).lower(),
            ollama_url=env("OLLAMA_URL", cls.ollama_url),
            ollama_model=env("OLLAMA_MODEL", cls.ollama_model),
            hf_model=env("HF_MODEL", cls.hf_model),
            hf_revision=env("HF_REVISION", cls.hf_revision),
            lora_adapter=env("LORA_ADAPTER") or None,
            llm_timeout_s=float(env("LLM_TIMEOUT_S", cls.llm_timeout_s)),
            min_retrieval_score=float(env("MIN_RETRIEVAL_SCORE", cls.min_retrieval_score)),
            retrieval_mode=env("RETRIEVAL_MODE", cls.retrieval_mode).lower(),
            embed_model=env("EMBED_MODEL", cls.embed_model),
            min_dense_score=float(env("MIN_DENSE_SCORE", cls.min_dense_score)),
            db_path=Path(env("DB_PATH")) if env("DB_PATH") else None,
            rate_limit_per_min=int(env("RATE_LIMIT_PER_MIN", cls.rate_limit_per_min)),
        )
