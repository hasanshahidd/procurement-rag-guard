"""Runtime settings, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    data_dir: Path = ROOT / "data" / "seed"
    audit_path: Path = ROOT / "logs" / "audit.jsonl"
    llm_backend: str = "ollama"  # ollama | transformers
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:1.5b"
    hf_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    lora_adapter: str | None = None
    llm_timeout_s: float = 120.0
    min_retrieval_score: float = 1.0
    rate_limit_per_min: int = 60

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
            lora_adapter=env("LORA_ADAPTER") or None,
            llm_timeout_s=float(env("LLM_TIMEOUT_S", cls.llm_timeout_s)),
            min_retrieval_score=float(env("MIN_RETRIEVAL_SCORE", cls.min_retrieval_score)),
            rate_limit_per_min=int(env("RATE_LIMIT_PER_MIN", cls.rate_limit_per_min)),
        )
