"""Local open-weight model backends.

ollama        Talks to a local Ollama server (default model qwen2.5:1.5b).
transformers  Loads a Hugging Face checkpoint in-process on CPU or GPU, with an
              optional LoRA adapter from finetune/. No server needed.

Both return the raw text of the model reply. Nothing here decides what the
reply is allowed to do; that happens in pipeline.verify.
"""

from __future__ import annotations

import threading
from typing import Protocol

import httpx

from app.config import Settings


class LLMUnavailableError(RuntimeError):
    """The configured backend cannot serve a request right now."""


class LLM(Protocol):
    name: str

    def chat(self, messages: list[dict[str, str]]) -> str: ...

    def health(self) -> dict: ...


class OllamaLLM:
    def __init__(self, base_url: str, model: str, timeout_s: float):
        self.name = f"ollama:{model}"
        self._model = model
        self._client = httpx.Client(base_url=base_url, timeout=timeout_s)

    def chat(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "seed": 7, "num_predict": 400},
        }
        try:
            resp = self._client.post("/api/chat", json=payload)
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(f"Ollama not reachable: {exc.__class__.__name__}") from exc
        if resp.status_code == 404:
            raise LLMUnavailableError(f"Model '{self._model}' is not pulled. Run: ollama pull {self._model}")
        if resp.status_code != 200:
            raise LLMUnavailableError(f"Ollama returned HTTP {resp.status_code}")
        return resp.json().get("message", {}).get("content", "")

    def health(self) -> dict:
        try:
            tags = self._client.get("/api/tags", timeout=3).json().get("models", [])
        except (httpx.HTTPError, ValueError):
            return {"reachable": False, "detail": "Ollama server not reachable"}
        names = {t.get("name", "") for t in tags}
        pulled = self._model in names or f"{self._model}:latest" in names
        return {"reachable": pulled, "detail": "ok" if pulled else f"model {self._model} not pulled"}


class TransformersLLM:
    def __init__(self, model_id: str, adapter: str | None = None, max_new_tokens: int = 320):
        self.name = f"transformers:{model_id}" + (f"+lora:{adapter}" if adapter else "")
        self._model_id, self._adapter, self._max_new = model_id, adapter, max_new_tokens
        self._model = self._tok = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise LLMUnavailableError("Install requirements-ml.txt to use the transformers backend") from exc
        self._tok = AutoTokenizer.from_pretrained(self._model_id)
        # fp32 matches the fine-tuning evaluation and is faster than bf16 on most CPUs.
        model = AutoModelForCausalLM.from_pretrained(self._model_id, dtype=torch.float32)
        if self._adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self._adapter)
        self._model = model.eval()

    def chat(self, messages: list[dict[str, str]]) -> str:
        import torch

        with self._lock:  # one generation at a time; the model is not thread safe
            if self._model is None:
                self._load()
            enc = self._tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
            )
            with torch.no_grad():
                out = self._model.generate(
                    **enc, max_new_tokens=self._max_new, do_sample=False, pad_token_id=self._tok.eos_token_id
                )
        return self._tok.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True)

    def health(self) -> dict:
        return {"reachable": True, "detail": "loaded" if self._model is not None else "loads on first request"}


def build_llm(settings: Settings) -> LLM:
    if settings.llm_backend == "transformers":
        return TransformersLLM(settings.hf_model, settings.lora_adapter)
    if settings.llm_backend == "ollama":
        return OllamaLLM(settings.ollama_url, settings.ollama_model, settings.llm_timeout_s)
    raise ValueError(f"Unknown LLM_BACKEND '{settings.llm_backend}'")
