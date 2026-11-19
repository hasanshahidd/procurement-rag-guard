"""Dense embeddings for hybrid retrieval, from a local open-weight model via Ollama.

Default model: nomic-embed-text (Apache 2.0, 768 dimensions). Vectors are
cached by a hash of the embedded text, so each record is embedded once. If the
embedding model is unavailable the retriever falls back to BM25 and says so in
the agent trace instead of failing the request.
"""

from __future__ import annotations

import hashlib
import math
import threading
from typing import Protocol

import httpx


class EmbeddingUnavailableError(RuntimeError):
    pass


class Embedder(Protocol):
    name: str

    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def health(self) -> dict: ...


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise EmbeddingUnavailableError(f"embedding sizes differ ({len(a)} vs {len(b)})")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class OllamaEmbedder:
    # nomic-embed-text was trained with these task prefixes.
    QUERY_PREFIX, DOC_PREFIX = "search_query: ", "search_document: "

    def __init__(self, base_url: str, model: str, timeout_s: float = 30.0):
        self.name = f"ollama:{model}"
        self._model = model
        self._client = httpx.Client(base_url=base_url, timeout=timeout_s)
        self._cache: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        try:
            resp = self._client.post("/api/embed", json={"model": self._model, "input": inputs})
        except httpx.HTTPError as exc:
            raise EmbeddingUnavailableError(f"Ollama not reachable: {exc.__class__.__name__}") from exc
        if resp.status_code != 200:
            raise EmbeddingUnavailableError(f"embedding model returned HTTP {resp.status_code}")
        try:
            vectors = resp.json()["embeddings"]
        except (ValueError, KeyError, TypeError) as exc:
            raise EmbeddingUnavailableError("embedding model returned a body that is not an embedding reply") from exc
        if not isinstance(vectors, list) or len(vectors) != len(inputs):
            raise EmbeddingUnavailableError("embedding count does not match input count")
        if not all(isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v) for v in vectors):
            raise EmbeddingUnavailableError("embedding vectors are malformed")
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self.QUERY_PREFIX + text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
        with self._lock:
            missing = {k: t for k, t in zip(keys, texts, strict=True) if k not in self._cache}
        if missing:
            vectors = self._embed([self.DOC_PREFIX + t for t in missing.values()])
            with self._lock:
                self._cache.update(zip(missing.keys(), vectors, strict=True))
        with self._lock:
            return [self._cache[k] for k in keys]

    def health(self) -> dict:
        try:
            names = {t.get("name", "") for t in self._client.get("/api/tags", timeout=3).json().get("models", [])}
        except (httpx.HTTPError, ValueError):
            return {"reachable": False, "detail": "Ollama server not reachable"}
        pulled = self._model in names or f"{self._model}:latest" in names
        return {"reachable": pulled, "detail": "ok" if pulled else f"run: ollama pull {self._model}"}
