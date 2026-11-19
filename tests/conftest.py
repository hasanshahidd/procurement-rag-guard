from __future__ import annotations

import json
from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.store import Store

HARBOR_BUYER = {"X-API-Key": "demo-harbor-buyer"}
HARBOR_REQUESTER = {"X-API-Key": "demo-harbor-requester"}
REDLINE_BUYER = {"X-API-Key": "demo-redline-buyer"}

INJECTION_QUESTION = "What is the status of PR-1042 and what did Bluegate quote for LAP-14-STD?"


def reply(status: str = "answered", answer: str = "", citations: list[str] | None = None, **extra: object) -> str:
    return json.dumps({"status": status, "answer": answer, "citations": citations or [], **extra})


class FakeLLM:
    """Test double for the model. Records every prompt it is shown."""

    def __init__(
        self,
        response: str | Callable[[list[dict]], str] = "",
        *,
        raises: Exception | None = None,
        reachable: bool = True,
    ):
        self.name = "fake-llm"
        self.response, self.raises, self.reachable = response, raises, reachable
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        if self.raises:
            raise self.raises
        return self.response(messages) if callable(self.response) else self.response

    def health(self) -> dict:
        return {"reachable": self.reachable, "detail": "fake"}

    @property
    def last_prompt(self) -> str:
        return "\n".join(m["content"] for m in self.calls[-1])


@pytest.fixture
def settings(tmp_path) -> Settings:
    # BM25 by default keeps the suite deterministic and offline; hybrid has its own tests with a fake embedder.
    return Settings(audit_path=tmp_path / "audit.jsonl", rate_limit_per_min=1000, retrieval_mode="bm25")


@pytest.fixture
def store(settings) -> Store:
    return Store.from_dir(settings.data_dir)


@pytest.fixture
def make_client(settings) -> Callable[..., TestClient]:
    def _make(llm: FakeLLM | None = None, embedder: object = None, **overrides: object) -> TestClient:
        cfg = Settings(**{**settings.__dict__, **overrides})
        app = create_app(cfg, llm=llm or FakeLLM(reply()), embedder=embedder)
        return TestClient(app, raise_server_exceptions=False)

    return _make


def ask(client: TestClient, question: str, headers: dict, **body: object):
    return client.post("/ask", json={"question": question, **body}, headers=headers)
