"""Hybrid retrieval (BM25 plus embeddings) and SQLite persistence."""

from __future__ import annotations

import json
import re

import httpx
import pytest

from app.embeddings import EmbeddingUnavailableError, OllamaEmbedder, cosine
from app.retrieval import hybrid_search
from conftest import HARBOR_BUYER, REDLINE_BUYER, FakeLLM, ask, reply

PARAPHRASE = "When will the computers arrive?"  # shares no keyword with any record


class FakeEmbedder:
    """Counts a few concepts, with synonyms, so paraphrases land near the right records."""

    name = "fake-embed"
    DIMS = ("laptop", "ship", "clinic", "price", "approval")
    SYNONYMS = {"computers": "laptop", "laptops": "laptop", "arrive": "ship", "delivery": "ship", "prices": "price"}

    def __init__(self, fail: bool = False):
        self.fail = fail

    def _vec(self, text: str) -> list[float]:
        words = [self.SYNONYMS.get(w, w) for w in re.findall(r"[a-z]+", text.lower())]
        return [float(words.count(d)) for d in self.DIMS]

    def embed_query(self, text: str) -> list[float]:
        if self.fail:
            raise EmbeddingUnavailableError("model not pulled")
        return self._vec(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def health(self) -> dict:
        return {"reachable": not self.fail, "detail": "fake"}


def test_hybrid_finds_a_paraphrase_that_keywords_miss(make_client):
    llm = FakeLLM(reply("answered", "It can ship in 10 business days.", ["TCK-H-102"]))
    bm25_only = ask(make_client(llm), PARAPHRASE, HARBOR_BUYER).json()
    assert bm25_only["reason"] == "no_relevant_sources"

    body = ask(make_client(llm, embedder=FakeEmbedder(), retrieval_mode="hybrid"), PARAPHRASE, HARBOR_BUYER).json()
    hit = next(r for r in body["retrieved"] if r["source_id"] == "TCK-H-102")
    assert hit["bm25"] is None and hit["dense"] >= 0.6
    assert body["agents"][0]["summary"].startswith("hybrid BM25 + fake-embed")


def test_hybrid_still_abstains_on_off_topic_questions(make_client):
    client = make_client(FakeLLM(reply()), embedder=FakeEmbedder(), retrieval_mode="hybrid")
    assert ask(client, "Who won the football final last night?", HARBOR_BUYER).json()["reason"] == "no_relevant_sources"


def test_hybrid_only_ranks_visible_documents(store):
    harbor = store.authenticate("demo-harbor-buyer")
    hits, _ = hybrid_search(store.visible_docs(harbor), "Redline price laptops", 20, FakeEmbedder(), 1.0, 0.1)
    assert hits and all(h.doc.tenant_id in ("harbor", "shared") for h in hits)


def test_hybrid_falls_back_to_bm25_when_embeddings_are_down(make_client):
    llm = FakeLLM(reply("answered", "PR-1042 is pending approval.", ["TCK-H-101"]))
    client = make_client(llm, embedder=FakeEmbedder(fail=True), retrieval_mode="hybrid")
    body = ask(client, "What is the status of PR-1042?", HARBOR_BUYER).json()
    assert body["status"] == "answered"
    assert "BM25 only, embeddings unavailable" in body["agents"][0]["summary"]
    health = client.get("/health").json()
    assert health["retrieval"]["mode"] == "hybrid" and health["retrieval"]["reachable"] is False


def test_cosine_handles_zero_vectors():
    assert cosine([0.0, 0.0], [1.0, 2.0]) == 0.0
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_ollama_embedder_caches_documents_and_maps_errors():
    calls: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload["input"])
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]] * len(payload["input"])})

    emb = OllamaEmbedder("http://ollama.test", "nomic-embed-text")
    emb._client = httpx.Client(base_url="http://ollama.test", transport=httpx.MockTransport(handler))
    emb.embed_documents(["a", "b"])
    emb.embed_documents(["a", "b", "c"])
    assert calls == [["search_document: a", "search_document: b"], ["search_document: c"]]
    assert emb.embed_query("q") == [1.0, 0.0]

    emb._client = httpx.Client(
        base_url="http://ollama.test", transport=httpx.MockTransport(lambda r: httpx.Response(500))
    )
    with pytest.raises(EmbeddingUnavailableError):
        emb.embed_query("q")


@pytest.mark.parametrize(
    "body", [b"<html>proxy</html>", b"[]", b'{"embeddings": null}', b'{"embeddings": [["x"]]}', b'{"embeddings": [[]]}']
)
def test_ollama_embedder_rejects_malformed_replies(body):
    emb = OllamaEmbedder("http://ollama.test", "nomic-embed-text")
    emb._client = httpx.Client(
        base_url="http://ollama.test", transport=httpx.MockTransport(lambda r: httpx.Response(200, content=body))
    )
    with pytest.raises(EmbeddingUnavailableError):
        emb.embed_query("q")


class _MismatchedEmbedder(FakeEmbedder):
    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]  # different size from the document vectors


def test_hybrid_falls_back_to_bm25_on_mismatched_vector_sizes(make_client):
    llm = FakeLLM(reply("answered", "PR-1042 is pending approval.", ["TCK-H-101"]))
    client = make_client(llm, embedder=_MismatchedEmbedder(), retrieval_mode="hybrid")
    body = ask(client, "What is the status of PR-1042?", HARBOR_BUYER).json()
    assert body["status"] == "answered"
    assert "embedding sizes differ" in body["agents"][0]["summary"]


def test_ingested_records_survive_a_restart(make_client, tmp_path):
    db = tmp_path / "store.sqlite3"
    first = make_client(db_path=db)
    payload = {"id": "TCK-H-500", "title": "Desk lamps", "text": "Ten desk lamps for the Eastgate clinic.", "facts": {}}
    assert first.post("/tickets", json=payload, headers=HARBOR_BUYER).status_code == 201

    second = make_client(db_path=db)
    assert second.get("/sources/TCK-H-500", headers=HARBOR_BUYER).json()["text"] == payload["text"]
    assert second.get("/sources/TCK-H-500", headers=REDLINE_BUYER).status_code == 404
    assert second.post("/tickets", json=payload, headers=HARBOR_BUYER).status_code == 409
    assert second.get("/health").json()["documents"] == 15


def test_tool_outputs_stay_verifiable_after_a_restart(make_client, tmp_path):
    db = tmp_path / "store.sqlite3"
    llm = FakeLLM(reply("answered", "Bluegate quoted 1,175 AUD per unit.", ["TOOL-Q-5531"]))
    cite = ask(make_client(llm, db_path=db), "What did Bluegate quote for LAP-14-STD?", HARBOR_BUYER).json()
    [citation] = cite["citations"]
    source = make_client(db_path=db).get("/sources/TOOL-Q-5531", headers=HARBOR_BUYER).json()
    assert source["text"][citation["start"] : citation["end"]] == citation["excerpt"]
    assert source["content_sha256"] == citation["content_sha256"]
