"""Failure handling: bad model output, model outages, unexpected errors, audit trail."""

from __future__ import annotations

import hashlib
import json

import httpx
import pytest

from app.config import Settings
from app.llm import LLMUnavailableError, OllamaLLM, build_llm
from conftest import HARBOR_BUYER, FakeLLM, ask, reply

QUESTION = "What is the status of PR-1042?"


@pytest.mark.parametrize(
    "raw",
    [
        "Sure! PR-1042 is pending.",
        '{"status": "maybe", "answer": "x", "citations": []}',
        '{"status": "answered", "answer": 42, "citations": []}',
        '{"status": "answered", "answer": "x", "citations": "TCK-H-101"}',
        "[1, 2, 3]",
        "",
    ],
)
def test_contract_violations_abstain(make_client, raw):
    body = ask(make_client(FakeLLM(raw)), QUESTION, HARBOR_BUYER).json()
    assert body["status"] == "abstained"
    assert body["reason"] == "model_output_unverifiable"
    assert body["citations"] == []


def test_json_wrapped_in_prose_and_fences_is_accepted(make_client):
    raw = "Here you go:\n```json\n" + reply("answered", "Pending approval.", ["TCK-H-101"]) + "\n```"
    assert ask(make_client(FakeLLM(raw)), QUESTION, HARBOR_BUYER).json()["status"] == "answered"


def test_model_outage_is_a_503(make_client):
    llm = FakeLLM(raises=LLMUnavailableError("Ollama not reachable: ConnectError"))
    resp = ask(make_client(llm), QUESTION, HARBOR_BUYER)
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "llm_unavailable"


def test_unexpected_error_is_a_500_without_internals(make_client):
    resp = ask(make_client(FakeLLM(raises=RuntimeError("secret stack detail"))), QUESTION, HARBOR_BUYER)
    assert resp.status_code == 500
    assert resp.json()["error"]["code"] == "internal_error"
    assert "secret stack detail" not in resp.text


def test_ollama_backend_maps_transport_errors():
    llm = OllamaLLM("http://127.0.0.1:9", "qwen2.5:1.5b", timeout_s=2)
    with pytest.raises(LLMUnavailableError):
        llm.chat([{"role": "user", "content": "hi"}])
    assert llm.health()["reachable"] is False


@pytest.mark.parametrize(("status", "fragment"), [(404, "ollama pull"), (500, "HTTP 500")])
def test_ollama_backend_maps_http_errors(status, fragment):
    llm = OllamaLLM("http://ollama.test", "qwen2.5:1.5b", timeout_s=2)
    llm._client = httpx.Client(
        base_url="http://ollama.test", transport=httpx.MockTransport(lambda r: httpx.Response(status))
    )
    with pytest.raises(LLMUnavailableError, match=fragment):
        llm.chat([{"role": "user", "content": "hi"}])


def test_ollama_backend_sends_json_mode_and_returns_content():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "{}"}})

    llm = OllamaLLM("http://ollama.test", "qwen2.5:1.5b", timeout_s=2)
    llm._client = httpx.Client(base_url="http://ollama.test", transport=httpx.MockTransport(handler))
    assert llm.chat([{"role": "user", "content": "hi"}]) == "{}"
    assert seen["format"] == "json" and seen["options"]["temperature"] == 0 and seen["stream"] is False
    assert seen["options"]["num_ctx"] == 8192


@pytest.mark.parametrize("bad", [{"llm_backend": "openai"}, {"retrieval_mode": "vector"}])
def test_invalid_settings_fail_at_startup(bad):
    with pytest.raises(ValueError):
        Settings(**bad)


def test_failed_logins_are_rate_limited(make_client):
    client = make_client(rate_limit_per_min=2)
    codes = [ask(client, "status of PR-1042?", {"X-API-Key": f"guess-{i}"}).status_code for i in range(3)]
    assert codes == [401, 401, 429]


def test_transformers_backend_is_lazy_and_names_the_adapter(monkeypatch):
    import importlib.util

    llm = build_llm(Settings(llm_backend="transformers", lora_adapter="finetune/outputs/adapter"))
    assert llm.name == "transformers:Qwen/Qwen2.5-0.5B-Instruct+lora:finetune/outputs/adapter"
    # Pin package discovery so the result does not depend on whether requirements-ml.txt is installed.
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    assert llm.health() == {"reachable": True, "detail": "loads on first request"}
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert llm.health() == {"reachable": False, "detail": "missing packages: torch, transformers, peft"}


def test_every_answer_is_audited_without_credentials(make_client, settings):
    raw = reply("answered", "PR-1042 is pending approval.", ["TCK-H-101"])
    body = ask(make_client(FakeLLM(raw)), QUESTION, HARBOR_BUYER).json()
    lines = settings.audit_path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[-1])
    assert record["event"] == "ask" and record["trace_id"] == body["trace_id"]
    assert record["tenant_id"] == "harbor" and record["status"] == "answered"
    assert record["citations"][0]["sha256"] == body["citations"][0]["content_sha256"]
    assert record["model_output_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert "demo-harbor-buyer" not in lines[-1]


def test_ingest_is_audited(make_client, settings):
    client = make_client()
    created = client.post("/tickets", json={"title": "Chairs", "text": "Twelve chairs."}, headers=HARBOR_BUYER).json()
    record = json.loads(settings.audit_path.read_text(encoding="utf-8").splitlines()[-1])
    assert record["event"] == "ingest" and record["source_id"] == created["id"]
    assert record["content_sha256"] == created["content_sha256"] and record["role"] == "buyer"


@pytest.mark.parametrize(
    "body",
    [
        b"<html>proxy error</html>",
        b"[1, 2]",
        b'{"done": true}',
        b'{"message": {"content": null}}',
        b'{"message": "text"}',
    ],
)
def test_ollama_malformed_reply_is_unavailable_not_a_crash(body):
    llm = OllamaLLM("http://ollama.test", "qwen2.5:1.5b", timeout_s=2)
    transport = httpx.MockTransport(lambda r: httpx.Response(200, content=body))
    llm._client = httpx.Client(base_url="http://ollama.test", transport=transport)
    with pytest.raises(LLMUnavailableError, match="chat reply"):
        llm.chat([{"role": "user", "content": "hi"}])


def test_ollama_health_survives_a_malformed_tags_reply():
    llm = OllamaLLM("http://ollama.test", "qwen2.5:1.5b", timeout_s=2)
    transport = httpx.MockTransport(lambda r: httpx.Response(200, content=b"[]"))
    llm._client = httpx.Client(base_url="http://ollama.test", transport=transport)
    assert llm.health()["reachable"] is False


def test_long_sources_are_truncated_to_the_prompt_budget():
    from app.prompting import MAX_SOURCE_CHARS, ContextSource, render_source

    rendered = render_source(ContextSource("TCK-X-1", "ticket", "2026-01-01", "a" * (MAX_SOURCE_CHARS + 500)))
    assert "[truncated]" in rendered and len(rendered) < MAX_SOURCE_CHARS + 200
