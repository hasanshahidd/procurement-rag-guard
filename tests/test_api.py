"""HTTP contract: validation, auth, error envelope, ingest rules."""

from __future__ import annotations

import pytest

from conftest import HARBOR_BUYER, HARBOR_REQUESTER, REDLINE_BUYER, FakeLLM, ask, reply


def _assert_error(resp, status: int, code: str) -> dict:
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert set(body) == {"error"}
    assert body["error"]["code"] == code
    assert body["error"]["message"]
    return body["error"]


def test_health_reports_documents_and_model(make_client):
    body = make_client().get("/health").json()
    assert body["status"] == "ok"
    assert body["documents"] == 14
    assert body["llm"]["reachable"] is True


def test_health_is_degraded_when_model_unreachable(make_client):
    resp = make_client(FakeLLM(reachable=False)).get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "degraded"


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "not-a-real-key"}, {"X-API-Key": ""}])
def test_ask_requires_known_api_key(make_client, headers):
    _assert_error(ask(make_client(), "What is PR-1042?", headers), 401, "unauthorized")


@pytest.mark.parametrize(
    ("body", "field"),
    [
        ({"question": "hi"}, "question"),
        ({"question": "x" * 1001}, "question"),
        ({"question": "   "}, "question"),
        ({"question": "valid question", "top_k": 0}, "top_k"),
        ({"question": "valid question", "top_k": 9}, "top_k"),
        ({"question": "valid question", "tenant_id": "redline"}, "tenant_id"),
        ({"question": "valid question", "role": "approver"}, "role"),
        ({}, "question"),
    ],
)
def test_ask_validation_errors(make_client, body, field):
    resp = make_client().post("/ask", json=body, headers=HARBOR_BUYER)
    err = _assert_error(resp, 422, "validation_error")
    assert any(field in d["loc"] for d in err["details"])


def test_malformed_json_is_a_clean_422(make_client):
    resp = make_client().post(
        "/ask", content=b"{not json", headers={**HARBOR_BUYER, "Content-Type": "application/json"}
    )
    _assert_error(resp, 422, "validation_error")


def test_unknown_route_and_wrong_method_use_error_envelope(make_client):
    client = make_client()
    _assert_error(client.get("/nope"), 404, "not_found")
    _assert_error(client.get("/ask"), 405, "method_not_allowed")


def test_create_ticket_uses_tenant_from_api_key(make_client):
    resp = make_client().post(
        "/tickets", json={"title": "Monitor arms", "text": "Need 6 monitor arms for reception."}, headers=HARBOR_BUYER
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["tenant_id"] == "harbor"
    assert body["id"].startswith("TCK-")
    assert body["injection_flags"] == []
    assert len(body["content_sha256"]) == 64


def test_ingest_alias_and_duplicate_id(make_client):
    client = make_client()
    payload = {"id": "TCK-H-900", "title": "Chairs", "text": "Twelve chairs for the waiting room."}
    assert client.post("/ingest", json=payload, headers=HARBOR_BUYER).status_code == 201
    _assert_error(client.post("/tickets", json=payload, headers=HARBOR_BUYER), 409, "duplicate_id")


def test_requester_cannot_create_restricted_records(make_client):
    resp = make_client().post(
        "/tickets",
        json={"title": "Prices", "text": "LAP-14-STD now 900 AUD", "sensitivity": "restricted"},
        headers=HARBOR_REQUESTER,
    )
    _assert_error(resp, 403, "forbidden")


@pytest.mark.parametrize(
    "payload",
    [
        {"title": "ok title", "text": ""},
        {"title": "ok title", "text": "x" * 8001},
        {"title": "ok title", "text": "body", "kind": "pricing"},
        {"title": "ok title", "text": "body", "id": "lowercase-id"},
        {"title": "ok title", "text": "body", "facts": {"Bad Key": "1"}},
        {"title": "ok title", "text": "body", "tenant_id": "redline"},
    ],
)
def test_ticket_validation_errors(make_client, payload):
    _assert_error(make_client().post("/tickets", json=payload, headers=HARBOR_BUYER), 422, "validation_error")


def test_cannot_supersede_another_tenants_record(make_client):
    resp = make_client().post(
        "/tickets",
        json={"title": "New prices", "text": "LAP-14-STD 1,000 AUD", "supersedes": ["PRC-R-001"]},
        headers=HARBOR_BUYER,
    )
    _assert_error(resp, 422, "invalid_supersedes")


def test_source_lookup_is_tenant_and_role_scoped(make_client):
    client = make_client()
    assert client.get("/sources/TCK-H-101", headers=HARBOR_BUYER).status_code == 200
    _assert_error(client.get("/sources/PRC-R-001", headers=HARBOR_BUYER), 404, "not_found")
    _assert_error(client.get("/sources/PRC-H-002", headers=HARBOR_REQUESTER), 404, "not_found")
    _assert_error(client.get("/sources/TCK-H-101", headers=REDLINE_BUYER), 404, "not_found")
    # Missing and forbidden look identical, so ids cannot be probed.
    missing = client.get("/sources/PRC-X-999", headers=HARBOR_BUYER).json()
    forbidden = client.get("/sources/PRC-R-001", headers=HARBOR_BUYER).json()
    assert missing["error"]["code"] == forbidden["error"]["code"]


def test_lone_surrogate_is_rejected_not_a_500(make_client):
    raw = b'{"question": "price of \\ud800 laptops"}'
    resp = make_client().post("/ask", content=raw, headers={**HARBOR_BUYER, "Content-Type": "application/json"})
    _assert_error(resp, 422, "validation_error")


def test_rate_limit_counts_every_path_for_a_principal(make_client):
    client = make_client(rate_limit_per_min=3)
    codes = [client.get(f"/sources/TCK-X-{i}", headers=HARBOR_BUYER).status_code for i in range(3)]
    assert codes == [404, 404, 404]
    _assert_error(client.get("/sources/TCK-X-9", headers=HARBOR_BUYER), 429, "rate_limited")
    assert client.get("/sources/TCK-H-101", headers=REDLINE_BUYER).status_code == 404  # other principal unaffected


def test_rate_limit_returns_429(make_client):
    client = make_client(FakeLLM(reply("insufficient_evidence", "no")), rate_limit_per_min=2)
    for _ in range(2):
        assert ask(client, "What is the status of PR-1042?", HARBOR_BUYER).status_code == 200
    _assert_error(ask(client, "What is the status of PR-1042?", HARBOR_BUYER), 429, "rate_limited")
