"""Retrieval, citations, supersession, conflicts and abstention."""

from __future__ import annotations

import hashlib
import threading

from app.pipeline import NO_EVIDENCE, _replacements
from app.retrieval import search, tokenize
from app.schemas import SourceNote
from app.store import Document
from conftest import HARBOR_BUYER, HARBOR_REQUESTER, INJECTION_QUESTION, FakeLLM, ask, reply


def test_tokenize_keeps_sku_and_its_parts():
    assert {"lap-14-std", "lap", "14", "std"} <= set(tokenize("Price for LAP-14-STD?"))
    assert "the" not in tokenize("the price")


def test_search_ranks_the_ticket_first(store):
    harbor = store.authenticate("demo-harbor-buyer")
    hits = search(store.visible_docs(harbor), "status of PR-1042", 3)
    assert hits[0].doc.id == "TCK-H-101"


def test_search_only_sees_the_callers_tenant(store):
    harbor = store.authenticate("demo-harbor-buyer")
    hits = search(store.visible_docs(harbor), "Redline Freight negotiated price list LAP-14-STD 1,045", 20)
    assert hits
    assert all(h.doc.tenant_id in ("harbor", "shared") for h in hits)


def test_requester_does_not_retrieve_restricted_pricing(store):
    requester = store.authenticate("demo-harbor-requester")
    ids = {h.doc.id for h in search(store.visible_docs(requester), "negotiated price LAP-14-STD", 20)}
    assert not ids & {"PRC-H-001", "PRC-H-002", "PRC-R-001"}


def test_store_survives_concurrent_ingest_and_reads(store):
    harbor = store.authenticate("demo-harbor-buyer")
    errors: list[Exception] = []

    def writer(n: int) -> None:
        try:
            for i in range(150):
                store.add(
                    Document.build(
                        id=f"TCK-C{n}-{i}",
                        tenant_id="harbor",
                        kind="ticket",
                        title="load",
                        text="Load test 1,234 units.",
                        updated_at="2026-01-01",
                        sensitivity="internal",
                    )
                )
        except Exception as exc:  # noqa: BLE001 - collected and asserted below
            errors.append(exc)

    def reader() -> None:
        try:
            for _ in range(150):
                store.visible_docs(harbor)
                store.hidden_amounts(harbor)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(2)] + [
        threading.Thread(target=reader) for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(store.visible_docs(harbor)) >= 300


def test_answer_with_verifiable_citations(make_client):
    llm = FakeLLM(reply("answered", "PR-1042 is pending approval.", ["TCK-H-101"]))
    client = make_client(llm)
    body = ask(client, "What is the status of PR-1042?", HARBOR_BUYER).json()
    assert body["status"] == "answered"
    [cite] = body["citations"]
    assert cite["source_id"] == "TCK-H-101"
    source = client.get(f"/sources/{cite['source_id']}", headers=HARBOR_BUYER).json()
    assert source["text"][cite["start"] : cite["end"]] == cite["excerpt"]
    assert hashlib.sha256(source["text"].encode()).hexdigest() == cite["content_sha256"]
    assert "pending approval" in cite["excerpt"]


def test_citation_outside_context_is_rejected_and_answer_withheld(make_client):
    llm = FakeLLM(reply("answered", "PR-1042 is pending.", ["KB-999", "PRC-R-001"]))
    body = ask(make_client(llm), "What is the status of PR-1042?", HARBOR_BUYER).json()
    assert body["status"] == "abstained"
    assert body["reason"] == "no_valid_citations"
    assert {c["source_id"] for c in body["security"]["rejected_citations"]} == {"KB-999", "PRC-R-001"}
    assert body["citations"] == []


def test_superseded_policy_is_excluded(make_client):
    llm = FakeLLM(reply("answered", "Above 5,000 AUD you need an approver and two quotes.", ["KB-001"]))
    client = make_client(llm)
    body = ask(client, "What approval is needed for a 7,000 AUD laptop order?", HARBOR_BUYER).json()
    assert {"source_id": "KB-002", "reason": "superseded by KB-001"} in body["excluded_sources"]
    assert "KB-002" not in llm.last_prompt
    assert "10,000" not in llm.last_prompt
    assert body["status"] == "answered"


def test_outdated_hit_pulls_in_its_replacement(make_client):
    llm = FakeLLM(reply("answered", "Above 5,000 AUD needs two quotes.", ["KB-001"]))
    body = ask(make_client(llm), "approval policy v2 retired July 2026", HARBOR_BUYER, top_k=1).json()
    assert [r["source_id"] for r in body["retrieved"]] == ["KB-001"]
    assert body["excluded_sources"] == [{"source_id": "KB-002", "reason": "superseded by KB-001"}]
    assert "[SOURCE id=KB-001" in llm.last_prompt and "KB-002" not in llm.last_prompt


def test_replacement_follows_supersede_chains(store):
    def doc(doc_id: str, supersedes: list[str]) -> Document:
        return Document.build(
            id=doc_id,
            tenant_id="harbor",
            kind="kb",
            title=doc_id,
            text="x",
            updated_at="2026-01-01",
            sensitivity="internal",
            supersedes=supersedes,
        )

    a, b, c = doc("KB-A", []), doc("KB-B", ["KB-A"]), doc("KB-C", ["KB-B"])
    superseded_by = {"KB-A": "KB-B", "KB-B": "KB-C"}
    found = _replacements([SourceNote(source_id="KB-A", reason="")], superseded_by, [a, b, c], [])
    assert [d.id for d in found] == ["KB-C"]
    assert _replacements([SourceNote(source_id="KB-A", reason="")], superseded_by, [a, b, c], [c]) == []


def test_outdated_price_list_never_reaches_the_model(make_client):
    llm = FakeLLM(reply("answered", "1,180 AUD per unit.", ["PRC-H-002"]))
    body = ask(make_client(llm), "What is our negotiated price for LAP-14-STD?", HARBOR_BUYER).json()
    assert any(e["source_id"] == "PRC-H-001" for e in body["excluded_sources"])
    assert "1,240" not in llm.last_prompt
    assert body["status"] == "answered"


def test_conflicting_lead_times_override_a_confident_answer(make_client):
    llm = FakeLLM(reply("answered", "It ships in 10 business days.", ["TCK-H-102"]))
    body = ask(make_client(llm), "When can the LAP-14-STD laptops for PR-1042 ship?", HARBOR_BUYER).json()
    assert body["status"] == "conflict"
    assert body["reason"] == "model_missed_conflict"
    [conflict] = body["conflicts"]
    assert conflict["fact"] == "lead_time_business_days:LAP-14-STD"
    assert {(v["source_id"], v["value"]) for v in conflict["values"]} == {("TCK-H-102", "10"), ("LOG-H-201", "30")}
    assert {"TCK-H-102", "LOG-H-201"} <= {c["source_id"] for c in body["citations"]}
    assert "10" in body["answer"] and "30" in body["answer"]


def test_conflict_outside_top_k_is_still_found(make_client):
    """Only one record fits in context, but the other side of the conflict is still checked."""
    llm = FakeLLM(reply("answered", "It ships in 10 business days.", ["TCK-H-102"]))
    body = ask(make_client(llm), "When can LAP-14-STD ship?", HARBOR_BUYER, top_k=1).json()
    assert "LOG-H-201" not in {r["source_id"] for r in body["retrieved"]}
    assert body["status"] == "conflict"
    assert {v["source_id"] for v in body["conflicts"][0]["values"]} == {"TCK-H-102", "LOG-H-201"}


def test_conflict_reported_by_model_is_kept(make_client):
    text = "TCK-H-102 says 10 business days but LOG-H-201 says 30."
    llm = FakeLLM(reply("conflict", text, ["TCK-H-102", "LOG-H-201"]))
    body = ask(make_client(llm), "When can the LAP-14-STD laptops for PR-1042 ship?", HARBOR_BUYER).json()
    assert body["status"] == "conflict"
    assert body["reason"] is None
    assert body["answer"] == text


def test_model_conflict_label_needs_two_supporting_sources(make_client):
    llm = FakeLLM(reply("conflict", "PR-1042 is pending approval.", ["TCK-H-101"]))
    body = ask(make_client(llm), "What is the status of PR-1042?", HARBOR_BUYER).json()
    assert body["status"] == "answered"


def test_no_relevant_sources_abstains_without_calling_model(make_client):
    llm = FakeLLM(reply("answered", "Team A.", ["KB-001"]))
    body = ask(make_client(llm), "Who won the football final last night?", HARBOR_BUYER).json()
    assert body["status"] == "abstained"
    assert body["reason"] == "no_relevant_sources"
    assert llm.calls == []


def test_model_insufficient_evidence_becomes_abstention(make_client):
    llm = FakeLLM(reply("insufficient_evidence", "The sources do not mention a warranty.", ["KB-005"]))
    body = ask(make_client(llm), "What is the warranty period for TAB-10-RUG?", HARBOR_BUYER).json()
    assert body["status"] == "abstained"
    assert body["reason"] == "model_reported_insufficient_evidence"
    assert body["answer"] == NO_EVIDENCE  # the model's own wording would be an uncited claim
    assert body["citations"] == []
    assert body["retrieved"]


def test_tool_output_is_a_citable_source_without_the_payload(make_client):
    llm = FakeLLM(
        reply("answered", "Bluegate quoted 1,175 AUD per unit; PR-1042 is pending.", ["TOOL-Q-5531", "TCK-H-101"])
    )
    client = make_client(llm)
    body = ask(client, INJECTION_QUESTION, HARBOR_BUYER).json()
    assert body["status"] == "answered"
    cites = {c["source_id"]: c for c in body["citations"]}
    assert "1,175 AUD per unit" in cites["TOOL-Q-5531"]["excerpt"]
    assert "SYSTEM" not in cites["TOOL-Q-5531"]["excerpt"]
    tool_src = client.get("/sources/TOOL-Q-5531", headers=HARBOR_BUYER).json()
    assert (
        tool_src["text"][cites["TOOL-Q-5531"]["start"] : cites["TOOL-Q-5531"]["end"]] == cites["TOOL-Q-5531"]["excerpt"]
    )


def test_requester_gets_no_tool_call_and_no_pricing(make_client):
    llm = FakeLLM(reply("insufficient_evidence", "Not in my sources.", []))
    body = ask(make_client(llm), "What is our negotiated price for LAP-14-STD?", HARBOR_REQUESTER).json()
    assert "TOOL-" not in llm.last_prompt
    assert "1,180" not in llm.last_prompt and "1,175" not in llm.last_prompt
    assert any("lacks clearance" in s["summary"] for s in body["agents"])
