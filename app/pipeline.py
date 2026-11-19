"""The /ask pipeline, written as a chain of small agents.

retriever    BM25 over the caller's visible documents, drops superseded records
tool_runner  read-only vendor_quote tool, chosen by code from the question
guard        strips instruction-like sentences from every source
answerer     local open-weight LLM, sees only sanitized, permitted sources
verifier     code checks: citations, conflicts, value leaks, approval claims

Every decision that matters for security is made outside the model. The tests
use a deliberately compromised fake model to prove that.
"""

from __future__ import annotations

import math
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field

from app.audit import AuditLog
from app.checks import describe_extra_field, numbers_in, run_output_checks
from app.config import Settings
from app.guard import scan, sentence_spans
from app.llm import LLM
from app.prompting import ContextSource, ModelOutput, build_messages, parse_model_output
from app.retrieval import search, tokenize
from app.schemas import (
    AgentStep,
    AskResponse,
    BlockedAction,
    Citation,
    Conflict,
    ConflictValue,
    FlaggedSource,
    RetrievedSource,
    SecurityReport,
    SourceNote,
)
from app.store import Document, Principal, Store
from app.tools import run_tool

_SKU = re.compile(r"\b[A-Z]{3}-\d{2}-[A-Z]{3}\b", re.I)
_DOC_ID = re.compile(r"^[A-Z]{2,5}-[A-Z0-9][A-Z0-9-]{0,30}$")
# Which question words make a structured fact relevant for conflict checks.
FACT_TOPICS: dict[str, frozenset[str]] = {
    "lead_time_business_days": frozenset(
        "ship shipping shipped deliver delivery lead eta arrive dispatch when".split()
    ),
    "unit_price_aud": frozenset("price prices pricing cost unit pay negotiated".split()),
    "list_price_aud": frozenset("list catalogue catalog rrp".split()),
    "approval_threshold_aud": frozenset("approval approve threshold sign-off signoff quotes limit".split()),
    "pr_status": frozenset("status approved pending state".split()),
}
WITHHELD = "The answer was withheld because it failed a safety check. The event has been logged."
NO_EVIDENCE = "I could not find enough evidence in the records you have access to."


@dataclass
class _Run:
    """Mutable scratchpad for one request; never shared between requests."""

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    steps: list[AgentStep] = field(default_factory=list)
    security: SecurityReport = field(default_factory=SecurityReport)
    excluded: list[SourceNote] = field(default_factory=list)
    retrieved: list[RetrievedSource] = field(default_factory=list)

    def step(self, agent: str, summary: str) -> None:
        self.steps.append(AgentStep(agent=agent, summary=summary))


class Pipeline:
    def __init__(self, store: Store, llm: LLM, settings: Settings, audit: AuditLog):
        self.store, self.llm, self.settings, self.audit = store, llm, settings, audit

    # agents ---------------------------------------------------------------

    def _retrieve(self, run: _Run, principal: Principal, question: str, top_k: int) -> list[Document]:
        visible = self.store.visible_docs(principal)
        superseded_by = {old: d.id for d in visible for old in d.supersedes}
        hits = search(visible, question, top_k + len(superseded_by))
        context: list[Document] = []
        for hit in hits:
            if hit.score < self.settings.min_retrieval_score:
                continue
            if hit.doc.id in superseded_by:
                run.excluded.append(
                    SourceNote(source_id=hit.doc.id, reason=f"superseded by {superseded_by[hit.doc.id]}")
                )
            elif len(context) < top_k:
                context.append(hit.doc)
                run.retrieved.append(RetrievedSource(source_id=hit.doc.id, score=hit.score))
        for newer in _replacements(run.excluded, superseded_by, visible, context):
            context.append(newer)  # an outdated hit pulls in the record that replaced it
            run.retrieved.append(RetrievedSource(source_id=newer.id, score=0.0))
        run.step("retriever", f"{len(visible)} visible records, kept {[d.id for d in context]}")
        return context

    def _run_tools(self, run: _Run, principal: Principal, question: str) -> list[Document]:
        skus = sorted({s.upper() for s in _SKU.findall(question)})
        if not skus:
            run.step("tool_runner", "no SKU in question, no tool call")
            return []
        if "restricted" not in principal.clearance:
            run.step("tool_runner", f"vendor_quote skipped: role '{principal.role}' lacks clearance")
            return []
        outputs = [doc for sku in skus for doc in run_tool("vendor_quote", self.store, principal, sku)]
        run.step("tool_runner", f"vendor_quote({', '.join(skus)}) returned {[d.id for d in outputs]}")
        return outputs

    def _guard(self, run: _Run, context: list[Document]) -> list[ContextSource]:
        for doc in context:
            if doc.scan.flagged:
                run.security.flagged_sources.append(FlaggedSource(source_id=doc.id, rules=doc.scan.rules))
        run.step("guard", f"flagged {[f.source_id for f in run.security.flagged_sources] or 'nothing'}")
        return [ContextSource(d.id, d.kind, d.updated_at, d.scan.clean_text) for d in context]

    def _answer(self, run: _Run, question: str, sources: list[ContextSource]) -> ModelOutput | None:
        started = time.perf_counter()
        raw = self.llm.chat(build_messages(question, sources))
        parsed = parse_model_output(raw)
        took = time.perf_counter() - started
        run.step("answerer", f"{self.llm.name} replied in {took:.1f}s, contract {'ok' if parsed else 'violated'}")
        return parsed

    # entry point ----------------------------------------------------------

    def ask(self, principal: Principal, question: str, top_k: int) -> AskResponse:
        run = _Run()
        context = self._retrieve(run, principal, question, top_k)
        context += self._run_tools(run, principal, question)
        if not context:
            run.step("verifier", "no evidence above threshold, abstaining without calling the model")
            return self._finish(run, principal, question, "abstained", "no_relevant_sources", NO_EVIDENCE, [])
        sources = self._guard(run, context)
        parsed = self._answer(run, question, sources)
        return self._verify(run, principal, question, context, parsed)

    # verifier -------------------------------------------------------------

    def _verify(
        self, run: _Run, principal: Principal, question: str, context: list[Document], out: ModelOutput | None
    ) -> AskResponse:
        if out is None:
            run.step("verifier", "model reply did not match the JSON contract")
            return self._finish(run, principal, question, "abstained", "model_output_unverifiable", NO_EVIDENCE, [])
        by_id = {d.id: d for d in context}
        for key, value in out.extra_keys.items():
            run.security.blocked_actions.append(
                BlockedAction(origin="model_output", detail=describe_extra_field(key, value))
            )
        cited = [by_id[c] for c in out.citations if c in by_id]
        for c in out.citations:
            if c not in by_id:
                shown = c if _DOC_ID.match(c) else "<not a source id>"  # never echo free text from the model
                run.security.rejected_citations.append(
                    SourceNote(source_id=shown, reason="not among the sources provided for this request")
                )
        failures = self._output_checks(principal, question, out.answer, context)
        run.security.output_checks.extend(failures)
        conflicts = find_conflicts(question, context)
        run.step(
            "verifier", f"citations ok={[d.id for d in cited]}, checks failed={failures}, conflicts={len(conflicts)}"
        )
        if failures:
            return self._finish(run, principal, question, "blocked", ",".join(failures), WITHHELD, [])
        if conflicts:
            return self._conflict(run, principal, question, out, cited, conflicts, by_id)
        if out.status == "insufficient_evidence":
            # No citations on an abstention: they would read as support for a claim. `retrieved` lists what was read.
            reason = "model_reported_insufficient_evidence"
            return self._finish(run, principal, question, "abstained", reason, out.answer or NO_EVIDENCE, [])
        if not cited:
            return self._finish(run, principal, question, "abstained", "no_valid_citations", NO_EVIDENCE, [])
        status = "conflict" if out.status == "conflict" else "answered"
        return self._finish(run, principal, question, status, None, out.answer, cited)

    def _conflict(
        self,
        run: _Run,
        principal: Principal,
        question: str,
        out: ModelOutput,
        cited: list[Document],
        conflicts: list[Conflict],
        by_id: dict[str, Document],
    ) -> AskResponse:
        involved = [by_id[v.source_id] for c in conflicts for v in c.values]
        docs = list(dict.fromkeys(cited + involved))
        if out.status == "conflict" and out.answer:
            answer, reason = out.answer, None
        else:
            answer, reason = describe_conflicts(conflicts), "model_missed_conflict"
        return self._finish(run, principal, question, "conflict", reason, answer, docs, conflicts)

    def _output_checks(self, principal: Principal, question: str, answer: str, context: list[Document]) -> list[str]:
        # Only what the model was actually shown counts as allowed, not text the guard removed.
        allowed = numbers_in(question) | {n for d in context for n in numbers_in(d.scan.clean_text)}
        approved = {
            key.split(":", 1)[1].upper()
            for d in context
            for key, value in d.facts
            if key.startswith("pr_status:") and value == "approved"
        }
        return run_output_checks(answer, self.store.hidden_amounts(principal), allowed, approved)

    def _finish(
        self,
        run: _Run,
        principal: Principal,
        question: str,
        status: str,
        reason: str | None,
        answer: str,
        docs: list[Document],
        conflicts: list[Conflict] | None = None,
    ) -> AskResponse:
        citations = [c for c in (make_citation(d, f"{question} {answer}") for d in docs) if c]
        response = AskResponse(
            trace_id=run.trace_id,
            status=status,
            reason=reason,
            answer=answer,
            citations=citations,
            conflicts=conflicts or [],
            excluded_sources=run.excluded,
            retrieved=run.retrieved,
            security=run.security,
            agents=run.steps,
            model=self.llm.name,
        )
        self.audit.write(principal, question, response)
        return response


def _replacements(
    excluded: list[SourceNote], superseded_by: dict[str, str], visible: list[Document], context: list[Document]
) -> list[Document]:
    """Newest version of each excluded record, following chains, that is not already in context."""
    by_id = {d.id: d for d in visible}
    have = {d.id for d in context}
    found: list[Document] = []
    for note in excluded:
        current, seen = note.source_id, set()
        while current in superseded_by and current not in seen:
            seen.add(current)
            current = superseded_by[current]
        if current not in have and current in by_id:
            have.add(current)
            found.append(by_id[current])
    return found


def find_conflicts(question: str, context: list[Document]) -> list[Conflict]:
    words = set(tokenize(question))
    by_key: dict[str, list[ConflictValue]] = defaultdict(list)
    for doc in context:
        for key, value in doc.facts:
            if FACT_TOPICS.get(key.split(":", 1)[0], frozenset()) & words:
                by_key[key].append(ConflictValue(source_id=doc.id, value=value, updated_at=doc.updated_at))
    return [Conflict(fact=k, values=v) for k, v in sorted(by_key.items()) if len({x.value for x in v}) > 1]


def describe_conflicts(conflicts: list[Conflict]) -> str:
    parts = [
        f"{c.fact}: " + "; ".join(f"{v.source_id} ({v.updated_at}) says {v.value}" for v in c.values) for c in conflicts
    ]
    return (
        "The records disagree, so I am not picking one. "
        + " | ".join(parts)
        + ". Please confirm with the source owner."
    )


def make_citation(doc: Document, focus: str) -> Citation | None:
    """Pick the unflagged sentence of the stored text that best matches the answer.

    The excerpt is a verbatim slice of doc.text, so anyone can check it with
    text[start:end] and the content hash.
    """
    focus_terms = set(tokenize(focus))
    best: tuple[float, int, int] | None = None
    for start, end in sentence_spans(doc.text):
        while start < end and doc.text[start].isspace():
            start += 1
        sentence = doc.text[start:end].rstrip()
        terms = set(tokenize(sentence))
        if not terms or scan(sentence).flagged:
            continue
        score = len(terms & focus_terms) / math.sqrt(len(terms))  # favour short, on-point sentences
        if best is None or score > best[0]:
            best = (score, start, start + len(sentence))
    if best is None:
        return None
    _, start, end = best
    return Citation(
        source_id=doc.id,
        kind=doc.kind,
        title=doc.title,
        excerpt=doc.text[start:end],
        start=start,
        end=end,
        content_sha256=doc.content_sha256,
    )
