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
from app.checks import (
    MIN_SUPPORT,
    contradicting_figures,
    describe_extra_field,
    numbers_in,
    run_output_checks,
    support_score,
    unmentioned_names,
)
from app.config import Settings
from app.embeddings import Embedder
from app.guard import sentence_spans
from app.llm import LLM
from app.prompting import MAX_SOURCE_CHARS, ContextSource, ModelOutput, build_messages, parse_model_output
from app.retrieval import Hit, hybrid_search, search, tokenize
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
from app.store import Document, Principal, Store, sha256
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
    raw_sha256: str | None = None  # hash of the raw model reply, so a blocked reply is traceable without storing it
    fact_pool: list[Document] = field(default_factory=list)  # visible, current records with structured facts

    def step(self, agent: str, summary: str) -> None:
        self.steps.append(AgentStep(agent=agent, summary=summary))


class Pipeline:
    def __init__(self, store: Store, llm: LLM, settings: Settings, audit: AuditLog, embedder: Embedder | None = None):
        self.store, self.llm, self.settings, self.audit = store, llm, settings, audit
        self.embedder = embedder

    def _rank(self, visible: list[Document], question: str, k: int) -> tuple[list[Hit], str]:
        s = self.settings
        if s.retrieval_mode == "hybrid" and self.embedder is not None:
            return hybrid_search(visible, question, k, self.embedder, s.min_retrieval_score, s.min_dense_score)
        return [h for h in search(visible, question, k) if h.score >= s.min_retrieval_score], "BM25"

    # agents ---------------------------------------------------------------

    def _retrieve(self, run: _Run, principal: Principal, question: str, top_k: int) -> list[Document]:
        visible = self.store.visible_docs(principal)
        superseded_by = {old: d.id for d in visible for old in d.supersedes}
        run.fact_pool = [d for d in visible if d.facts and d.id not in superseded_by]
        hits, mode = self._rank(visible, question, top_k + len(superseded_by))
        context: list[Document] = []
        for hit in hits:
            if hit.doc.id in superseded_by:
                run.excluded.append(
                    SourceNote(source_id=hit.doc.id, reason=f"superseded by {superseded_by[hit.doc.id]}")
                )
            elif len(context) < top_k:
                context.append(hit.doc)
                run.retrieved.append(
                    RetrievedSource(source_id=hit.doc.id, score=hit.score, bm25=hit.bm25, dense=hit.dense)
                )
        for newer in _replacements(run.excluded, superseded_by, visible, context):
            context.append(newer)  # an outdated hit pulls in the record that replaced it
            run.retrieved.append(RetrievedSource(source_id=newer.id, score=0.0))
        run.step("retriever", f"{mode} over {len(visible)} visible records, kept {[d.id for d in context]}")
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
        run.raw_sha256 = sha256(raw)
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
        missing = unmentioned_names(question, " ".join(f"{d.title} {d.scan.clean_text}" for d in context))
        if missing:
            # Same reply for another customer's name and a made-up one, so this reveals nothing.
            run.step("verifier", f"question names {missing}; no record available to this caller mentions them")
            answer = f"None of the records available to you mention {', '.join(missing)}."
            return self._finish(run, principal, question, "abstained", "question_names_unknown_entity", answer, [])
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
        cited = self._check_citations(run, out, by_id)
        failures = self._output_checks(question, out.answer, cited or context)
        run.security.output_checks.extend(failures)
        conflicts = find_conflicts(question, context, run.fact_pool)
        by_id.update({d.id: d for d in run.fact_pool})
        run.step(
            "verifier", f"citations ok={[d.id for d in cited]}, checks failed={failures}, conflicts={len(conflicts)}"
        )
        if failures:
            return self._finish(run, principal, question, "blocked", ",".join(failures), WITHHELD, [])
        if conflicts:
            return self._conflict(run, principal, question, out, cited, conflicts, by_id)
        if out.status == "insufficient_evidence":
            # Fixed text and no citations: free text here would be an uncited claim. `retrieved` lists what was read.
            reason = "model_reported_insufficient_evidence"
            return self._finish(run, principal, question, "abstained", reason, NO_EVIDENCE, [])
        if not cited:
            return self._finish(run, principal, question, "abstained", "no_valid_citations", NO_EVIDENCE, [])
        # A conflict needs two sides: the model's label is kept only with two supporting citations.
        status = "conflict" if out.status == "conflict" and len(cited) >= 2 else "answered"
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

    def _check_citations(self, run: _Run, out: ModelOutput, by_id: dict[str, Document]) -> list[Document]:
        """Keep citations that were in this request's context and whose text supports the answer."""
        kept: list[Document] = []
        for cid in out.citations:
            doc = by_id.get(cid)
            if doc is None:
                shown = cid if _DOC_ID.match(cid) else "<not a source id>"  # never echo free text from the model
                reason = "not among the sources provided for this request"
            elif (citation := make_citation(doc, out.answer)) is None:
                shown, reason = cid, "cited source does not support the answer"
            elif contradicting_figures(citation.excerpt, out.answer):
                shown, reason = cid, "cited text states different figures than the answer"
            else:
                kept.append(doc)
                continue
            run.security.rejected_citations.append(SourceNote(source_id=shown, reason=reason))
        return kept

    def _output_checks(self, question: str, answer: str, backing: list[Document]) -> list[str]:
        """backing: the cited sources (or, with no valid citation, everything the model was shown).

        Allowlist: numbers must appear in what the model was shown of those sources (header and the part
        of the text the guard kept, within the prompt budget) or in the question. Text the guard removed
        never counts. Figures found only in requester-submitted records are reported separately.
        """

        def numbers_of(docs: list[Document]) -> set[str]:
            return numbers_in(" ".join(f"{d.id} {d.updated_at} {d.scan.clean_text[:MAX_SOURCE_CHARS]}" for d in docs))

        typed = numbers_in(question)
        approved = {
            key.split(":", 1)[1].upper()
            for d in backing
            for key, value in d.facts
            if key.startswith("pr_status:") and value == "approved"
        }
        trusted = typed | numbers_of([d for d in backing if d.trusted])
        return run_output_checks(answer, typed | numbers_of(backing), approved, trusted)

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
        citations = [c for c in (make_citation(d, answer) for d in docs) if c]
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
        self.audit.write(principal, question, response, model_output_sha256=run.raw_sha256)
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


def find_conflicts(question: str, context: list[Document], pool: list[Document] | None = None) -> list[Conflict]:
    """Facts relevant to the question that disagree.

    Context records count when the fact's topic matches the question. Other
    visible records (pool) also count when the fact's subject, such as the SKU,
    is named in the question, so a conflict cannot hide outside the top-k.
    """
    words = set(tokenize(question))
    asked = question.upper()
    by_key: dict[str, dict[str, ConflictValue]] = defaultdict(dict)
    candidates = [(d, False) for d in context] + [(d, True) for d in pool or []]
    for doc, needs_subject in candidates:
        for key, value in doc.facts:
            topic, _, subject = key.partition(":")
            if not FACT_TOPICS.get(topic, frozenset()) & words:
                continue
            if needs_subject and (not subject or subject.upper() not in asked):
                continue
            by_key[key].setdefault(doc.id, ConflictValue(source_id=doc.id, value=value, updated_at=doc.updated_at))
    return [
        Conflict(fact=k, values=list(v.values()))
        for k, v in sorted(by_key.items())
        if len({x.value for x in v.values()}) > 1
    ]


def describe_conflicts(conflicts: list[Conflict]) -> str:
    parts = [
        f"{c.fact}: " + "; ".join(f"{v.source_id} ({v.updated_at}) says {v.value}" for v in c.values) for c in conflicts
    ]
    return (
        "The records disagree, so I am not picking one. "
        + " | ".join(parts)
        + ". Please confirm with the source owner."
    )


def make_citation(doc: Document, answer: str) -> Citation | None:
    """Pick the sentence of the stored text that best supports the answer, or None if none does.

    Sentences the guard removed are never cited. A sentence must share at least
    MIN_SUPPORT with the answer (an id or number, or two words); otherwise the
    source is in context but does not back the claim. The excerpt is a verbatim
    slice of doc.text, so anyone can check it with text[start:end] and the hash.
    """
    best: tuple[float, int, int] | None = None
    for start, end in sentence_spans(doc.text):
        while start < end and doc.text[start].isspace():
            start += 1
        sentence = doc.text[start:end].rstrip()
        terms = set(tokenize(sentence, parts=False))
        if not terms or doc.scan.overlaps(start, end):
            continue
        support = support_score(sentence, answer)
        if support < MIN_SUPPORT:
            continue
        score = support / math.sqrt(len(terms))  # favour short, on-point sentences
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
