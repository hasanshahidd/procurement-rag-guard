"""Request and response models. Requests reject unknown fields, so a caller
cannot smuggle tenant_id or role through the body."""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

DocId = Annotated[str, Field(pattern=r"^[A-Z]{2,5}-[A-Z0-9][A-Z0-9-]{0,30}$")]
_FACT_KEY = re.compile(r"[a-z_]{2,40}(:[A-Z0-9-]{1,30})?")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    @field_validator("*")
    @classmethod
    def _encodable(cls, v: object) -> object:
        # JSON allows lone surrogates ("\ud800"); they would break hashing and logging later.
        values = v.values() if isinstance(v, dict) else v if isinstance(v, list) else [v]
        for item in values:
            if isinstance(item, str):
                try:
                    item.encode("utf-8")
                except UnicodeEncodeError as exc:
                    raise ValueError("text contains invalid unicode") from exc
        return v


class AskRequest(_Strict):
    question: str = Field(min_length=3, max_length=1000)
    top_k: StrictInt = Field(default=4, ge=1, le=8)  # strict: "3" and true are rejected, not coerced


class TicketCreate(_Strict):
    id: DocId | None = None
    kind: Literal["ticket", "log", "kb"] = "ticket"
    title: str = Field(min_length=3, max_length=200)
    text: str = Field(min_length=1, max_length=8000)
    sensitivity: Literal["public", "internal", "restricted"] = "internal"
    updated_at: date | None = None
    supersedes: list[DocId] = Field(default_factory=list, max_length=10)
    facts: dict[str, str] = Field(default_factory=dict, max_length=20)

    @field_validator("updated_at")
    @classmethod
    def _not_in_future(cls, v: date | None) -> date | None:
        # A future date would make a record look newer than everything else.
        if v is not None and v > date.today() + timedelta(days=1):
            raise ValueError("updated_at cannot be in the future")
        return v

    @field_validator("facts")
    @classmethod
    def _fact_keys(cls, v: dict[str, str]) -> dict[str, str]:
        bad = [k for k in v if not _FACT_KEY.fullmatch(k)]
        if bad:
            raise ValueError(f"fact keys must look like 'unit_price_aud:SKU-1', got {bad}")
        if any(len(val) > 100 for val in v.values()):
            raise ValueError("fact values are limited to 100 characters")
        return v


class TicketCreated(BaseModel):
    id: str
    tenant_id: str
    kind: str
    title: str
    sensitivity: str
    updated_at: str
    content_sha256: str
    injection_flags: list[str]


class Citation(BaseModel):
    source_id: str
    kind: str
    title: str
    excerpt: str = Field(description="Verbatim substring of the stored source text")
    start: int = Field(description="Offset of the excerpt in the source text")
    end: int
    content_sha256: str


class ConflictValue(BaseModel):
    source_id: str
    value: str
    updated_at: str


class Conflict(BaseModel):
    fact: str
    values: list[ConflictValue]


class SourceNote(BaseModel):
    source_id: str
    reason: str


class FlaggedSource(BaseModel):
    source_id: str
    rules: list[str]


class BlockedAction(BaseModel):
    origin: str
    detail: str


class SecurityReport(BaseModel):
    flagged_sources: list[FlaggedSource] = []
    blocked_actions: list[BlockedAction] = []
    rejected_citations: list[SourceNote] = []
    output_checks: list[str] = []


class RetrievedSource(BaseModel):
    source_id: str
    score: float = Field(description="Fused rank score used for ordering")
    bm25: float | None = None
    dense: float | None = Field(default=None, description="Cosine similarity, when embeddings are on")


class AgentStep(BaseModel):
    agent: str
    summary: str


class AskResponse(BaseModel):
    trace_id: str
    status: Literal["answered", "abstained", "conflict", "blocked"]
    reason: str | None = None
    answer: str
    citations: list[Citation] = []
    conflicts: list[Conflict] = []
    excluded_sources: list[SourceNote] = []
    retrieved: list[RetrievedSource] = []
    security: SecurityReport = SecurityReport()
    agents: list[AgentStep] = []
    model: str


class FlaggedSpan(BaseModel):
    start: int
    end: int
    rules: list[str]


class SourceView(BaseModel):
    id: str
    tenant_id: str
    kind: str
    title: str
    text: str
    updated_at: str
    sensitivity: str
    content_sha256: str
    injection_flags: list[str]
    flagged_spans: list[FlaggedSpan] = Field(
        default=[], description="Sentences the guard removed before the model saw this source (code point offsets)"
    )


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    documents: int
    llm: dict
    retrieval: dict = {}


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list | dict | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
