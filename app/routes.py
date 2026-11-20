"""HTTP endpoints. Each request gets its services from app.state, set once by create_app."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date

from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse
from fastapi.security import APIKeyHeader

from app import __version__
from app.audit import AuditLog
from app.config import Settings
from app.embeddings import Embedder
from app.errors import AppError
from app.llm import LLM
from app.pipeline import Pipeline
from app.schemas import (
    AskRequest,
    AskResponse,
    FlaggedSpan,
    HealthResponse,
    SourceView,
    TicketCreate,
    TicketCreated,
)
from app.store import CURATOR_ROLES, Document, Principal, Store

log = logging.getLogger("procurement")
router = APIRouter()
_api_key = APIKeyHeader(
    name="X-API-Key", auto_error=False, description="Synthetic demo key, see data/seed/principals.json"
)
_ID_PREFIX = {"ticket": "TCK", "log": "LOG", "kb": "KB"}


class RateLimiter:
    """Sliding one-minute window per principal. shortcut: in-process only, use Redis behind a load balancer."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            window = self._hits[key]
            while window and now - window[0] > 60:
                window.popleft()
            if len(window) >= self.per_minute:
                raise AppError(429, "rate_limited", f"Limit is {self.per_minute} requests per minute")
            window.append(now)


@dataclass(frozen=True)
class Services:
    settings: Settings
    store: Store
    llm: LLM
    embedder: Embedder | None
    audit: AuditLog
    pipeline: Pipeline
    limiter: RateLimiter


def services(request: Request) -> Services:
    return request.app.state.services


def principal(
    request: Request, api_key: str | None = Depends(_api_key), svc: Services = Depends(services)
) -> Principal:
    found = svc.store.authenticate(api_key)
    if found is None:
        # Failed attempts are limited too, per client address, so keys cannot be guessed at full speed.
        svc.limiter.check(f"anonymous:{request.client.host if request.client else 'unknown'}")
        raise AppError(401, "unauthorized", "Missing or unknown X-API-Key header")
    svc.limiter.check(f"{found.tenant_id}:{found.role}")  # per principal, not per path, so ids cannot dodge it
    return found


@router.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/ui/")


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health(svc: Services = Depends(services)) -> HealthResponse:
    llm_state = {"backend": svc.settings.llm_backend, "model": svc.llm.name, **svc.llm.health()}
    retrieval = {"mode": "bm25" if svc.embedder is None else "hybrid"}
    if svc.embedder is not None:
        retrieval.update(embedding_model=svc.embedder.name, **svc.embedder.health())
    return HealthResponse(
        status="ok" if llm_state["reachable"] else "degraded",
        version=__version__,
        documents=len(svc.store),
        llm=llm_state,
        retrieval=retrieval,
    )


@router.post("/tickets", response_model=TicketCreated, status_code=201, tags=["data"])
@router.post("/ingest", response_model=TicketCreated, status_code=201, tags=["data"], include_in_schema=False)
def create_ticket(
    body: TicketCreate, who: Principal = Depends(principal), svc: Services = Depends(services)
) -> TicketCreated:
    doc = svc.store.add(build_document(svc.store, who, body))
    svc.audit.write_ingest(who, doc)
    if doc.scan.flagged:
        log.warning("ingested %s for %s with injection flags %s", doc.id, who.tenant_id, doc.scan.rules)
    return TicketCreated(
        id=doc.id,
        tenant_id=doc.tenant_id,
        kind=doc.kind,
        title=doc.title,
        sensitivity=doc.sensitivity,
        updated_at=doc.updated_at,
        content_sha256=doc.content_sha256,
        injection_flags=doc.scan.rules,
    )


@router.post("/ask", response_model=AskResponse, tags=["assistant"])
def ask(body: AskRequest, who: Principal = Depends(principal), svc: Services = Depends(services)) -> AskResponse:
    # Sync on purpose: model calls block, so FastAPI runs this in its threadpool.
    response = svc.pipeline.ask(who, body.question, body.top_k)
    if response.status == "blocked" or response.security.blocked_actions:
        log.warning("trace %s blocked for %s: %s", response.trace_id, who.tenant_id, response.reason)
    return response


@router.get("/sources/{source_id}", response_model=SourceView, tags=["assistant"])
def get_source(source_id: str, who: Principal = Depends(principal), svc: Services = Depends(services)) -> SourceView:
    doc = svc.store.get_visible(who, source_id)
    if doc is None:  # same answer for "missing" and "not yours", so ids cannot be probed
        raise AppError(404, "not_found", f"No source '{source_id[:60]}' available to this caller")
    return SourceView(
        **{
            k: getattr(doc, k)
            for k in ("id", "tenant_id", "kind", "title", "text", "updated_at", "sensitivity", "content_sha256")
        },
        injection_flags=doc.scan.rules,
        flagged_spans=[FlaggedSpan(start=s, end=e, rules=rules) for s, e, rules in doc.scan.spans],
    )


def build_document(store: Store, who: Principal, body: TicketCreate) -> Document:
    """Turn a submitted record into a Document, enforcing who may create what."""
    if body.sensitivity not in who.clearance:
        raise AppError(403, "forbidden", f"Role '{who.role}' cannot create {body.sensitivity} records")
    if (body.facts or body.supersedes) and who.role not in CURATOR_ROLES:
        raise AppError(403, "forbidden", "Only buyers and approvers can set facts or supersede records")
    if who.role not in CURATOR_ROLES and (body.kind != "ticket" or body.id):
        # Requesters file tickets; knowledge-base entries, logs and chosen ids are curator actions.
        # Server-assigned ids also mean a requester cannot test which ids exist.
        raise AppError(403, "forbidden", "Requesters can only file tickets, with a server-assigned id")
    if body.id and not body.id.startswith(_ID_PREFIX[body.kind] + "-"):
        raise AppError(422, "reserved_id", f"A {body.kind} id must start with {_ID_PREFIX[body.kind]}-")
    if any(key.startswith("pr_status") for key in body.facts):
        raise AppError(422, "reserved_fact", "Purchase status comes from the purchasing system, not from tickets")
    for sid in body.supersedes:
        target = store.get_visible(who, sid)
        if target is None or target.tenant_id != who.tenant_id or target.origin == "tool":
            raise AppError(422, "invalid_supersedes", f"'{sid}' is not a record of your tenant")
    doc = Document.build(
        id=body.id or f"{_ID_PREFIX[body.kind]}-{uuid.uuid4().hex[:8].upper()}",
        tenant_id=who.tenant_id,
        kind=body.kind,
        title=body.title,
        text=body.text,
        updated_at=(body.updated_at or date.today()).isoformat(),
        sensitivity=body.sensitivity,
        supersedes=body.supersedes,
        facts=body.facts,
        origin="api",
        author_role=who.role,
    )
    if doc.scan.flagged and (body.supersedes or body.facts):
        # A record with injected text may be stored and read as data, but may not reshape other records.
        raise AppError(
            422,
            "untrusted_content",
            f"Flagged content ({', '.join(doc.scan.rules)}) cannot set facts or supersede records",
        )
    return doc
