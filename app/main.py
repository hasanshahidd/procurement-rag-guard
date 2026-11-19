"""FastAPI entry point. Run with: uvicorn app.main:create_app --factory --port 8000

An app factory, not a module-level app, so importing this module (tests, tools)
never reads settings, opens a database or builds model clients.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from datetime import date
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import __version__
from app.audit import AuditLog
from app.config import Settings
from app.embeddings import Embedder, OllamaEmbedder
from app.llm import LLM, LLMUnavailableError, build_llm
from app.pipeline import Pipeline
from app.schemas import (
    AskRequest,
    AskResponse,
    ErrorResponse,
    FlaggedSpan,
    HealthResponse,
    SourceView,
    TicketCreate,
    TicketCreated,
)
from app.store import CURATOR_ROLES, Document, DuplicateDocumentError, Principal, Store

log = logging.getLogger("procurement")
STATIC_DIR = Path(__file__).resolve().parent / "static"
_api_key = APIKeyHeader(
    name="X-API-Key", auto_error=False, description="Synthetic demo key, see data/seed/principals.json"
)
_ID_PREFIX = {"ticket": "TCK", "log": "LOG", "kb": "KB"}
_ERRORS = {
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    429: "rate_limited",
}


class AppError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def _error(status: int, code: str, message: str, details: object = None) -> JSONResponse:
    body = {"error": {"code": code, "message": message, "details": details}}
    return JSONResponse(status_code=status, content=body)


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


def create_app(
    settings: Settings | None = None,
    llm: LLM | None = None,
    store: Store | None = None,
    embedder: Embedder | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    store = store or Store.from_dir(settings.data_dir, db_path=settings.db_path)
    llm = llm or build_llm(settings)
    if embedder is None and settings.retrieval_mode == "hybrid":
        embedder = OllamaEmbedder(settings.ollama_url, settings.embed_model)
    audit = AuditLog(settings.audit_path)
    pipeline = Pipeline(store, llm, settings, audit, embedder)
    limiter = RateLimiter(settings.rate_limit_per_min)

    app = FastAPI(
        title="Procurement RAG Guard",
        version=__version__,
        description="Tenant-scoped procurement Q&A with tool-output prompt injection containment.",
        responses={422: {"model": ErrorResponse}, 401: {"model": ErrorResponse}},
    )
    app.state.store, app.state.llm = store, llm

    def principal(request: Request, api_key: str | None = Depends(_api_key)) -> Principal:
        found = store.authenticate(api_key)
        if found is None:
            # Failed attempts are limited too, per client address, so keys cannot be guessed at full speed.
            limiter.check(f"anonymous:{request.client.host if request.client else 'unknown'}")
            raise AppError(401, "unauthorized", "Missing or unknown X-API-Key header")
        limiter.check(f"{found.tenant_id}:{found.role}")  # per principal, not per path, so ids cannot dodge it
        return found

    app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/ui/")

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health() -> HealthResponse:
        llm_state = {"backend": settings.llm_backend, "model": llm.name, **llm.health()}
        retrieval = {"mode": "bm25" if embedder is None else "hybrid"}
        if embedder is not None:
            retrieval.update(embedding_model=embedder.name, **embedder.health())
        return HealthResponse(
            status="ok" if llm_state["reachable"] else "degraded",
            version=__version__,
            documents=len(store),
            llm=llm_state,
            retrieval=retrieval,
        )

    @app.post("/tickets", response_model=TicketCreated, status_code=201, tags=["data"])
    @app.post("/ingest", response_model=TicketCreated, status_code=201, tags=["data"], include_in_schema=False)
    def create_ticket(body: TicketCreate, who: Principal = Depends(principal)) -> TicketCreated:
        doc = store.add(_build_document(store, who, body))
        audit.write_ingest(who, doc)
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

    @app.post("/ask", response_model=AskResponse, tags=["assistant"])
    def ask(body: AskRequest, who: Principal = Depends(principal)) -> AskResponse:
        # Sync on purpose: model calls block, so FastAPI runs this in its threadpool.
        response = pipeline.ask(who, body.question, body.top_k)
        if response.status == "blocked" or response.security.blocked_actions:
            log.warning("trace %s blocked for %s: %s", response.trace_id, who.tenant_id, response.reason)
        return response

    @app.get("/sources/{source_id}", response_model=SourceView, tags=["assistant"])
    def get_source(source_id: str, who: Principal = Depends(principal)) -> SourceView:
        doc = store.get_visible(who, source_id)
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

    _install_error_handlers(app)
    return app


def _build_document(store: Store, who: Principal, body: TicketCreate) -> Document:
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
        if target is None or target.tenant_id != who.tenant_id:
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


def _install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        return _error(exc.status, exc.code, exc.message)

    @app.exception_handler(DuplicateDocumentError)
    async def _duplicate(_: Request, exc: DuplicateDocumentError) -> JSONResponse:
        return _error(409, "duplicate_id", f"A record with id '{exc}' already exists")

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        return _error(422, "validation_error", "Request body or parameters are invalid", details)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _error(exc.status_code, _ERRORS.get(exc.status_code, "http_error"), str(exc.detail))

    @app.exception_handler(LLMUnavailableError)
    async def _llm_down(_: Request, exc: LLMUnavailableError) -> JSONResponse:
        log.error("LLM unavailable: %s", exc)
        return _error(503, "llm_unavailable", str(exc))

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error", exc_info=exc)
        return _error(500, "internal_error", "Unexpected server error. See server logs.")
