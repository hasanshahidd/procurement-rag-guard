"""FastAPI entry point. Run with: uvicorn app.main:create_app --factory --port 8000

An app factory, not a module-level app, so importing this module (tests, tools)
never reads settings, opens a database or builds model clients.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.audit import AuditLog
from app.config import Settings
from app.embeddings import Embedder, OllamaEmbedder
from app.errors import install_error_handlers
from app.llm import LLM, build_llm
from app.pipeline import Pipeline
from app.routes import RateLimiter, Services, router
from app.schemas import ErrorResponse
from app.store import Store

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(
    settings: Settings | None = None,
    llm: LLM | None = None,
    store: Store | None = None,
    embedder: Embedder | None = None,
) -> FastAPI:
    """Build the app. Tests pass fakes for the model, store and embedder; production reads the environment."""
    settings = settings or Settings.from_env()
    store = store or Store.from_dir(settings.data_dir, db_path=settings.db_path)
    llm = llm or build_llm(settings)
    if embedder is None and settings.retrieval_mode == "hybrid":
        embedder = OllamaEmbedder(settings.ollama_url, settings.embed_model)
    audit = AuditLog(settings.audit_path)
    svc = Services(
        settings=settings,
        store=store,
        llm=llm,
        embedder=embedder,
        audit=audit,
        pipeline=Pipeline(store, llm, settings, audit, embedder),
        limiter=RateLimiter(settings.rate_limit_per_min),
    )

    app = FastAPI(
        title="Procurement RAG Guard",
        version=__version__,
        description="Tenant-scoped procurement Q&A with tool-output prompt injection containment.",
        responses={422: {"model": ErrorResponse}, 401: {"model": ErrorResponse}},
    )
    app.state.services = svc
    app.state.store, app.state.llm = store, llm  # direct handles for tests and tools
    app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
    app.include_router(router)
    install_error_handlers(app)
    return app
