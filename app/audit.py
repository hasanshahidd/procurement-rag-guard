"""Append-only JSONL audit trail: one line per /ask and per ingested record.

If a line cannot be written the request fails: the service does not hand out
answers, or accept records, it cannot account for. API keys are never logged,
and raw model replies are logged only as a hash, so a reply that leaked
something is traceable without the log becoming a second copy of the leak.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

from app.schemas import AskResponse
from app.store import Document, Principal


class AuditLog:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def _append(self, record: dict) -> None:
        line = json.dumps({"ts": datetime.now(UTC).isoformat(timespec="seconds"), **record}, ensure_ascii=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def write(
        self, principal: Principal, question: str, response: AskResponse, model_output_sha256: str | None = None
    ) -> None:
        self._append(
            {
                "event": "ask",
                "trace_id": response.trace_id,
                "tenant_id": principal.tenant_id,
                "role": principal.role,
                "question": question,
                "status": response.status,
                "reason": response.reason,
                "model": response.model,
                "model_output_sha256": model_output_sha256,
                "retrieved": [r.model_dump() for r in response.retrieved],
                "excluded": [e.model_dump() for e in response.excluded_sources],
                "citations": [
                    {"source_id": c.source_id, "sha256": c.content_sha256, "start": c.start, "end": c.end}
                    for c in response.citations
                ],
                "security": response.security.model_dump(),
            }
        )

    def write_ingest(self, principal: Principal, doc: Document) -> None:
        self._append(
            {
                "event": "ingest",
                "tenant_id": principal.tenant_id,
                "role": principal.role,
                "source_id": doc.id,
                "kind": doc.kind,
                "sensitivity": doc.sensitivity,
                "content_sha256": doc.content_sha256,
                "injection_flags": doc.scan.rules,
                "supersedes": list(doc.supersedes),
            }
        )
