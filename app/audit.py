"""Append-only JSONL audit trail, one line per /ask request.

If the line cannot be written the request fails: the service does not hand
out answers it cannot account for.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

from app.schemas import AskResponse
from app.store import Principal


class AuditLog:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def write(self, principal: Principal, question: str, response: AskResponse) -> None:
        record = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "trace_id": response.trace_id,
            "tenant_id": principal.tenant_id,
            "role": principal.role,
            "question": question,
            "status": response.status,
            "reason": response.reason,
            "model": response.model,
            "retrieved": [r.model_dump() for r in response.retrieved],
            "excluded": [e.model_dump() for e in response.excluded_sources],
            "citations": [
                {"source_id": c.source_id, "sha256": c.content_sha256, "start": c.start, "end": c.end}
                for c in response.citations
            ],
            "security": response.security.model_dump(),
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
