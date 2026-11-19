"""In-memory document store with tenant and role based visibility.

Access control lives here, in code. The model only ever receives documents
returned by Store.visible_docs, so a prompt cannot widen what it sees.
Documents are keyed by (tenant_id, id): one tenant can never collide with,
shadow or probe another tenant's ids.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

from app.guard import ScanResult, scan

SHARED_TENANT = "shared"
ROLE_CLEARANCE: dict[str, frozenset[str]] = {
    "requester": frozenset({"public", "internal"}),
    "buyer": frozenset({"public", "internal", "restricted"}),
    "approver": frozenset({"public", "internal", "restricted"}),
}
_AMOUNT = re.compile(r"\b\d{1,3}(?:,\d{3})+\b|\b\d{3,}\b")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def amounts_in(text: str) -> set[str]:
    """Numbers of three or more digits in stored text, commas removed."""
    return {m.group(0).replace(",", "") for m in _AMOUNT.finditer(text)}


@dataclass(frozen=True)
class Principal:
    tenant_id: str
    role: str
    label: str

    @property
    def clearance(self) -> frozenset[str]:
        return ROLE_CLEARANCE.get(self.role, frozenset())


@dataclass(frozen=True)
class Document:
    id: str
    tenant_id: str
    kind: str
    title: str
    text: str
    updated_at: str
    sensitivity: str
    supersedes: tuple[str, ...] = ()
    facts: tuple[tuple[str, str], ...] = ()
    scan: ScanResult = field(default=None, compare=False)  # type: ignore[assignment]
    content_sha256: str = ""

    @classmethod
    def build(cls, **kw: object) -> Document:
        text = str(kw["text"])
        kw["supersedes"] = tuple(kw.get("supersedes") or ())  # type: ignore[arg-type]
        kw["facts"] = tuple(sorted(dict(kw.get("facts") or {}).items()))  # type: ignore[arg-type]
        return cls(**kw, scan=scan(text), content_sha256=sha256(text))  # type: ignore[arg-type]


class DuplicateDocumentError(Exception):
    pass


class Store:
    def __init__(self, documents: list[Document], principals: dict[str, Principal], quotes: list[dict]):
        self._docs: dict[tuple[str, str], Document] = {(d.tenant_id, d.id): d for d in documents}
        self._tool_outputs: dict[tuple[str, str], Document] = {}  # for citation lookup, never indexed
        self._principals = principals
        self._quotes = quotes
        self._lock = threading.Lock()

    @classmethod
    def from_dir(cls, data_dir: Path) -> Store:
        def load(name: str) -> list[dict]:
            return json.loads((data_dir / name).read_text(encoding="utf-8"))

        docs = [Document.build(**d) for d in load("documents.json")]
        principals = {p["api_key"]: Principal(p["tenant_id"], p["role"], p["label"]) for p in load("principals.json")}
        return cls(docs, principals, load("vendor_quotes.json"))

    def __len__(self) -> int:
        return len(self._docs)

    def _snapshot(self) -> list[Document]:
        with self._lock:  # sync endpoints run in a threadpool while /tickets writes
            return list(self._docs.values())

    def authenticate(self, api_key: str | None) -> Principal | None:
        if not api_key:
            return None
        for key, principal in self._principals.items():
            if hmac.compare_digest(key.encode(), api_key.encode()):
                return principal
        return None

    def can_see(self, principal: Principal, doc: Document) -> bool:
        return doc.tenant_id in (principal.tenant_id, SHARED_TENANT) and doc.sensitivity in principal.clearance

    def visible_docs(self, principal: Principal) -> list[Document]:
        return [d for d in self._snapshot() if self.can_see(principal, d)]

    def get_visible(self, principal: Principal, doc_id: str) -> Document | None:
        with self._lock:
            candidates = [
                self._docs.get((principal.tenant_id, doc_id)),
                self._docs.get((SHARED_TENANT, doc_id)),
                self._tool_outputs.get((principal.tenant_id, doc_id)),
            ]
        return next((d for d in candidates if d and self.can_see(principal, d)), None)

    def add(self, doc: Document) -> Document:
        with self._lock:
            # Ids must be unique within what this tenant can see, so citations stay unambiguous.
            if (doc.tenant_id, doc.id) in self._docs or (SHARED_TENANT, doc.id) in self._docs:
                raise DuplicateDocumentError(doc.id)
            self._docs[(doc.tenant_id, doc.id)] = doc
        return doc

    def remember_tool_output(self, doc: Document) -> None:
        with self._lock:
            self._tool_outputs[(doc.tenant_id, doc.id)] = doc

    def quotes_for(self, principal: Principal, sku: str) -> list[dict]:
        return [q for q in self._quotes if q["tenant_id"] == principal.tenant_id and q["sku"] == sku]

    def _quote_visible(self, principal: Principal, quote: dict) -> bool:
        return quote["tenant_id"] == principal.tenant_id and "restricted" in principal.clearance

    def hidden_amounts(self, principal: Principal) -> set[str]:
        """Numbers that appear only in records this principal may not see.

        Numbers that also appear somewhere the caller can see (years, common
        quantities) are left out, so they do not cause false blocks.
        """
        hidden: set[str] = set()
        seen: set[str] = set()
        for doc in self._snapshot():
            (seen if self.can_see(principal, doc) else hidden).update(amounts_in(doc.text))
        for quote in self._quotes:
            (seen if self._quote_visible(principal, quote) else hidden).update(amounts_in(quote["text"]))
        return hidden - seen
