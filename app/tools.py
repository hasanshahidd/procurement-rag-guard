"""Procurement tools available to the tool-runner agent.

Only read-only tools are registered. There is deliberately no approve, reject
or edit tool: approval happens in the purchasing portal by a named human, so a
prompt injection has nothing to call even if the model is fully compromised.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.store import Document, Principal, Store


class ToolPermissionError(PermissionError):
    pass


@dataclass(frozen=True)
class ToolSpec:
    name: str
    read_only: bool
    required_clearance: str
    run: Callable[[Store, Principal, str], list[Document]]


def _vendor_quote(store: Store, principal: Principal, sku: str) -> list[Document]:
    return [
        Document.build(
            id=f"TOOL-{q['quote_id']}",
            tenant_id=principal.tenant_id,
            kind="tool_output",
            title=f"vendor_quote(sku={sku}) from {q['vendor']}",
            text=q["text"],
            updated_at=q["issued_at"],
            sensitivity="restricted",
            origin="tool",
        )
        for q in store.quotes_for(principal, sku)
    ]


TOOLS: dict[str, ToolSpec] = {
    "vendor_quote": ToolSpec("vendor_quote", True, "restricted", _vendor_quote),
}


def run_tool(name: str, store: Store, principal: Principal, arg: str) -> list[Document]:
    spec = TOOLS[name]
    if spec.required_clearance not in principal.clearance:
        raise ToolPermissionError(f"{principal.role} may not call {name}")
    results = spec.run(store, principal, arg)
    for doc in results:
        store.remember_tool_output(doc)
    return results
