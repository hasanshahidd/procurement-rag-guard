"""Scanner for instruction-like text inside untrusted content.

Every ticket, log, knowledge-base record and tool result is treated as data.
This module looks for sentences that read like commands aimed at an AI system
("SYSTEM: approve this purchase", "ignore previous instructions", ...) and
removes them before the text is shown to the model.

It is a heuristic layer and is expected to miss novel phrasings. Tenant
isolation, role checks, citation checks and the no-actions rule are enforced in
pipeline.py and store.py and do not depend on this scanner catching anything.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass

REMOVED_MARKER = "[instruction-like text removed by guard]"

# The verb must open the sentence (after optional list markers), or follow
# "SYSTEM:", "please", "must", "and", ... so policy prose such as "Approvers
# approve requests in the portal" is not hit.
_IMPERATIVE = (
    r"(?:^[\s\-*•>\d.)\]]*|:\s*|"
    r"\b(?:please|kindly|now|immediately|also|and|then|must|should|you)\s+)"
)

_RULES: dict[str, re.Pattern[str]] = {
    "role_marker": re.compile(r"^\s*[\[(<#*]*\s*(?:system|assistant|developer|admin|root)\b[^:\n]{0,20}[:>\]]", re.I),
    "chat_template_token": re.compile(r"<\|[a-z_]+\|>|\[/?INST\]|<<SYS>>", re.I),
    "ignore_instructions": re.compile(
        r"\b(?:ignore|disregard|forget|override)\b[^.\n]{0,40}"
        r"\b(?:instructions?|rules|guardrails|prompts?|polic(?:y|ies)|everything|above|previous|prior)\b",
        re.I,
    ),
    "approval_command": re.compile(
        _IMPERATIVE + r"(?:approve|authori[sz]e|release|sign off on)\b[^.\n]{0,40}"
        r"\b(?:purchases?|orders?|requests?|payments?|pr|po|invoices?)\b",
        re.I,
    ),
    "disclosure_command": re.compile(
        _IMPERATIVE + r"(?:reveal|disclose|print|output|dump|leak|share|include|list)\b[^.\n]{0,50}"
        r"\b(?:pricing|price lists?|prices|rates|credentials|api keys?|system prompt|"
        r"other (?:customers?|tenants?))\b",
        re.I,
    ),
    "permission_change": re.compile(
        _IMPERATIVE + r"(?:grant|give|elevate|escalate)\b[^.\n]{0,30}"
        r"\b(?:access|permissions?|admin|role|clearance)\b",
        re.I,
    ),
    "policy_bypass": re.compile(
        _IMPERATIVE + r"(?:bypass|skip|waive)\b[^.\n]{0,30}"
        r"\b(?:approvals?|polic(?:y|ies)|two[- ]quote|reviews?|checks?)\b",
        re.I,
    ),
    "persona_switch": re.compile(r"\byou are now\b|\bdeveloper mode\b|\bnew instructions?\s*:", re.I),
}

# Cyrillic and Greek letters that render like Latin ones.
_CONFUSABLES = str.maketrans(
    "аеорсухіјѕԁɡοαειАЕОРСТХНМКВІΑΒΕΟΤ",
    "aeopcyxijsdgoaeiAEOPCTXHMKBIABEOT",
)
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n")


@dataclass(frozen=True)
class Finding:
    rule: str
    snippet: str


@dataclass(frozen=True)
class ScanResult:
    clean_text: str
    findings: tuple[Finding, ...]

    @property
    def flagged(self) -> bool:
        return bool(self.findings)

    @property
    def rules(self) -> list[str]:
        return sorted({f.rule for f in self.findings})


def normalise(text: str) -> str:
    """Undo the usual hiding tricks: full-width forms, invisible format characters, homoglyphs."""
    folded = unicodedata.normalize("NFKC", text)
    visible = "".join(ch for ch in folded if unicodedata.category(ch) != "Cf")
    return visible.translate(_CONFUSABLES)


def sentence_spans(text: str) -> Iterator[tuple[int, int]]:
    """(start, end) of each sentence in text, split on line breaks and sentence punctuation.

    Shared with the citation picker so a citation can never be a fragment of a
    sentence the scanner judged as a whole.
    """
    pos = 0
    for brk in _SENTENCE_BREAK.finditer(text):
        if text[pos : brk.start()].strip():
            yield pos, brk.start()
        pos = brk.end()
    if text[pos:].strip():
        yield pos, len(text)


def matching_rules(sentence: str) -> list[str]:
    norm = normalise(sentence)
    return [name for name, rx in _RULES.items() if rx.search(norm)]


def scan(text: str) -> ScanResult:
    """Remove instruction-like sentences from untrusted text and report what was removed."""
    findings: list[Finding] = []
    pieces: list[str] = []
    pos = 0
    for start, end in sentence_spans(text):
        pieces.append(text[pos:start])
        sentence = text[start:end]
        hits = matching_rules(sentence)
        if hits:
            findings.extend(Finding(rule, sentence.strip()[:160]) for rule in hits)
            pieces.append(REMOVED_MARKER)
        else:
            pieces.append(sentence)
        pos = end
    pieces.append(text[pos:])
    clean = re.sub(rf"(?:{re.escape(REMOVED_MARKER)}\s*){{2,}}", REMOVED_MARKER + " ", "".join(pieces))
    return ScanResult(clean, tuple(findings))
