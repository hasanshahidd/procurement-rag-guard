"""Checks the verifier runs on the model's answer before anyone sees it.

All pure functions so each one is easy to test in isolation.
"""

from __future__ import annotations

import re
import unicodedata

from app.guard import matching_rules, normalise, sentence_spans
from app.retrieval import tokenize

# Guard rules that only make sense as text aimed at an AI. Approval or
# disclosure wording is left out because an honest answer may describe the
# attack ("the quote asked to reveal pricing").
ANSWER_RULES = frozenset({"role_marker", "chat_template_token", "persona_switch", "ignore_instructions"})
MIN_SUPPORT = 2  # a cited sentence must share one id or number, or two words, with the answer

_NUM_GROUP = re.compile(r"\d+(?:[\s.,_'  ]+\d+)*")
_PR_ID = re.compile(r"\bPR-\d+\b", re.I)
_APPROVAL_WORD = re.compile(
    r"\b(?:approved|authori[sz]ed|signed[- ]?off|released|green[- ]?lit|cleared|okayed|ok'?d)\b"
    r"|\b(?:go-ahead|green light)\b"
    r"|\b(?:i|we)(?:'ve|'ll| have| will| am| are)?\s+(?:now\s+)?approv(?:e|ing)\b"
    r"|\bapproval\s+(?:is\s+|has been\s+)?(?:granted|given|complete|completed|done)\b",
    re.I,
)
# Wording right before the approval word that turns it into a negation or a
# rule ("has not been approved", "can be signed off by"), not a claim.
_NOT_A_CLAIM_BEFORE = re.compile(
    r"(?:\bnot|\bnever|n't|\byet to be|\bpending|\bawaiting|\bcan be|\bmust be|\bmay be|\bshould be|"
    r"\bneeds? to be|\bto be|\bwill be|\bunless|\bonce|\bif|\bbefore|\buntil|\bafter|"
    r"\brequires?|\brequired|\bneeds?)"
    r"\s+(?:[\w-]+\s+){0,3}$",
    re.I,
)
_IDENTIFIER = re.compile(r"^[a-z][a-z_]{2,39}$")


def ascii_digits(text: str) -> str:
    """Full-width, Arabic-Indic and other decimal digits become 0-9."""
    folded = normalise(text)
    return "".join(str(unicodedata.decimal(ch)) if ch.isdecimal() and not ch.isascii() else ch for ch in folded)


def numbers_in(text: str) -> set[str]:
    """Every way the digits in text can be read: each run, and each group joined.

    "1 045", "1.045", "1,0 45" and "1_045" all yield "1045". Used to build the
    allowlist from the sources and the question.
    """
    found: set[str] = set()
    for m in _NUM_GROUP.finditer(ascii_digits(text)):
        found.add(re.sub(r"\D", "", m.group(0)))
        found.update(re.findall(r"\d+", m.group(0)))
    return found


def _number_candidates(group: str) -> set[str]:
    """Numbers a group of digits in the answer could stand for.

    Every run of 3+ digits counts on its own ("1045 1042" gives both). A group
    made only of short runs ("1 0 4 5", "1,0 45") or written with thousand
    separators ("1,045", "1 045") also counts as the joined number. A short
    trailing decimal part ("1175.00") is not joined.
    """
    runs = re.findall(r"\d+", group)
    candidates = {r for r in runs if len(r) >= 3}
    thousands = len(runs) > 1 and 1 <= len(runs[0]) <= 3 and all(len(r) == 3 for r in runs[1:])
    if thousands or not candidates:
        joined = "".join(runs)
        if len(joined) >= 3:
            candidates.add(joined)
    return candidates


def ungrounded_numbers(answer: str, allowed: set[str]) -> set[str]:
    """Numbers (3+ digits) in the answer that are in no source shown to the model and not in the question.

    An allowlist on purpose: the decision never looks at other tenants' data, so
    whether an answer passes cannot be used to probe hidden values, and it also
    catches figures the model made up.
    """
    found: set[str] = set()
    for m in _NUM_GROUP.finditer(ascii_digits(answer)):
        found |= _number_candidates(m.group(0)) - allowed
    return found


def claims_approval(sentence: str) -> bool:
    return any(not _NOT_A_CLAIM_BEFORE.search(sentence[: m.start()]) for m in _APPROVAL_WORD.finditer(sentence))


def unsupported_approval(answer: str, approved_prs: set[str]) -> bool:
    """True when the answer says something is approved and the system-of-record facts do not back it."""
    for start, end in sentence_spans(answer):
        sentence = normalise(answer[start:end])
        if claims_approval(sentence):
            prs = {p.upper() for p in _PR_ID.findall(sentence)}
            if not prs or not prs <= approved_prs:
                return True
    return False


def instruction_like(answer: str) -> bool:
    return any(ANSWER_RULES & set(matching_rules(answer[s:e])) for s, e in sentence_spans(answer))


def support_score(excerpt: str, answer: str) -> int:
    """How much a cited sentence shares with the answer: ids and numbers count 2, other words 1."""
    shared = set(tokenize(excerpt, parts=False)) & set(tokenize(answer, parts=False))
    return sum(2 if any(ch.isdigit() for ch in tok) else 1 for tok in shared)


def run_output_checks(answer: str, allowed: set[str], approved_prs: set[str]) -> list[str]:
    failures = []
    if ungrounded_numbers(answer, allowed):
        failures.append("ungrounded_number_in_answer")
    if unsupported_approval(answer, approved_prs):
        failures.append("unsupported_approval_claim")
    if instruction_like(answer):
        failures.append("instruction_like_text_in_answer")
    return failures


def describe_extra_field(key: str, value: object) -> str:
    """Summarise a field the model added (for example "actions") without echoing its content.

    Only identifier-like words such as tool names are kept, so a model cannot
    use this field to smuggle a value past the number check.
    """
    safe_key = key if _IDENTIFIER.match(key) else "<field>"
    words: set[str] = set()
    stack = [value]
    while stack and len(words) < 5:
        item = stack.pop()
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, str) and _IDENTIFIER.match(item):
            words.add(item)
    return f"{safe_key}: {', '.join(sorted(words)) or 'content not shown'}"
