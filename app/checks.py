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

_NUM_GROUP = re.compile(r"\d+(?:[\s.,_'/\-  ]+\d+)*")
_PR_ID = re.compile(r"\bPR-\d+\b", re.I)
_ITEM_ID = re.compile(r"\b(?:[A-Z]{3}-\d{2}-[A-Z]{3}|PR-\d+)\b", re.I)  # SKUs and purchase requests
_RECORD_ID = re.compile(r"\b[A-Z]{1,5}(?:-[A-Z]{1,3})?-\d+\b")  # TCK-H-102, KB-001, TOOL-Q-5531, Q-5531
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}")  # also the date part of 2026-10-05T03:12Z
_APPROVAL_WORD = re.compile(
    r"\b(?:approved|authori[sz]ed|signed[- ]?off|released|green[- ]?lit|cleared|okayed|ok'?d)\b"
    r"|\b(?:go-ahead|green light|good to go|(?:may|can) (?:now )?proceed|cleared to proceed)\b"
    r"|\b(?:i|we)(?:'ve|'ll| have| will| am| are)?\s+(?:now\s+)?approv(?:e|ing)\b"
    r"|\bapproval(?:\s+for\s+[\w-]+)?\s*(?::|-|\bis\b|\bwas\b|\bhas been\b|\bhad been\b)?\s*"
    r"(?:granted|given|complete|completed|done|obtained|received|secured)\b"
    r"|\b(?:granted|gave|given|obtained|received|secured|got)\s+(?:(?!before\b|for\b|pending\b)[\w-]+\s+){0,2}approval\b"
    r"|\b(?:purchase|order|request|po|pr-\d+)\s+(?:is\s+|was\s+|has been\s+)?(?:accepted|confirmed)\b",
    re.I,
)
# "Bluegate is an approved supplier" or "the approved policy" describe something else, not a purchase.
_NOT_A_PURCHASE_AFTER = re.compile(
    r"^\s+(?:supplier|vendor|policy|policies|list|budget|catalogue|catalog|device|model|sku|template|form)s?\b", re.I
)
# Text right before an approval word that makes it a non-claim. Each form is
# tied to the word itself, so "After review PR-1042 is approved" or "moved from
# pending to approved" still count as claims.
_NEGATED_BEFORE = re.compile(  # "has not been approved", "never formally signed off"
    r"(?:\bnot|\bnever|n't|\bno|\bno longer|\byet to be|\bawaiting|\bpending|\bwithout)"
    r"(?:\s+(?!(?:to|into|and|but|then|now|is|was|are|were)\b)[\w-]+){0,3}\s+$",
    re.I,
)
_RULE_BEFORE = re.compile(  # "can be signed off by", "needs to be approved", "requires approved quotes"
    r"(?:\b(?:can|must|may|might|should|will|would|could|shall)(?:\s+(?:only|first|still|also|not|never|then))?\s+be"
    r"|(?<!happens )(?<!happened )(?<!seems )(?<!appears )(?<!out )(?<!proved )\bto\s+be"
    r"|\brequires?|\brequired)\s+(?:\w+ly\s+)?$",
    re.I,
)
_CONDITION_BEFORE = re.compile(  # "once approved", "before a purchase is approved", "until it is approved"
    r"\b(?:once|if|unless|until|before|whether)\s+"
    r"(?:(?:it|they|this|that|pr-\d+|(?:the|a|an)\s+[\w-]+(?:\s+[\w-]+)?|purchase|request|order)\s+)?"
    # present or future only: "before it was approved" reports a past approval, so it stays a claim
    r"(?:(?:is|are|gets?|has|have)(?:\s+been)?|(?:can|will|would|must)\s+be)?\s*$",
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
    """Default deny: any approval wording is a claim unless it is negated, a rule or a condition."""
    for m in _APPROVAL_WORD.finditer(sentence):
        before, after = sentence[: m.start()], sentence[m.end() :]
        if _NOT_A_PURCHASE_AFTER.match(after) or re.search(r"\b(?:not|never|no)\b|n't", m.group(0), re.I):
            continue
        if not any(rule.search(before) for rule in (_NEGATED_BEFORE, _RULE_BEFORE, _CONDITION_BEFORE)):
            return True
    return False


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


def contradicting_figures(excerpt: str, answer: str) -> bool:
    """True when the cited sentence states amounts and the answer states amounts, and none match.

    "Price is 1,180 AUD" cannot be the support for an answer of "990 AUD". Ids
    (PR-2210, LAP-14-STD, TCK-H-102) and dates are not amounts, so they are left out.
    """
    _, cited = _item_figures(excerpt)
    _, claimed = _item_figures(answer)
    return bool(cited and claimed and not cited & claimed)


def _item_figures(sentence: str) -> tuple[set[str], set[str]]:
    """SKU and PR ids in a sentence, and its figures (3+ digits) outside ids and dates."""
    text = ascii_digits(sentence)
    ids = {i.upper() for i in _ITEM_ID.findall(text)}
    rest = _DATE.sub(" ", _RECORD_ID.sub(" ", _ITEM_ID.sub(" ", text)))
    return ids, {n for m in _NUM_GROUP.finditer(rest) for n in _number_candidates(m.group(0))}


def _same_item(a: set[str], b: set[str]) -> bool:
    """False only when both name items of the same kind (SKU or PR) and share none."""
    for is_pr in (True, False):
        x = {i for i in a if i.startswith("PR-") == is_pr}
        y = {i for i in b if i.startswith("PR-") == is_pr}
        if x and y and not x & y:
            return False
    return True


def misattributed_figures(answer: str, source_texts: list[str]) -> bool:
    """True when the answer ties a figure to an item, but every source sentence with that figure is about other items.

    "TAB-10-RUG costs 1,180 AUD" fails when 1,180 only appears in the LAP-14-STD
    sentence. A figure in a sentence that names no item (a policy threshold) is fine.
    """
    held = [_item_figures(text[s:e]) for text in source_texts for s, e in sentence_spans(text)]
    for start, end in sentence_spans(answer):
        ids, figures = _item_figures(answer[start:end])
        for figure in figures if ids else ():
            holders = [held_ids for held_ids, nums in held if figure in nums]
            if holders and not any(_same_item(ids, held_ids) for held_ids in holders):
                return True
    return False


_QUESTION_WORDS = frozenset(
    "what when where which who whom whose why how is are was were has have had does do did can could should would "
    "will shall may might any our the a an please tell show list give i we "
    # calendar words are capitalised but are not names of anything in the records
    "january february march april may june july august september october november december "
    "monday tuesday wednesday thursday friday saturday sunday".split()
)
_PROPER_NAME = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")


def question_names(question: str) -> list[str]:
    """Capitalised names in the question (companies, vendors, sites), minus question words.

    The first word of a sentence is capitalised anyway ("Explain why...", "Summarise..."), so it is not counted.
    """
    names = []
    for m in _PROPER_NAME.finditer(question):
        words = m.group(0).split()
        lead = question[: m.start()].rstrip()
        if not lead or lead[-1] in ".!?:;":
            words = words[1:]
        words = [w for w in words if w.lower() not in _QUESTION_WORDS]
        if words:
            names.append(" ".join(words))
    return list(dict.fromkeys(names))


def unmentioned_names(question: str, shown_text: str) -> list[str]:
    """Names from the question that no source shown to the model mentions.

    Uses only the caller's own sources, so the result is the same for a real
    other customer and for a made-up company: it cannot reveal who the
    customers are.
    """
    haystack = shown_text.lower()
    return [n for n in question_names(question) if not all(w.lower() in haystack for w in n.split())]


def run_output_checks(
    answer: str,
    allowed: set[str],
    approved_prs: set[str],
    trusted: set[str] | None = None,
    source_texts: list[str] | None = None,
) -> list[str]:
    """allowed: numbers in the cited sources and the question. trusted: the subset backed by curated records.

    source_texts: what the model was shown of the backing sources, for the figure-to-item check.
    """
    failures = []
    if ungrounded_numbers(answer, allowed):
        failures.append("ungrounded_number_in_answer")
    elif trusted is not None and ungrounded_numbers(answer, trusted):
        failures.append("figure_only_in_unverified_source")
    elif source_texts and misattributed_figures(answer, source_texts):
        failures.append("figure_attributed_to_wrong_item")
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
