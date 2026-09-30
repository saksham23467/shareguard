"""Pattern-based secret / PII detectors (CPU).

Every detector is deterministic and unit-tested. Confusable-tolerant matching
normalises common OCR swaps (O->0, l/I->1, S->5, B->8) inside digit-like tokens
before checksum validation, so an OCR misread does not turn a secret into "safe".
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

# --- checksums ---------------------------------------------------------------

_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def verhoeff_valid(digits: str) -> bool:
    if not digits.isdigit():
        return False
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
    return c == 0


def luhn_valid(digits: str) -> bool:
    if not digits.isdigit() or len(digits) < 12:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# --- helpers -----------------------------------------------------------------

_CONFUSABLE_TO_DIGIT = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0",
                                       "I": "1", "l": "1", "i": "1", "|": "1",
                                       "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2",
                                       "G": "6", "b": "6", "g": "9", "q": "9"})


def to_digits(token: str) -> str:
    """Map OCR-confusable letters to digits and drop separators."""
    t = token.translate(_CONFUSABLE_TO_DIGIT)
    return re.sub(r"[\s\-./]", "", t)


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


@dataclass(frozen=True)
class Finding:
    kind: str
    text: str


# Candidate spans: runs of digit-ish characters with optional separators.
_DIGITISH = r"[0-9OoDQIl|iSsBZzGbgq]"
_RE_12 = re.compile(rf"(?<![0-9A-Za-z]){_DIGITISH}{{4}}[\s\-]?{_DIGITISH}{{4}}[\s\-]?{_DIGITISH}{{4}}(?![0-9A-Za-z])")
_RE_CARD = re.compile(rf"(?<![0-9A-Za-z])(?:{_DIGITISH}[\s\-]?){{12,18}}{_DIGITISH}(?![0-9A-Za-z])")
# PAN: 5 letters, 4 digits, 1 letter; 4th letter encodes holder type.
_RE_PAN = re.compile(r"(?<![A-Za-z0-9])[A-Z]{3}[ABCFGHLJPTK][A-Z][0-9OIlSB]{4}[A-Z](?![A-Za-z0-9])")
_RE_AWS = re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])")
_RE_AWS_PARTIAL = re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{0,15}(?![A-Z0-9])")
_RE_KEYLIKE = re.compile(r"(?:sk|pk|ghp|gho|xox[abp]|glpat|AIza)[-_A-Za-z0-9]{16,}")
_RE_ASSIGN = re.compile(r"(?i)(?:password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key)\s*[:=]\s*(\S{6,})")
_RE_TOKEN = re.compile(r"[A-Za-z0-9+/_\-=]{20,}")
_RE_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_RE_PHONE_IN = re.compile(r"(?<![0-9])(?:\+?91[\s\-]?)?[6-9][0-9]{4}[\s\-]?[0-9]{5}(?![0-9])")

LABELS = re.compile(r"(?i)\b(name|address|dob|date of birth|account\s*(no|number)|ifsc|aadhaar|pan|phone|mobile|email|password)\b\s*[:\-]?")


def detect(text: str, entropy_bits: float = 3.5, min_token_len: int = 20) -> list[Finding]:
    """Return every sensitive finding in one OCR line."""
    out: list[Finding] = []
    for m in _RE_12.finditer(text):
        d = to_digits(m.group(0))
        if len(d) == 12 and d[0] in "23456789" and verhoeff_valid(d):
            out.append(Finding("aadhaar", m.group(0)))
    for m in _RE_CARD.finditer(text):
        d = to_digits(m.group(0))
        if 13 <= len(d) <= 19 and luhn_valid(d):
            out.append(Finding("card", m.group(0)))
    for rx, kind in ((_RE_PAN, "pan"), (_RE_AWS, "aws_key"), (_RE_KEYLIKE, "api_key"),
                     (_RE_EMAIL, "email"), (_RE_PHONE_IN, "phone")):
        for m in rx.finditer(text):
            out.append(Finding(kind, m.group(0)))
    for m in _RE_AWS_PARTIAL.finditer(text):
        if not any(f.kind == "aws_key" for f in out):
            out.append(Finding("aws_key_partial", m.group(0)))
    for m in _RE_ASSIGN.finditer(text):
        out.append(Finding("assigned_secret", m.group(1)))
    for m in _RE_TOKEN.finditer(text):
        tok = m.group(0)
        if len(tok) >= min_token_len and shannon_entropy(tok) >= entropy_bits \
                and any(c.isdigit() for c in tok) and any(c.isalpha() for c in tok):
            out.append(Finding("high_entropy", tok))
    return out


# OCR often drops or spaces underscores ("AWS ACCESS KEY ID"), so separators are optional.
_KEY_LABELS = re.compile(r"(?i)(access[\s_\-]?key|secret|token|api[\s_\-]?key|passw)")


def has_field_label(text: str) -> bool:
    """Layout rule: a line containing a PII/secret field label. The value next to
    it stays blurred even while it is only partially typed (no pattern matches yet)."""
    return bool(LABELS.search(text) or _KEY_LABELS.search(text))
