"""Parsing, cleaning, redaction and tokenization."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

STOPWORDS = frozenset(
    "a an the and or of to in on for is are be by at as with from that this it its can may must our we us "
    "how what when where which who do does did after before than then there their they them".split()
)
_TOKEN = re.compile(r"\$?\d+(?:\.\d+)?%?|[a-z]+(?:'[a-z]+)?")

# Redact before anything is embedded or indexed: vectors and indexes are hard to scrub later.
PII_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"\b\d{10,16}\b"), "[ACCOUNT]"),
    (re.compile(r"\b\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b"), "[PHONE]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "[EMAIL]"),
]


def redact(text: str) -> tuple[str, int]:
    hits = 0
    for pattern, label in PII_PATTERNS:
        text, n = pattern.subn(label, text)
        hits += n
    return text, hits


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


@dataclass
class Section:
    path: tuple[str, ...]  # heading trail, e.g. ("Late Payments in Ohio", "Grace period")
    text: str


def sections(markdown: str) -> tuple[str, list[Section]]:
    """Split Markdown into sections by heading. Returns (title, sections)."""
    title, trail, out, buf = "", [], [], []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            out.append(Section(tuple(trail), body))
        buf.clear()

    for line in markdown.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            flush()
            level, heading = len(m.group(1)), m.group(2).strip()
            if level == 1 and not title:
                title = heading
            trail[:] = trail[: level - 1] + [heading]
        elif line.strip() == "---":
            flush()
        else:
            buf.append(line)
    flush()
    return title, out


def boilerplate(documents: list[str], min_share: float = 0.5) -> set[str]:
    """Paragraphs that appear in at least `min_share` of documents (footers, disclaimers).

    They add nothing to retrieval but make every chunk look alike, so they're removed before chunking.
    """
    counts: Counter[str] = Counter()
    for doc in documents:
        counts.update({p.strip() for p in re.split(r"\n\s*\n", doc) if p.strip()})
    return {p for p, c in counts.items() if c / max(len(documents), 1) >= min_share and not p.startswith("#")}


def strip_boilerplate(doc: str, paragraphs: set[str]) -> str:
    parts = [p for p in re.split(r"\n\s*\n", doc) if p.strip() and p.strip() not in paragraphs]
    return "\n\n".join(parts)
