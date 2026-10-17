"""Heading-aware chunking with stable IDs.

* Chunks never cross a section boundary, so a chunk is about one thing.
* Long sections are split on sentence boundaries into windows of at most `max_words`, with
  `overlap_words` carried over so a fact split across a boundary is still retrievable.
* Each chunk is prefixed with its document title and heading trail ("contextual header"). On its
  own, "The late charge is the lesser of $25 or 5%" doesn't say which state it's about.
* The chunk ID hashes doc_id + heading trail + text, so re-processing an unchanged document
  produces the same IDs, and unchanged chunks are never re-embedded.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .text import Section


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    ordinal: int
    heading: str
    text: str  # what gets embedded and indexed (with the contextual header)
    content_hash: str  # of `text` only, so identical chunks share one embedding

    @property
    def words(self) -> int:
        return len(self.text.split())


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def chunk_document(doc_id: str, title: str, secs: list[Section], max_words: int = 120, overlap_words: int = 25) -> list[Chunk]:
    chunks: list[Chunk] = []
    for sec in secs:
        trail = " > ".join(sec.path) if sec.path else title
        header = trail if trail.startswith(title) else f"{title} > {trail}"
        windows: list[list[str]] = [[]]
        for sentence in _sentences(sec.text):
            cur = windows[-1]
            if cur and len(" ".join(cur + [sentence]).split()) > max_words:
                carry: list[str] = []
                for s in reversed(cur):  # overlap: carry trailing sentences up to overlap_words
                    if len(" ".join([s] + carry).split()) > overlap_words:
                        break
                    carry.insert(0, s)
                windows.append(carry + [sentence])
            else:
                cur.append(sentence)
        for w in windows:
            if not w:
                continue
            body = " ".join(w)
            text = f"{header}\n{body}"
            content_hash = hashlib.sha256(text.encode()).hexdigest()
            chunk_id = hashlib.sha256(f"{doc_id}|{trail}|{body}".encode()).hexdigest()[:20]
            chunks.append(Chunk(chunk_id, doc_id, len(chunks), trail, text, content_hash))
    return chunks
