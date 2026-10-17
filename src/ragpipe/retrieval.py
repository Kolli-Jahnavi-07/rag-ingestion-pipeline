"""Retrieval: vector, BM25 and hybrid (reciprocal rank fusion), with metadata filters.

BM25 is computed in SQL over the postings table, so keyword search lives in the same store as
the vectors and needs no extra service.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .corpus import STATES
from .embeddings import Embedder
from .store import Store
from .text import tokenize

K1, B, RRF_K = 1.2, 0.75, 60
_STATE_NAMES = {name.lower(): code for code, name in STATES.items()}


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    heading: str
    text: str
    score: float


def detect_state(query: str) -> str | None:
    """Query understanding: 'a customer in North Carolina' or 'NC customer' -> NC, used as a metadata filter."""
    q = query.lower()
    for name in sorted(_STATE_NAMES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(name)}\b", q):
            return _STATE_NAMES[name]
    for code in re.findall(r"\b[A-Z]{2}\b", query):  # postal abbreviations, upper case only ("IN" vs "in")
        if code in STATES:
            return code
    return None


def _filter_sql(state: str | None) -> tuple[str, list]:
    if not state:
        return "", []
    # Keep state-specific docs for that state plus documents that apply everywhere (FAQs).
    return " AND (d.state = ? OR d.state IS NULL)", [state]


def vector_search(store: Store, embedder: Embedder, query: str, k: int = 10, state: str | None = None) -> list[Hit]:
    qv = embedder.embed([query])[0]
    flt, params = _filter_sql(state)
    rows = store.sql(
        f"""
        SELECT c.chunk_id, c.doc_id, c.heading, c.text,
               array_cosine_similarity(e.vector, ?::FLOAT[{store.dim}]) AS score
        FROM chunks c
        JOIN embeddings e ON e.content_hash = c.content_hash AND e.model_id = ?
        JOIN documents d ON d.doc_id = c.doc_id
        WHERE TRUE{flt}
        ORDER BY score DESC LIMIT ?""",
        [qv, embedder.model_id, *params, k],
    )
    return [Hit(*r) for r in rows]


def bm25_search(store: Store, query: str, k: int = 10, state: str | None = None) -> list[Hit]:
    terms = sorted(set(tokenize(query)))
    if not terms:
        return []
    flt, params = _filter_sql(state)
    rows = store.sql(
        f"""
        WITH stats AS (SELECT COUNT(*) AS n, AVG(words) AS avgdl FROM chunks),
        df AS (SELECT term, COUNT(*) AS df FROM postings WHERE term IN (SELECT UNNEST(?)) GROUP BY term),
        scored AS (
          SELECT p.chunk_id,
                 SUM(LN(1 + (s.n - df.df + 0.5) / (df.df + 0.5))
                     * p.tf * ({K1} + 1) / (p.tf + {K1} * (1 - {B} + {B} * c.words / s.avgdl))) AS score
          FROM postings p JOIN df USING (term) JOIN chunks c USING (chunk_id), stats s
          GROUP BY p.chunk_id
        )
        SELECT c.chunk_id, c.doc_id, c.heading, c.text, sc.score
        FROM scored sc JOIN chunks c USING (chunk_id) JOIN documents d ON d.doc_id = c.doc_id
        WHERE TRUE{flt}
        ORDER BY sc.score DESC LIMIT ?""",
        [terms, *params, k],
    )
    return [Hit(*r) for r in rows]


def hybrid_search(
    store: Store, embedder: Embedder, query: str, k: int = 10, state: str | None = None, pool: int = 40
) -> list[Hit]:
    """Reciprocal rank fusion: robust to the two scores being on different scales."""
    fused: dict[str, float] = {}
    hits: dict[str, Hit] = {}
    for results in (vector_search(store, embedder, query, pool, state), bm25_search(store, query, pool, state)):
        for rank, h in enumerate(results, start=1):
            fused[h.chunk_id] = fused.get(h.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            hits.setdefault(h.chunk_id, h)
    ranked = sorted(fused, key=fused.get, reverse=True)[:k]
    return [Hit(hits[c].chunk_id, hits[c].doc_id, hits[c].heading, hits[c].text, fused[c]) for c in ranked]


MODES = ("vector", "bm25", "hybrid", "hybrid+filter")


def search(store: Store, embedder: Embedder, query: str, mode: str = "hybrid+filter", k: int = 10) -> list[Hit]:
    if mode == "vector":
        return vector_search(store, embedder, query, k)
    if mode == "bm25":
        return bm25_search(store, query, k)
    if mode == "hybrid":
        return hybrid_search(store, embedder, query, k)
    if mode == "hybrid+filter":
        return hybrid_search(store, embedder, query, k, state=detect_state(query))
    raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
