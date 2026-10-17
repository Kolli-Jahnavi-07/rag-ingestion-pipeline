"""Retrieval evaluation against graded questions.

A retrieved chunk counts as correct when it comes from the expected document *and* contains the
expected fact. Checking the fact (not just the document) is what catches stale chunks after a
policy change.
"""

from __future__ import annotations

from dataclasses import dataclass

from .corpus import Question
from .embeddings import Embedder
from .retrieval import MODES, search
from .store import Store


@dataclass
class Scores:
    mode: str
    questions: int
    recall_at_1: float
    recall_at_5: float
    mrr_at_10: float
    misses: list[str]


def first_correct_rank(hits, q: Question) -> int | None:
    for rank, h in enumerate(hits, start=1):
        if h.doc_id == q.doc_id and q.answer in h.text:
            return rank
    return None


def evaluate(store: Store, embedder: Embedder, questions: list[Question], modes=MODES, k: int = 10) -> list[Scores]:
    out = []
    for mode in modes:
        ranks = [first_correct_rank(search(store, embedder, q.text, mode, k), q) for q in questions]
        n = len(questions)
        out.append(
            Scores(
                mode=mode,
                questions=n,
                recall_at_1=sum(r == 1 for r in ranks) / n,
                recall_at_5=sum(r is not None and r <= 5 for r in ranks) / n,
                mrr_at_10=sum(1 / r for r in ranks if r) / n,
                misses=[q.text for q, r in zip(questions, ranks, strict=True) if r is None or r > 5],
            )
        )
    return out


def print_scores(scores: list[Scores]) -> None:
    print(f"{'mode':<15} {'recall@1':>9} {'recall@5':>9} {'MRR@10':>8}")
    for s in scores:
        print(f"{s.mode:<15} {s.recall_at_1:>9.1%} {s.recall_at_5:>9.1%} {s.mrr_at_10:>8.3f}")
