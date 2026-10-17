import math
from collections import Counter

from ragpipe import corpus
from ragpipe.evaluate import evaluate
from ragpipe.retrieval import bm25_search, detect_state, search
from ragpipe.text import tokenize


def test_state_detection_handles_names_and_abbreviations():
    assert detect_state("late fee for a customer in North Carolina") == "NC"
    assert detect_state("max late fee for an OH acct?") == "OH"
    assert detect_state("how do I log in to the portal") is None  # lower-case "in" is not Indiana


def test_bm25_in_sql_matches_a_reference_implementation(synced):
    store, *_ = synced
    chunks = store.sql("SELECT chunk_id, text, words FROM chunks")
    n, avgdl = len(chunks), sum(c[2] for c in chunks) / len(chunks)
    tfs = {cid: Counter(tokenize(t)) for cid, t, _ in chunks}
    q = "payoff quote valid days Texas"
    terms = set(tokenize(q))
    df = {t: sum(1 for c in tfs.values() if t in c) for t in terms}

    def score(cid, words):
        s = 0.0
        for t in terms:
            tf = tfs[cid].get(t, 0)
            if tf:
                s += math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * words / avgdl))
        return s

    ref = {cid: score(cid, w) for cid, _, w in chunks}
    want = sorted(ref.values(), reverse=True)[:5]
    got = bm25_search(store, q, 5)
    assert all(math.isclose(h.score, s, rel_tol=1e-6) for h, s in zip(got, want, strict=True))
    assert all(math.isclose(h.score, ref[h.chunk_id], rel_tol=1e-6) for h in got)  # ties may order either way


def test_filtered_search_never_returns_another_states_policy(synced):
    store, emb, *_ = synced
    for h in search(store, emb, "NC payoff letter expires after how long?", "hybrid+filter", k=10):
        assert h.doc_id.startswith("policies/nc/") or h.doc_id.startswith("faq/")


def test_retrieval_quality_gate_and_freshness(synced):
    """After the version 2 sync, answers must reflect the *new* policy values."""
    store, emb, *_ = synced
    scores = {s.mode: s for s in evaluate(store, emb, corpus.questions(2))}
    best = scores["hybrid+filter"]
    assert best.recall_at_5 >= 0.97, best.misses
    assert best.mrr_at_10 > scores["vector"].mrr_at_10 and best.mrr_at_10 > scores["bm25"].mrr_at_10
