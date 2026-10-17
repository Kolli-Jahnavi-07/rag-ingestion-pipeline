"""Command line.

python -m ragpipe corpus --version 1 --out data/docs     # write the sample knowledge base
python -m ragpipe sync data/docs                          # incremental ingest into data/index.duckdb
python -m ragpipe ask "late fee in Ohio?"                 # top chunks (hybrid + metadata filter)
python -m ragpipe eval --min-recall5 0.9                  # retrieval quality gate (exit 1 if below)
python -m ragpipe demo                                     # version 1 -> eval -> version 2 -> freshness eval
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from . import corpus
from .embeddings import build as build_embedder
from .evaluate import evaluate, print_scores
from .pipeline import SyncReport, sync
from .retrieval import search
from .store import Store


def _report(r: SyncReport) -> None:
    print(
        f"  docs: {r.new_docs} new, {r.changed_docs} changed, {r.deleted_docs} deleted, {r.unchanged_docs} unchanged, "
        f"{r.duplicate_docs} duplicate"
    )
    print(
        f"  chunks written: {r.chunks_written}; embeddings computed: {r.embeddings_computed}, "
        f"reused from cache: {r.embeddings_reused} ({r.embedding_savings:.0%} saved); PII redactions: {r.pii_redactions}; "
        f"{r.seconds:.1f}s"
    )


def demo(workdir: Path) -> int:
    if workdir.exists():
        shutil.rmtree(workdir)
    docs_dir = workdir / "docs"
    workdir.mkdir(parents=True)
    emb = build_embedder("hashing")
    store = Store(str(workdir / "index.duckdb"), dim=emb.dim)

    corpus.build(docs_dir, version=1)
    print("Version 1 of the knowledge base")
    _report(sync(docs_dir, store, emb))
    qs1 = corpus.questions(1)
    print(f"\nRetrieval quality on {len(qs1)} graded questions:")
    scores = evaluate(store, emb, qs1)
    print_scores(scores)

    print("\nRe-sync with nothing changed")
    _report(sync(docs_dir, store, emb))

    corpus.build(docs_dir, version=2)
    print("\nVersion 2: 5 policies changed, 2 programs retired, 2 lease programs launched, 1 FAQ uploaded twice")
    _report(sync(docs_dir, store, emb))
    qs2 = corpus.questions(2)
    print(f"\nRetrieval quality after the update ({len(qs2)} questions, answers reflect the new policy values):")
    print_scores(evaluate(store, emb, qs2))
    stale = store.sql("SELECT COUNT(*) FROM chunks WHERE doc_id LIKE '%/ga/hardship%' OR doc_id LIKE '%/az/hardship%'")[0][0]
    print(f"chunks left from retired programs: {stale}")

    q = "What late charge can we assess on a North Carolina account?"
    print(f"\nExample: {q}")
    for h in search(store, emb, q, k=3):
        print(f"  [{h.score:.3f}] {h.doc_id} :: {h.heading}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ragpipe")
    ap.add_argument("--index", default="data/index.duckdb")
    ap.add_argument("--embedder", choices=["hashing", "api"], default="hashing")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("corpus")
    c.add_argument("--version", type=int, default=1)
    c.add_argument("--out", type=Path, default=Path("data/docs"))
    s = sub.add_parser("sync")
    s.add_argument("docs", type=Path)
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--mode", default="hybrid+filter")
    a.add_argument("-k", type=int, default=5)
    e = sub.add_parser("eval")
    e.add_argument("--version", type=int, default=1)
    e.add_argument("--min-recall5", type=float, default=0.0)
    d = sub.add_parser("demo")
    d.add_argument("--workdir", type=Path, default=Path("data/demo"))
    args = ap.parse_args(argv)

    if args.cmd == "demo":
        return demo(args.workdir)
    if args.cmd == "corpus":
        docs = corpus.build(args.out, args.version)
        print(f"wrote {len(docs)} documents to {args.out}")
        return 0
    emb = build_embedder(args.embedder)
    Path(args.index).parent.mkdir(parents=True, exist_ok=True)
    store = Store(args.index, dim=emb.dim)
    if args.cmd == "sync":
        _report(sync(args.docs, store, emb))
        return 0
    if args.cmd == "ask":
        for h in search(store, emb, args.question, args.mode, args.k):
            print(f"[{h.score:.3f}] {h.doc_id} :: {h.heading}\n    {h.text.splitlines()[-1][:200]}")
        return 0
    if args.cmd == "eval":
        scores = evaluate(store, emb, corpus.questions(args.version))
        print_scores(scores)
        best = next(s for s in scores if s.mode == "hybrid+filter")
        if best.recall_at_5 < args.min_recall5:
            print(f"FAIL: hybrid+filter recall@5 {best.recall_at_5:.1%} < {args.min_recall5:.0%}")
            for m in best.misses:
                print(f"  missed: {m}")
            return 1
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
