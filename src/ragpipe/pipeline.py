"""Sync a document folder into the index, doing only the work that changed.

discover -> diff against the index (new / changed / deleted / unchanged / duplicate)
         -> parse, strip boilerplate, redact PII
         -> chunk (stable IDs)
         -> embed only chunks whose content hash has no cached vector
         -> write chunks + BM25 postings, remove stale ones
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .chunking import Chunk, chunk_document
from .embeddings import Embedder
from .store import Store
from .text import boilerplate, redact, sections, strip_boilerplate


@dataclass
class SyncReport:
    new_docs: int = 0
    changed_docs: int = 0
    deleted_docs: int = 0
    unchanged_docs: int = 0
    duplicate_docs: int = 0
    chunks_written: int = 0
    embeddings_computed: int = 0
    embeddings_reused: int = 0
    pii_redactions: int = 0
    seconds: float = 0.0

    @property
    def embedding_savings(self) -> float:
        total = self.embeddings_computed + self.embeddings_reused
        return self.embeddings_reused / total if total else 0.0

    def as_dict(self) -> dict:
        return {**asdict(self), "embedding_savings": round(self.embedding_savings, 3)}


def _metadata(doc_id: str) -> tuple[str | None, str | None]:
    parts = doc_id.split("/")
    if parts[0] == "policies" and len(parts) == 3:
        return parts[1].upper(), parts[2]
    return None, parts[-1]


def discover(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root).with_suffix("")): p.read_text(encoding="utf-8") for p in sorted(root.rglob("*.md"))}


def sync(root: Path, store: Store, embedder: Embedder) -> SyncReport:
    t0 = time.perf_counter()
    rep = SyncReport()
    docs = discover(root)
    indexed = store.documents()
    hashes = {d: hashlib.sha256(t.encode()).hexdigest() for d, t in docs.items()}

    # Deleted upstream -> remove from the index, or the assistant keeps quoting retired policy.
    for doc_id in set(indexed) - set(docs):
        store.delete_document(doc_id)
        rep.deleted_docs += 1

    # Exact duplicates (same content, different path) are tracked but not indexed twice.
    first_by_hash: dict[str, str] = {}
    for doc_id in sorted(docs):
        first_by_hash.setdefault(hashes[doc_id], doc_id)

    common = boilerplate(list(docs.values()))
    to_embed: list[Chunk] = []
    for doc_id, text in docs.items():
        h = hashes[doc_id]
        prev = indexed.get(doc_id)
        if prev and prev[0] == h:
            rep.unchanged_docs += 1
            continue
        version = prev[1] + 1 if prev else 1
        rep.changed_docs += bool(prev)
        rep.new_docs += not prev
        state, topic = _metadata(doc_id)
        store.remove_chunks(doc_id)
        original = first_by_hash[h]
        clean, n_pii = redact(strip_boilerplate(text, common))
        rep.pii_redactions += n_pii
        title, secs = sections(clean)
        if original != doc_id:
            store.upsert_document(doc_id, h, title, state, topic, original, version)
            rep.duplicate_docs += 1
            continue
        store.upsert_document(doc_id, h, title, state, topic, None, version)
        chunks = chunk_document(doc_id, title, secs)
        store.write_chunks(chunks)
        rep.chunks_written += len(chunks)
        to_embed += chunks

    unique = {c.content_hash: c.text for c in to_embed}
    cached = store.cached_hashes(embedder.model_id, list(unique))
    missing = [h for h in unique if h not in cached]
    if missing:
        vectors = embedder.embed([unique[h] for h in missing])
        store.put_embeddings(embedder.model_id, list(zip(missing, vectors, strict=True)))
    rep.embeddings_computed = len(missing)
    rep.embeddings_reused = len(cached)
    store.prune_embeddings()
    rep.seconds = round(time.perf_counter() - t0, 2)
    store.con.execute(
        "INSERT INTO sync_log VALUES (now(), ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            rep.new_docs,
            rep.changed_docs,
            rep.deleted_docs,
            rep.unchanged_docs,
            rep.duplicate_docs,
            rep.chunks_written,
            rep.embeddings_computed,
            rep.embeddings_reused,
        ],
    )
    return rep
