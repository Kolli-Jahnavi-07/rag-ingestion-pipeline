"""DuckDB-backed index: documents, chunks, an embedding cache, and a BM25 inverted index.

The same layout maps onto Postgres + pgvector (sql/pgvector_schema.sql) or Snowflake
(VECTOR columns + VECTOR_COSINE_SIMILARITY) for production.
"""

from __future__ import annotations

from collections import Counter

import duckdb

from .chunking import Chunk
from .text import tokenize


class Store:
    def __init__(self, path: str = ":memory:", dim: int = 1024):
        self.con = duckdb.connect(path)
        self.dim = dim
        self.con.execute(f"""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY, content_hash TEXT NOT NULL, title TEXT, state TEXT, topic TEXT,
                duplicate_of TEXT, version INTEGER NOT NULL, synced_at TIMESTAMP NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, ordinal INTEGER, heading TEXT, text TEXT,
                content_hash TEXT NOT NULL, words INTEGER
            );
            CREATE TABLE IF NOT EXISTS embeddings (
                content_hash TEXT, model_id TEXT, vector FLOAT[{dim}], PRIMARY KEY (content_hash, model_id)
            );
            CREATE TABLE IF NOT EXISTS postings (term TEXT, chunk_id TEXT, tf INTEGER);
            CREATE TABLE IF NOT EXISTS sync_log (
                run_at TIMESTAMP, new_docs INTEGER, changed_docs INTEGER, deleted_docs INTEGER, unchanged_docs INTEGER,
                duplicate_docs INTEGER, chunks_written INTEGER, embeddings_computed INTEGER, embeddings_reused INTEGER
            );
        """)

    def sql(self, q: str, params: list | None = None) -> list[tuple]:
        return self.con.execute(q, params or []).fetchall()

    # ------------------------------------------------------------------ documents
    def documents(self) -> dict[str, tuple[str, int]]:
        return {d: (h, v) for d, h, v in self.sql("SELECT doc_id, content_hash, version FROM documents")}

    def upsert_document(
        self,
        doc_id: str,
        content_hash: str,
        title: str,
        state: str | None,
        topic: str | None,
        duplicate_of: str | None,
        version: int,
    ) -> None:
        self.con.execute("DELETE FROM documents WHERE doc_id = ?", [doc_id])
        self.con.execute(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, now())",
            [doc_id, content_hash, title, state, topic, duplicate_of, version],
        )

    def delete_document(self, doc_id: str) -> None:
        self.remove_chunks(doc_id)
        self.con.execute("DELETE FROM documents WHERE doc_id = ?", [doc_id])

    # ------------------------------------------------------------------ chunks
    def remove_chunks(self, doc_id: str) -> None:
        self.con.execute("DELETE FROM postings WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE doc_id = ?)", [doc_id])
        self.con.execute("DELETE FROM chunks WHERE doc_id = ?", [doc_id])

    def write_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self.con.executemany(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(c.chunk_id, c.doc_id, c.ordinal, c.heading, c.text, c.content_hash, c.words) for c in chunks],
        )
        rows = [(term, c.chunk_id, tf) for c in chunks for term, tf in Counter(tokenize(c.text)).items()]
        self.con.executemany("INSERT INTO postings VALUES (?, ?, ?)", rows)

    # ------------------------------------------------------------------ embeddings cache
    def cached_hashes(self, model_id: str, hashes: list[str]) -> set[str]:
        if not hashes:
            return set()
        rows = self.sql(
            "SELECT content_hash FROM embeddings WHERE model_id = ? AND content_hash IN (SELECT UNNEST(?))",
            [model_id, hashes],
        )
        return {r[0] for r in rows}

    def put_embeddings(self, model_id: str, items: list[tuple[str, list[float]]]) -> None:
        if items:
            self.con.executemany("INSERT INTO embeddings VALUES (?, ?, ?)", [(h, model_id, v) for h, v in items])

    def prune_embeddings(self) -> int:
        """Drop cached vectors no chunk uses any more (keeps the index from growing forever)."""
        n = self.sql("SELECT COUNT(*) FROM embeddings WHERE content_hash NOT IN (SELECT content_hash FROM chunks)")[0][0]
        self.con.execute("DELETE FROM embeddings WHERE content_hash NOT IN (SELECT content_hash FROM chunks)")
        return n
