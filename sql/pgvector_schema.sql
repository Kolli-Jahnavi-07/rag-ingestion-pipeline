-- Production layout of the same index on Postgres + pgvector.
-- Mirrors src/ragpipe/store.py; the pipeline's Store class is the only thing that changes.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    doc_id        TEXT PRIMARY KEY,
    content_hash  TEXT NOT NULL,
    title         TEXT,
    state         TEXT,
    topic         TEXT,
    duplicate_of  TEXT REFERENCES documents (doc_id),
    version       INTEGER NOT NULL,
    synced_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      TEXT PRIMARY KEY,
    doc_id        TEXT NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
    ordinal       INTEGER,
    heading       TEXT,
    text          TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    words         INTEGER,
    -- keyword search: Postgres full-text search stands in for the BM25 postings table
    tsv           TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS chunks_tsv ON chunks USING GIN (tsv);

-- Embedding cache keyed by content, so identical chunks and unchanged chunks are embedded once.
CREATE TABLE IF NOT EXISTS embeddings (
    content_hash  TEXT NOT NULL,
    model_id      TEXT NOT NULL,
    embedding     VECTOR(1536) NOT NULL,
    PRIMARY KEY (content_hash, model_id)
);
CREATE INDEX IF NOT EXISTS embeddings_hnsw ON embeddings USING hnsw (embedding vector_cosine_ops);

-- Filtered vector search (metadata filter applied before ranking):
-- SELECT c.chunk_id, c.doc_id, 1 - (e.embedding <=> $1) AS score
-- FROM chunks c
-- JOIN embeddings e ON e.content_hash = c.content_hash AND e.model_id = $2
-- JOIN documents d ON d.doc_id = c.doc_id
-- WHERE d.state = $3 OR d.state IS NULL
-- ORDER BY e.embedding <=> $1
-- LIMIT 40;
