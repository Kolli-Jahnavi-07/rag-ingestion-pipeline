from ragpipe import corpus
from ragpipe.pipeline import discover


def test_first_sync_indexes_everything(synced):
    store, _, r1, _, _ = synced
    assert r1.new_docs == len(corpus.corpus(1)[0]) and r1.changed_docs == r1.deleted_docs == 0
    assert r1.chunks_written > len(corpus.corpus(1)[0])  # several chunks per document
    assert r1.embeddings_computed > 0 and r1.pii_redactions >= 1


def test_resync_without_changes_does_nothing(synced):
    _, _, r1, again, _ = synced
    assert again.unchanged_docs == r1.new_docs
    assert again.chunks_written == again.embeddings_computed == 0


def test_incremental_sync_reports_exactly_what_changed(synced):
    _, _, _, _, r2 = synced
    assert (r2.new_docs, r2.changed_docs, r2.deleted_docs, r2.duplicate_docs) == (3, 5, 2, 1)
    # Changed documents keep most of their chunks; those vectors come from the cache.
    assert r2.embeddings_reused > 0 and r2.embeddings_computed < r2.chunks_written


def test_index_matches_the_source_folder(synced, tmp_path):
    store, *_ = synced
    corpus.build(tmp_path, 2)
    on_disk = set(discover(tmp_path))
    indexed = {d for (d,) in store.sql("SELECT doc_id FROM documents")}
    assert indexed == on_disk
    assert store.sql("SELECT COUNT(*) FROM chunks WHERE doc_id NOT IN (SELECT doc_id FROM documents)") == [(0,)]
    # The duplicate upload is tracked but contributes no chunks of its own.
    assert store.sql("SELECT duplicate_of FROM documents WHERE doc_id = 'faq/autopay_copy'") == [("faq/autopay",)]
    assert store.sql("SELECT COUNT(*) FROM chunks WHERE doc_id = 'faq/autopay_copy'") == [(0,)]


def test_no_stale_or_sensitive_text_in_the_index(synced):
    store, *_ = synced
    assert store.sql("SELECT COUNT(*) FROM chunks WHERE doc_id LIKE '%/ga/hardship%'") == [(0,)]
    assert store.sql("SELECT COUNT(*) FROM chunks WHERE text LIKE '%123-45-6789%'") == [(0,)]
    assert store.sql("SELECT COUNT(*) FROM chunks WHERE text LIKE '%internal servicing use%'") == [(0,)]  # footer stripped
    assert store.sql("SELECT COUNT(*) FROM embeddings WHERE content_hash NOT IN (SELECT content_hash FROM chunks)") == [(0,)]
