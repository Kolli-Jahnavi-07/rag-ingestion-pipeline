import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragpipe import corpus  # noqa: E402
from ragpipe.embeddings import HashingEmbedder  # noqa: E402
from ragpipe.pipeline import sync  # noqa: E402
from ragpipe.store import Store  # noqa: E402


@pytest.fixture(scope="module")
def synced(tmp_path_factory):
    """Index version 1, then sync version 2; keep both reports."""
    docs = tmp_path_factory.mktemp("docs")
    emb = HashingEmbedder()
    store = Store(dim=emb.dim)
    corpus.build(docs, 1)
    r1 = sync(docs, store, emb)
    r_again = sync(docs, store, emb)
    corpus.build(docs, 2)
    r2 = sync(docs, store, emb)
    return store, emb, r1, r_again, r2
