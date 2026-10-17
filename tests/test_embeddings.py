import math

import pytest

from ragpipe.embeddings import HashingEmbedder, OpenAICompatibleEmbedder


def cos(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_hashing_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder(256)
    a, b = e.embed(["late charge for Ohio accounts"]), e.embed(["late charge for Ohio accounts"])
    assert a == b
    assert math.isclose(sum(v * v for v in a[0]), 1.0, rel_tol=1e-9)


def test_related_text_is_closer_than_unrelated_text():
    e = HashingEmbedder()
    q, rel, unrel = e.embed(["repo notice days", "Repossession notices: days of notice", "Paperless statements by email"])
    assert cos(q, rel) > cos(q, unrel) + 0.1


class FakeResponse:
    def __init__(self, status, body=None, headers=None):
        self.status_code, self._body, self.headers = status, body, headers or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeSession:
    def __init__(self):
        self.headers, self.calls = {}, []

    def post(self, url, json, timeout):
        self.calls.append(json)
        if len(self.calls) == 1:
            return FakeResponse(429, headers={"Retry-After": "1"})
        data = [{"index": i, "embedding": [float(len(t))]} for i, t in enumerate(json["input"])]
        return FakeResponse(200, {"data": list(reversed(data))})  # API may return out of order


def test_api_embedder_batches_retries_and_keeps_order(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    sleeps = []
    s = FakeSession()
    e = OpenAICompatibleEmbedder(dim=1, batch_size=2, session=s, sleep=sleeps.append)
    out = e.embed(["a", "bb", "ccc"])
    assert out == [[1.0], [2.0], [3.0]]
    assert [len(c["input"]) for c in s.calls] == [2, 2, 1]  # first call was the 429, retried
    assert sleeps == [1.0] and s.headers["Authorization"] == "Bearer k"


def test_unknown_embedder():
    from ragpipe.embeddings import build

    with pytest.raises(ValueError):
        build("nope")
