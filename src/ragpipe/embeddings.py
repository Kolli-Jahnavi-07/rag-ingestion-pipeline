"""Embedding providers behind one interface.

`HashingEmbedder` is local and deterministic: signed feature hashing of word unigrams, bigrams and
character 4-grams (so "repo" lands near "repossession"), sublinear term frequency, L2-normalised.
It needs no model download or API key, never changes between runs (so stored vectors stay valid),
and is what CI and the demo use. It captures word overlap, not meaning; production uses a learned
embedding model through the same interface.

`OpenAICompatibleEmbedder` calls any OpenAI-compatible /v1/embeddings endpoint (OpenAI, Azure
OpenAI, a self-hosted model server) with batching and retries. Swap it in with `--embedder api`.
Vectors are cached by (model, chunk content hash), so a model is only ever asked once per chunk.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from collections import Counter
from typing import Protocol

import requests

from .text import tokenize


def _char_ngrams(token: str, n: int = 4) -> list[str]:
    t = f"<{token}>"
    return [t[i : i + n] for i in range(len(t) - n + 1)]


class Embedder(Protocol):
    model_id: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.model_id = f"hashing-words-chars-{dim}"

    def _index(self, feature: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")
        return h % self.dim, (1.0 if (h >> 63) & 1 else -1.0)

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            toks = tokenize(text)
            feats = Counter(toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)])
            grams = Counter(g for t in toks if len(t) > 3 for g in _char_ngrams(t))
            vec = [0.0] * self.dim
            for f, tf in feats.items():
                i, sign = self._index(f)
                vec[i] += sign * (1 + math.log(tf))
            for g, tf in grams.items():
                i, sign = self._index("#" + g)
                vec[i] += sign * 0.35 * (1 + math.log(tf))
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class OpenAICompatibleEmbedder:
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        batch_size: int = 96,
        session: requests.Session | None = None,
        sleep=time.sleep,
    ):
        self.model_id, self.dim, self.batch_size = model, dim, batch_size
        self.url = (base_url or os.environ.get("EMBEDDINGS_BASE_URL", "https://api.openai.com")).rstrip("/") + "/v1/embeddings"
        self.session = session or requests.Session()
        key = os.environ.get(api_key_env)
        if key:
            self.session.headers["Authorization"] = f"Bearer {key}"
        self.sleep = sleep

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            for attempt in range(6):
                resp = self.session.post(
                    self.url, json={"model": self.model_id, "input": batch, "dimensions": self.dim}, timeout=60
                )
                if resp.status_code in (429, 500, 502, 503) and attempt < 5:
                    self.sleep(float(resp.headers.get("Retry-After", 2**attempt)))
                    continue
                resp.raise_for_status()
                break
            data = sorted(resp.json()["data"], key=lambda d: d["index"])
            out += [d["embedding"] for d in data]
        return out


def build(kind: str = "hashing") -> Embedder:
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "api":
        return OpenAICompatibleEmbedder()
    raise ValueError(f"unknown embedder {kind!r}")
