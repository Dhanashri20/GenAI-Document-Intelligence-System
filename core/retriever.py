"""
core.retriever
==============

Turns a question into a ranked set of context chunks.

Two modes:
* **Dense only** — embed the query, search the vector store, return top-k.
* **Hybrid** — also score candidates with BM25 keyword matching and fuse the
  two normalized score lists with configurable weights. Hybrid retrieval helps
  on rare tokens (drug names, gene symbols, equation labels) that dense models
  sometimes under-weight.

If ``rank_bm25`` is not installed, hybrid silently degrades to dense-only.
"""

from __future__ import annotations

from typing import List

import numpy as np

from config.settings import RetrievalConfig
from core.embeddings import EmbeddingManager
from core.schema import Chunk
from core.vectorstore import VectorStore


def _minmax(scores: List[float]) -> List[float]:
    if not scores:
        return scores
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: EmbeddingManager,
        config: RetrievalConfig | None = None,
    ):
        self.store = store
        self.embedder = embedder
        self.config = config or RetrievalConfig()
        self._bm25 = None
        self._bm25_chunks: List[Chunk] = []
        if self.config.use_hybrid:
            self._build_bm25()

    # ------------------------------------------------------------------ #
    def _build_bm25(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            self._bm25 = None
            return
        self._bm25_chunks = self.store.chunks
        corpus = [c.text.lower().split() for c in self._bm25_chunks]
        if corpus:
            self._bm25 = BM25Okapi(corpus)

    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, top_k: int | None = None) -> List[Chunk]:
        top_k = top_k or self.config.top_k
        q_vec = self.embedder.embed_one(query)

        dense = self.store.search(q_vec, k=self.config.fetch_k)
        if not dense:
            return []

        if not (self.config.use_hybrid and self._bm25 is not None):
            return dense[:top_k]

        return self._fuse(query, dense)[:top_k]

    # ------------------------------------------------------------------ #
    def _fuse(self, query: str, dense: List[Chunk]) -> List[Chunk]:
        """Blend dense similarity with BM25 keyword score by chunk_id."""
        bm_scores = self._bm25.get_scores(query.lower().split())
        bm_by_id = {
            c.chunk_id: float(bm_scores[i])
            for i, c in enumerate(self._bm25_chunks)
        }

        dense_norm = _minmax([c.score for c in dense])
        kw_raw = [bm_by_id.get(c.chunk_id, 0.0) for c in dense]
        kw_norm = _minmax(kw_raw)

        sw, kw = self.config.semantic_weight, self.config.keyword_weight
        for chunk, d, k in zip(dense, dense_norm, kw_norm):
            chunk.score = sw * d + kw * k

        return sorted(dense, key=lambda c: c.score, reverse=True)
