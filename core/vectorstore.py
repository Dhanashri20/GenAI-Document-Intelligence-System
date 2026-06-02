"""
core.vectorstore
================

A thin vector store over FAISS (inner-product index on normalized vectors ==
cosine similarity). Chunk objects are stored alongside the index so search
returns full provenance (page, section, type). If FAISS is unavailable the
store falls back to a brute-force numpy implementation with identical behavior.
"""

from __future__ import annotations

import pickle
from typing import List, Tuple

import numpy as np

from core.schema import Chunk


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self._chunks: List[Chunk] = []
        self._backend = "faiss"
        self._matrix: np.ndarray | None = None  # numpy fallback storage

        try:
            import faiss  # noqa: F401

            self._faiss = faiss
            self._index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors
        except ImportError:
            self._faiss = None
            self._backend = "numpy"
            self._index = None

    # ------------------------------------------------------------------ #
    def add(self, chunks: List[Chunk], vectors: np.ndarray) -> None:
        """Add chunks and their (already L2-normalized) embedding vectors."""
        if vectors.shape[0] != len(chunks):
            raise ValueError("chunks and vectors length mismatch")
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        if self._backend == "faiss":
            self._index.add(vectors)
        else:
            self._matrix = (
                vectors if self._matrix is None
                else np.vstack([self._matrix, vectors])
            )
        self._chunks.extend(chunks)

    # ------------------------------------------------------------------ #
    def search(self, query_vec: np.ndarray, k: int) -> List[Chunk]:
        """Return the top-k chunks, each with its similarity score attached."""
        if not self._chunks:
            return []
        k = min(k, len(self._chunks))
        q = np.ascontiguousarray(query_vec.reshape(1, -1), dtype=np.float32)

        if self._backend == "faiss":
            scores, idxs = self._index.search(q, k)
            scores, idxs = scores[0], idxs[0]
        else:
            sims = (self._matrix @ q[0])
            idxs = np.argsort(-sims)[:k]
            scores = sims[idxs]

        results: List[Chunk] = []
        for score, idx in zip(scores, idxs):
            if idx < 0:
                continue
            chunk = self._chunks[int(idx)]
            # Return a shallow copy so per-query scores don't mutate the store.
            scored = Chunk(
                text=chunk.text,
                chunk_id=chunk.chunk_id,
                metadata=dict(chunk.metadata),
                score=float(score),
            )
            results.append(scored)
        return results

    # ------------------------------------------------------------------ #
    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> List[Chunk]:
        return self._chunks

    # ------------------------------------------------------------------ #
    def save(self, path_prefix: str) -> None:
        """Persist the store to ``<prefix>.index`` / ``<prefix>.pkl``."""
        if self._backend == "faiss":
            self._faiss.write_index(self._index, f"{path_prefix}.index")
            payload = {"chunks": self._chunks, "dim": self.dim, "backend": "faiss"}
        else:
            payload = {
                "chunks": self._chunks, "dim": self.dim,
                "backend": "numpy", "matrix": self._matrix,
            }
        with open(f"{path_prefix}.pkl", "wb") as fh:
            pickle.dump(payload, fh)

    @classmethod
    def load(cls, path_prefix: str) -> "VectorStore":
        with open(f"{path_prefix}.pkl", "rb") as fh:
            payload = pickle.load(fh)
        store = cls(payload["dim"])
        store._chunks = payload["chunks"]
        if payload["backend"] == "faiss" and store._faiss is not None:
            store._index = store._faiss.read_index(f"{path_prefix}.index")
        else:
            store._backend = "numpy"
            store._matrix = payload.get("matrix")
        return store
