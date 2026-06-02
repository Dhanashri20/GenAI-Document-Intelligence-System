"""
core.embeddings
===============

Wraps sentence-transformers behind a small manager that:

* exposes logical model names ("general", "biomedical", "scientific"),
* lazily loads and caches models so switching is cheap,
* normalizes vectors (so cosine similarity == inner product downstream),
* degrades gracefully to a deterministic hashing embedder if
  sentence-transformers / the model download is unavailable (keeps the demo
  runnable offline; quality is obviously lower).
"""

from __future__ import annotations

import hashlib
from typing import Dict, List

import numpy as np

from config.settings import EMBEDDING_MODELS


class _HashingEmbedder:
    """Offline fallback. Deterministic bag-of-words hashing into a fixed dim.

    This is NOT semantically strong; it exists only so the pipeline runs end to
    end when real models cannot be downloaded. The real path is SentenceTransformer.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, texts: List[str], **_) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in text.lower().split():
                h = int(hashlib.md5(token.encode()).hexdigest(), 16)
                vecs[i, h % self.dim] += 1.0
        # L2 normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


class EmbeddingManager:
    """Loads, caches and serves embedding models by logical name."""

    def __init__(self, default_model: str = "general"):
        self._cache: Dict[str, object] = {}
        self._dims: Dict[str, int] = {}
        self.active_name: str = default_model

    # ------------------------------------------------------------------ #
    def _load(self, name: str):
        if name in self._cache:
            return self._cache[name]

        model_id = EMBEDDING_MODELS.get(name, EMBEDDING_MODELS["general"])
        try:
            from sentence_transformers import SentenceTransformer

            # device="cpu" keeps this deployable anywhere.
            model = SentenceTransformer(model_id, device="cpu")
            dim = model.get_sentence_embedding_dimension()
        except Exception as exc:  # noqa: BLE001 - we intentionally fall back
            print(f"[embeddings] Falling back to hashing embedder ({exc}).")
            model = _HashingEmbedder()
            dim = model.dim

        self._cache[name] = model
        self._dims[name] = dim
        return model

    # ------------------------------------------------------------------ #
    def set_active(self, name: str) -> None:
        """Switch the active embedding model at runtime."""
        if name not in EMBEDDING_MODELS:
            raise ValueError(f"Unknown embedding model '{name}'.")
        self.active_name = name
        self._load(name)  # warm the cache

    @property
    def dimension(self) -> int:
        self._load(self.active_name)
        return self._dims[self.active_name]

    # ------------------------------------------------------------------ #
    def embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Embed a list of texts with the active model. Returns float32 (n, dim)."""
        model = self._load(self.active_name)
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ) if not isinstance(model, _HashingEmbedder) else model.encode(texts)
        return np.asarray(vectors, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
