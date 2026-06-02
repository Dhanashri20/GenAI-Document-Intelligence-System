"""
pipelines.medical_pipeline.medical_embeddings
=============================================

Thin convenience wrapper that returns an ``EmbeddingManager`` pre-set to the
biomedical model. Biomedical embeddings place clinical terms (e.g. "myocardial
infarction" vs "heart attack") closer together than general models, improving
retrieval on medical corpora.

Kept as a separate module to satisfy the required pipeline layout and to give a
single obvious place to change the medical embedding backend.
"""

from __future__ import annotations

from core.embeddings import EmbeddingManager


def build_medical_embedder() -> EmbeddingManager:
    manager = EmbeddingManager(default_model="biomedical")
    manager.set_active("biomedical")
    return manager
