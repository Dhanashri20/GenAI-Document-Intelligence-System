"""
pipelines.latex_pipeline.latex_pipeline
=======================================

The scientific / LaTeX pipeline. It uses scientific embeddings, the
equation-aware ``LatexChunker``, and adds an ``explain_equation`` method that
produces a plain-English, step-by-step breakdown of a formula grounded in the
surrounding document context.

QA and summarization reuse the base orchestration with math-preserving prompts.
"""

from __future__ import annotations

import re
from typing import Optional

from config.settings import AppConfig, DEFAULT_CONFIG
from core.chunker import BaseChunker
from core.embeddings import EmbeddingManager
from core.generator import Generator
from core.schema import Document
from pipelines.base_pipeline.pipeline import BasePipeline, QAResult
from pipelines.latex_pipeline.latex_chunker import LatexChunker, _EQ_REGEX
from utils import prompt_templates as pt
from utils.formatting import clean_answer


def build_scientific_embedder() -> EmbeddingManager:
    manager = EmbeddingManager(default_model="scientific")
    manager.set_active("scientific")
    return manager


class LatexPipeline(BasePipeline):
    name = "scientific"
    qa_template = pt.QA_TEMPLATE
    summary_template = pt.LATEX_SUMMARY_TEMPLATE

    def __init__(self, config: AppConfig = DEFAULT_CONFIG,
                 generator: Optional[Generator] = None):
        super().__init__(
            config=config,
            embedder=build_scientific_embedder(),
            generator=generator,
        )

    def _build_chunker(self) -> BaseChunker:
        return LatexChunker(self.config.chunking)

    def _preprocess(self, document: Document) -> Document:
        document.doc_type = "scientific"
        return document

    # ------------------------------------------------------------------ #
    def explain_equation(self, query: str, top_k: Optional[int] = None) -> QAResult:
        """Retrieve equation-bearing context and explain it step by step."""
        self._require_index()
        chunks = self.retriever.retrieve(query, top_k=top_k)
        # Prefer chunks that actually contain equations, if any were retrieved.
        eq_chunks = [c for c in chunks if c.metadata.get("has_equation")]
        ordered = eq_chunks + [c for c in chunks if c not in eq_chunks]
        context = pt.build_context(ordered)
        prompt = pt.LATEX_EXPLAIN_TEMPLATE.format(context=context, question=query)
        raw = clean_answer(self.generator.generate(prompt))
        final, report = self.checker.gate(raw, ordered)
        return QAResult(answer=final, sources=ordered, report=report)

    # ------------------------------------------------------------------ #
    def list_equations(self, limit: int = 20):
        """Return distinct equation strings found in the indexed document."""
        if self.store is None:
            return []
        found = []
        seen = set()
        for c in self.store.chunks:
            for m in _EQ_REGEX.finditer(c.text):
                eq = m.group(0).strip()
                if eq not in seen and len(eq) > 3:
                    seen.add(eq)
                    found.append(eq)
                if len(found) >= limit:
                    return found
        return found
