"""
pipelines.base_pipeline.pipeline
================================

``BasePipeline`` wires the core components into the end-to-end workflow:

    ingest -> chunk -> embed -> store        (index time)
    question -> retrieve -> generate -> gate (query time)

Domain pipelines (medical, LaTeX) subclass this and override the chunker,
embedding choice, and prompt templates while reusing all the orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections import OrderedDict
from typing import List, Optional

from config.settings import AppConfig, DEFAULT_CONFIG, PIPELINE_DEFAULT_EMBEDDING
from core.chunker import BaseChunker
from core.embeddings import EmbeddingManager
from core.generator import Generator
from core.retriever import Retriever
from core.schema import Chunk, Document
from core.vectorstore import VectorStore
from evaluation.hallucination_checker import GroundednessReport, HallucinationChecker
from utils import prompt_templates as pt
from utils.formatting import clean_answer


@dataclass
class QAResult:
    answer: str
    sources: List[Chunk]
    report: GroundednessReport


@dataclass
class SummaryResult:
    summary: str
    sections_covered: int


class BasePipeline:
    name = "general"
    qa_template = pt.QA_TEMPLATE
    summary_template = pt.SUMMARY_TEMPLATE

    def __init__(
        self,
        config: AppConfig = DEFAULT_CONFIG,
        embedder: Optional[EmbeddingManager] = None,
        generator: Optional[Generator] = None,
    ):
        self.config = config
        self.embedder = embedder or EmbeddingManager(
            default_model=PIPELINE_DEFAULT_EMBEDDING.get(self.name, "general")
        )
        self.embedder.set_active(PIPELINE_DEFAULT_EMBEDDING.get(self.name, "general"))
        self.generator = generator or Generator(config.generator_model, config.generation)
        self.checker = HallucinationChecker(self.embedder, config.safety)

        self.chunker = self._build_chunker()
        self.store: Optional[VectorStore] = None
        self.retriever: Optional[Retriever] = None
        self.document: Optional[Document] = None

    # ------------------------------------------------------------------ #
    # Overridable hooks
    # ------------------------------------------------------------------ #
    def _build_chunker(self) -> BaseChunker:
        return BaseChunker(self.config.chunking)

    def _preprocess(self, document: Document) -> Document:
        """Hook for domain enrichment (e.g. NER tagging). Default: passthrough."""
        return document

    # ------------------------------------------------------------------ #
    # Index time
    # ------------------------------------------------------------------ #
    def index(self, document: Document) -> int:
        """Chunk, embed and store a document. Returns the number of chunks."""
        document = self._preprocess(document)
        self.document = document

        chunks: List[Chunk] = self.chunker.chunk(document)
        if not chunks:
            raise ValueError("Document produced no chunks.")

        vectors = self.embedder.embed([c.text for c in chunks])
        self.store = VectorStore(self.embedder.dimension)
        self.store.add(chunks, vectors)
        self.retriever = Retriever(self.store, self.embedder, self.config.retrieval)
        return len(chunks)

    # ------------------------------------------------------------------ #
    # Query time
    # ------------------------------------------------------------------ #
    def ask(self, question: str, top_k: Optional[int] = None) -> QAResult:
        self._require_index()
        chunks = self.retriever.retrieve(question, top_k=top_k)
        context = pt.build_context(chunks)
        prompt = self.qa_template.format(context=context, question=question)
        raw = clean_answer(self.generator.generate(prompt))
        final, report = self.checker.gate(raw, chunks)
        return QAResult(answer=final, sources=chunks, report=report)

    def summarize(
        self,
        query: Optional[str] = None,
        max_groups: int = 6,
    ) -> SummaryResult:
        """Summarize the indexed document.

        Two modes:

        * **Focused** (``query`` provided): retrieve the chunks most relevant to
          the query and summarize only those — a topic-driven summary that goes
          through the RAG retriever.
        * **Full-document** (default): a map-reduce summary that covers the
          *whole* document instead of just the leading chunks. Chunks are grouped
          (by detected section when available, otherwise into sequential
          windows), each group is summarized independently ("map"), and the
          partial summaries are then synthesized into one coherent summary
          ("reduce"). The number of map calls is bounded by ``max_groups`` to
          keep CPU cost predictable.
        """
        self._require_index()

        # ---- Focused, query-driven summary (uses retrieval) ----
        if query:
            chunks = self.retriever.retrieve(query, top_k=self.config.retrieval.top_k)
            context = pt.build_context(chunks, max_chars=4000)
            prompt = self.summary_template.format(context=context)
            summary = clean_answer(self.generator.generate(prompt))
            return SummaryResult(summary=summary, sections_covered=len(chunks))

        # ---- Full-document map-reduce summary ----
        groups = self._group_chunks(max_groups)
        if not groups:
            return SummaryResult(summary="", sections_covered=0)

        # Small document -> a single pass over everything is enough.
        if len(groups) == 1:
            context = pt.build_context(groups[0], max_chars=4000)
            prompt = self.summary_template.format(context=context)
            summary = clean_answer(self.generator.generate(prompt))
            return SummaryResult(summary=summary, sections_covered=len(groups))

        # MAP: summarize each group independently.
        partials: List[str] = []
        for group in groups:
            context = pt.build_context(group, max_chars=2500)
            prompt = self.summary_template.format(context=context)
            part = clean_answer(self.generator.generate(prompt))
            if part:
                partials.append(part)

        # REDUCE: synthesize partials into one coherent summary.
        joined = "\n".join(f"- {p}" for p in partials)
        reduce_prompt = pt.REDUCE_SUMMARY_TEMPLATE.format(partial_summaries=joined)
        summary = clean_answer(self.generator.generate(reduce_prompt))
        return SummaryResult(summary=summary, sections_covered=len(groups))

    # ------------------------------------------------------------------ #
    # Chunk grouping for map-reduce summarization
    # ------------------------------------------------------------------ #
    def _group_chunks(self, max_groups: int) -> List[List[Chunk]]:
        """Group indexed chunks for map-reduce summarization.

        Prefers grouping by detected section (order preserved). If there are
        more sections than ``max_groups``, adjacent sections are merged into
        balanced buckets. Documents without sections are split into sequential
        windows.
        """
        chunks = self.store.chunks
        if not chunks:
            return []

        by_section: "OrderedDict[str, List[Chunk]]" = OrderedDict()
        for c in chunks:
            key = c.section or "Document"
            by_section.setdefault(key, []).append(c)
        groups = list(by_section.values())

        # No real section structure -> sequential windows.
        if len(groups) <= 1:
            return self._window(chunks, max_groups)

        # Too many sections -> merge adjacent into ~max_groups buckets.
        if len(groups) > max_groups:
            groups = self._merge_adjacent(groups, max_groups)
        return groups

    @staticmethod
    def _window(items: List[Chunk], n: int) -> List[List[Chunk]]:
        """Split items into at most ``n`` contiguous windows."""
        if len(items) <= n:
            return [items]  # small enough for a single pass
        size = math.ceil(len(items) / n)
        return [items[i:i + size] for i in range(0, len(items), size)]

    @staticmethod
    def _merge_adjacent(groups: List[List[Chunk]], n: int) -> List[List[Chunk]]:
        """Merge adjacent section groups down to ~``n`` balanced buckets."""
        size = math.ceil(len(groups) / n)
        merged: List[List[Chunk]] = []
        for i in range(0, len(groups), size):
            bucket: List[Chunk] = []
            for g in groups[i:i + size]:
                bucket.extend(g)
            merged.append(bucket)
        return merged

    # ------------------------------------------------------------------ #
    def _require_index(self) -> None:
        if self.store is None or self.retriever is None:
            raise RuntimeError("No document indexed. Call index(document) first.")