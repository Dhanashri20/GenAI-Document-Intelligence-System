"""
pipelines.medical_pipeline.medical_qa  (+ summarizer + pipeline)
================================================================

Defines the medical-domain pipeline. It reuses ``BasePipeline`` orchestration
but swaps in:

* the biomedical embedding model,
* the entity-aware ``MedicalChunker`` seeded by scispaCy NER,
* safety-framed medical prompt templates (no diagnosis, no certainty),
* an NER pre-processing step that attaches extracted entities to the document.

``medical_summarizer`` and the QA logic are exposed via the same ``ask`` /
``summarize`` surface as the base pipeline so the UI is pipeline-agnostic.
"""

from __future__ import annotations

from typing import Optional

from config.settings import AppConfig, DEFAULT_CONFIG
from core.chunker import BaseChunker
from core.generator import Generator
from core.schema import Document
from pipelines.base_pipeline.pipeline import BasePipeline
from pipelines.medical_pipeline.medical_chunker import MedicalChunker
from pipelines.medical_pipeline.medical_embeddings import build_medical_embedder
from pipelines.medical_pipeline.medical_ner import MedicalEntities, MedicalNER
from utils import prompt_templates as pt


class MedicalPipeline(BasePipeline):
    name = "medical"
    qa_template = pt.MEDICAL_QA_TEMPLATE
    summary_template = pt.MEDICAL_SUMMARY_TEMPLATE

    def __init__(self, config: AppConfig = DEFAULT_CONFIG,
                 generator: Optional[Generator] = None):
        self._ner = MedicalNER()
        self._entities: MedicalEntities = MedicalEntities()
        super().__init__(
            config=config,
            embedder=build_medical_embedder(),
            generator=generator,
        )

    # ------------------------------------------------------------------ #
    def _build_chunker(self) -> BaseChunker:
        # Entities are not known until _preprocess runs, so start with a base
        # medical chunker and rebuild it with entities during preprocessing.
        return MedicalChunker(self.config.chunking)

    def _preprocess(self, document: Document) -> Document:
        """Run NER, store entities, and make the chunker entity-aware."""
        self._entities = self._ner.extract(document.text)
        document.metadata["entities"] = self._entities.as_dict()
        document.metadata["ner_backend"] = (
            "fallback" if self._ner.is_fallback else "scispacy"
        )
        # Rebuild chunker now that we have entities to protect/tag.
        self.chunker = MedicalChunker(self.config.chunking, entities=self._entities)
        document.doc_type = "medical"
        return document

    # ------------------------------------------------------------------ #
    @property
    def entities(self) -> MedicalEntities:
        return self._entities

    @property
    def ner_backend(self) -> str:
        return "fallback" if self._ner.is_fallback else "scispacy"
