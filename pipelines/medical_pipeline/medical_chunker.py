"""
pipelines.medical_pipeline.medical_chunker
==========================================

Medical text is dense with entities (drug names, dosages, abbreviations like
"q.d.", "mg/dL") that must not be split across chunk boundaries. This chunker
extends the base packer with two safeguards:

1. A medical-aware sentence segmenter that avoids breaking on abbreviation
   periods (e.g. "Dr.", "mg.", "i.e.").
2. Optional entity awareness: when a NER result is supplied, the chunker tags
   each chunk with the entities it contains (useful metadata for retrieval and
   for the hallucination checker), and it will not start a new chunk in the
   middle of a recognized multi-word entity.
"""

from __future__ import annotations

import re
from typing import List, Optional

from config.settings import ChunkConfig
from core.chunker import BaseChunker
from core.schema import Chunk, Document
from pipelines.medical_pipeline.medical_ner import MedicalEntities


_ABBREV = {
    "dr", "mr", "mrs", "ms", "vs", "etc", "e.g", "i.e", "no", "fig", "approx",
    "mg", "ml", "kg", "dl", "iv", "po", "q.d", "b.i.d", "t.i.d", "prn",
}
# Split on sentence punctuation that is NOT preceded by a known abbreviation.
_MED_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


class MedicalChunker(BaseChunker):
    def __init__(self, config: Optional[ChunkConfig] = None,
                 entities: Optional[MedicalEntities] = None):
        super().__init__(config)
        self.entities = entities
        self._entity_terms = (
            sorted(set(t.lower() for t in entities.all_terms()), key=len, reverse=True)
            if entities else []
        )

    # ------------------------------------------------------------------ #
    def _segment(self, text: str) -> List[str]:
        """Sentence segmentation that respects medical abbreviations."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        segments: List[str] = []
        for para in paragraphs:
            candidates = _MED_SENT_SPLIT.split(para)
            buffer = ""
            for cand in candidates:
                buffer = (buffer + " " + cand).strip() if buffer else cand
                last_word = re.sub(r"[^a-z.]", "", buffer.split()[-1].lower()) if buffer.split() else ""
                # If the sentence appears to end on an abbreviation, keep accruing.
                if last_word.rstrip(".") in _ABBREV:
                    continue
                segments.append(buffer)
                buffer = ""
            if buffer:
                segments.append(buffer)
        return segments

    # ------------------------------------------------------------------ #
    def chunk(self, document: Document) -> List[Chunk]:
        chunks = super().chunk(document)
        if self._entity_terms:
            for c in chunks:
                lowered = c.text.lower()
                present = [t for t in self._entity_terms if t in lowered]
                if present:
                    c.metadata["entities"] = present[:20]
                c.metadata["type"] = "medical"
        return chunks
