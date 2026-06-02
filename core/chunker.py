"""
core.chunker
============

The general-purpose chunker. It performs sentence-aware, overlap-preserving
"semantic" splitting: text is first broken at paragraph/sentence boundaries,
then those units are greedily packed into chunks near a target size with a
configurable character overlap so context is not lost across cut points.

Specialized chunkers (medical, LaTeX) subclass ``BaseChunker`` and override
``_segment`` or ``chunk`` while reusing the packing logic here.
"""

from __future__ import annotations

import re
from typing import List

from config.settings import ChunkConfig
from core.schema import Chunk, Document, Section


# Sentence splitter that respects common abbreviations only loosely. For a
# production system you might swap in spaCy's sentencizer; this keeps the core
# dependency-free.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


class BaseChunker:
    """Semantic packing chunker shared by all pipelines."""

    def __init__(self, config: ChunkConfig | None = None):
        self.config = config or ChunkConfig()

    # ------------------------------------------------------------------ #
    # Overridable: how raw text becomes atomic segments before packing.
    # ------------------------------------------------------------------ #
    def _segment(self, text: str) -> List[str]:
        """Split text into sentence-level atomic segments."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        segments: List[str] = []
        for para in paragraphs:
            sents = _SENT_SPLIT.split(para)
            segments.extend(s.strip() for s in sents if s.strip())
        return segments

    # ------------------------------------------------------------------ #
    # Greedy packer with overlap. Atomic segments are never split.
    # ------------------------------------------------------------------ #
    def _pack(self, segments: List[str], base_meta: dict) -> List[Chunk]:
        chunks: List[Chunk] = []
        cur: List[str] = []
        cur_len = 0
        cid = base_meta.get("_next_id", 0)

        def flush():
            nonlocal cur, cur_len, cid
            if not cur:
                return
            text = " ".join(cur).strip()
            if len(text) >= self.config.min_chunk_size or not chunks:
                meta = {k: v for k, v in base_meta.items() if not k.startswith("_")}
                chunks.append(Chunk(text=text, chunk_id=cid, metadata=meta))
                cid += 1
            cur, cur_len = [], 0

        for seg in segments:
            seg_len = len(seg)
            # A single oversized segment becomes its own chunk.
            if seg_len > self.config.chunk_size:
                flush()
                meta = {k: v for k, v in base_meta.items() if not k.startswith("_")}
                chunks.append(Chunk(text=seg, chunk_id=cid, metadata=meta))
                cid += 1
                continue

            if cur_len + seg_len > self.config.chunk_size and cur:
                flush()
                # Re-seed the next chunk with trailing overlap for continuity.
                overlap = self._tail_overlap(chunks[-1].text)
                if overlap:
                    cur.append(overlap)
                    cur_len += len(overlap)
            cur.append(seg)
            cur_len += seg_len + 1

        flush()
        base_meta["_next_id"] = cid
        return chunks

    def _tail_overlap(self, text: str) -> str:
        """Return the last ``chunk_overlap`` characters at a word boundary."""
        if self.config.chunk_overlap <= 0:
            return ""
        tail = text[-self.config.chunk_overlap:]
        # snap to the start of a word so we don't begin mid-token
        space = tail.find(" ")
        return tail[space + 1:] if space != -1 else tail

    # ------------------------------------------------------------------ #
    # Public entrypoint
    # ------------------------------------------------------------------ #
    def chunk(self, document: Document) -> List[Chunk]:
        """Chunk a document section-by-section, preserving section metadata."""
        all_chunks: List[Chunk] = []
        shared = {"_next_id": 0, "source": document.source, "doc_type": document.doc_type}

        sections: List[Section] = document.sections or [
            Section(title="Document", body=document.text)
        ]
        for section in sections:
            shared["section"] = section.title
            shared["page"] = section.page
            segments = self._segment(section.body)
            all_chunks.extend(self._pack(segments, shared))

        return all_chunks
