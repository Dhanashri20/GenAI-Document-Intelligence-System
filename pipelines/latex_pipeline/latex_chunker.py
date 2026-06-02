"""
pipelines.latex_pipeline.latex_chunker
======================================

Equation-aware chunking. Mathematical content must survive intact: splitting a
formula across two chunks destroys its meaning and breaks retrieval. This
chunker:

1. Extracts equations (``$...$``, ``$$...$$``, ``\\[...\\]``, ``\\begin{env}...``)
   and replaces them with placeholders before sentence segmentation.
2. Treats each equation as an **atomic segment** — it is never split and is
   always packed whole, tagged with ``type="equation"``.
3. Restores equations after packing so chunk text contains the real math.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from config.settings import ChunkConfig
from core.chunker import BaseChunker
from core.schema import Chunk, Document


_EQUATION_PATTERNS = [
    r"\$\$.+?\$\$",
    r"\\\[.+?\\\]",
    r"\\begin\{(?:equation|align|gather|multline|cases|matrix)\*?\}.+?"
    r"\\end\{(?:equation|align|gather|multline|cases|matrix)\*?\}",
    r"\$[^$]+?\$",
]
_EQ_REGEX = re.compile("|".join(_EQUATION_PATTERNS), re.S)


class LatexChunker(BaseChunker):
    def __init__(self, config: Optional[ChunkConfig] = None):
        super().__init__(config)

    # ------------------------------------------------------------------ #
    def _extract_equations(self, text: str) -> Tuple[str, Dict[str, str]]:
        mapping: Dict[str, str] = {}

        def _sub(match: "re.Match") -> str:
            token = f" \x00EQ{len(mapping)}\x00 "
            mapping[token.strip()] = match.group(0)
            return token

        masked = _EQ_REGEX.sub(_sub, text)
        return masked, mapping

    def _restore(self, text: str, mapping: Dict[str, str]) -> str:
        for token, eq in mapping.items():
            text = text.replace(token, eq)
        return text

    # ------------------------------------------------------------------ #
    def _segment(self, text: str) -> List[str]:
        masked, mapping = self._extract_equations(text)
        base_segments = super()._segment(masked)

        segments: List[str] = []
        eq_tokens = set(mapping.keys())
        for seg in base_segments:
            # Pull standalone equations out as their own atomic segments.
            tokens_in_seg = [t for t in eq_tokens if t in seg]
            if not tokens_in_seg:
                segments.append(seg)
                continue
            # Split the segment around equation tokens, keeping equations whole.
            parts = re.split(r"(\x00EQ\d+\x00)", seg)
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                segments.append(self._restore(part, mapping))
        return segments

    # ------------------------------------------------------------------ #
    def chunk(self, document: Document) -> List[Chunk]:
        chunks = super().chunk(document)
        for c in chunks:
            if _EQ_REGEX.search(c.text):
                c.metadata["type"] = "equation"
                c.metadata["has_equation"] = True
            else:
                c.metadata.setdefault("type", "text")
        return chunks
