"""
utils.formatting
================

Presentation helpers used by the UI and CLI: turning retrieved chunks into
readable source cards, mapping a numeric confidence to a label/color, and small
text-cleaning utilities.
"""

from __future__ import annotations

from typing import Dict, List

from core.schema import Chunk


def confidence_label(score: float) -> Dict[str, str]:
    """Map a 0–1 confidence into a label and a soft color for the UI."""
    if score >= 0.70:
        return {"label": "High", "color": "#2e7d6f", "emoji": "🟢"}
    if score >= 0.45:
        return {"label": "Moderate", "color": "#b8860b", "emoji": "🟡"}
    return {"label": "Low", "color": "#c0564b", "emoji": "🔴"}


def format_source(chunk: Chunk, index: int) -> Dict[str, str]:
    """Structured source-card data for rendering."""
    locator_bits = []
    if chunk.section:
        locator_bits.append(chunk.section)
    if chunk.page is not None:
        locator_bits.append(f"page {chunk.page}")
    if chunk.chunk_type and chunk.chunk_type != "text":
        locator_bits.append(chunk.chunk_type)
    return {
        "index": str(index),
        "locator": " · ".join(locator_bits) or "document",
        "score": f"{chunk.score:.2f}",
        "preview": truncate(chunk.text, 400),
    }


def format_sources(chunks: List[Chunk]) -> List[Dict[str, str]]:
    return [format_source(c, i) for i, c in enumerate(chunks, 1)]


def truncate(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def clean_answer(text: str) -> str:
    """Tidy model output: strip dangling labels and excess whitespace."""
    text = text.strip()
    for prefix in ("Answer:", "Grounded answer:", "Summary:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    return text
