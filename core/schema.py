"""
Shared data structures passed between pipeline stages.

Keeping these in one place means the loader, chunker, vector store, retriever
and generator all speak the same language.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    """A loaded document, normalized to plain text plus structural metadata."""

    text: str
    source: str                              # filename or path
    doc_type: str = "general"                # general | medical | scientific
    pages: List[str] = field(default_factory=list)   # per-page text (PDFs)
    sections: List["Section"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_math(self) -> bool:
        return bool(self.metadata.get("has_math"))


@dataclass
class Section:
    """A detected logical section (heading + body)."""

    title: str
    body: str
    start_char: int = 0
    page: Optional[int] = None


@dataclass
class Chunk:
    """A retrievable unit of text with provenance metadata."""

    text: str
    chunk_id: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Populated by the vector store / retriever at query time.
    score: float = 0.0

    @property
    def page(self) -> Optional[int]:
        return self.metadata.get("page")

    @property
    def section(self) -> Optional[str]:
        return self.metadata.get("section")

    @property
    def chunk_type(self) -> str:
        return self.metadata.get("type", "text")
