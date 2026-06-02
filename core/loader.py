"""
core.loader
===========

Robust ingestion of PDFs and text files into the normalized ``Document`` schema.

Design choices
--------------
* PDF text is extracted with PyMuPDF (``fitz``) which is fast and preserves
  reading order well. ``pdfplumber`` is used as a fallback if PyMuPDF is absent.
* Math-awareness: we detect LaTeX / mathematical content heuristically so the
  router can pick the LaTeX-aware chunker downstream.
* Section detection uses lightweight regex over common heading patterns
  (numbered sections, ALL-CAPS headings, and known scientific section names).
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple

from core.schema import Document, Section


# --------------------------------------------------------------------------- #
# Heuristics for math / LaTeX detection
# --------------------------------------------------------------------------- #
_MATH_PATTERNS = [
    r"\$.+?\$",                       # inline $...$
    r"\$\$.+?\$\$",                   # display $$...$$
    r"\\begin\{(equation|align|cases|matrix|gather)\}",
    r"\\frac\{",
    r"\\sum_", r"\\int_", r"\\prod_",
    r"\\alpha|\\beta|\\gamma|\\theta|\\lambda|\\sigma|\\mu|\\pi",
    r"[∑∫∏√≈≠≤≥∂∇αβγθλσμπ]",          # unicode math glyphs
]
_MATH_REGEX = re.compile("|".join(_MATH_PATTERNS))

# Common scientific section headings used to assist detection.
_KNOWN_SECTIONS = {
    "abstract", "introduction", "background", "related work", "methods",
    "methodology", "materials and methods", "results", "discussion",
    "conclusion", "conclusions", "references", "acknowledgements",
    # clinical
    "history", "findings", "impression", "assessment", "plan", "diagnosis",
}


def detect_math(text: str) -> bool:
    """Return True if the text appears to contain mathematical / LaTeX content."""
    return bool(_MATH_REGEX.search(text))


# --------------------------------------------------------------------------- #
# Section detection
# --------------------------------------------------------------------------- #
_HEADING_REGEX = re.compile(
    r"^\s*("
    r"(?:\d{1,2}(?:\.\d{1,2})*\s+[A-Z][^\n]{2,60})"   # "3.1 Methods"
    r"|(?:[A-Z][A-Z \-]{3,60})"                       # "RESULTS"
    r")\s*$",
    re.MULTILINE,
)


def detect_sections(text: str) -> List[Section]:
    """Split text into logical sections based on heading patterns."""
    matches = list(_HEADING_REGEX.finditer(text))
    sections: List[Section] = []

    if not matches:
        return [Section(title="Document", body=text.strip(), start_char=0)]

    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        # Keep a section even if short, but skip empty bodies.
        if body:
            sections.append(Section(title=title, body=body, start_char=m.start()))

    # Capture any preamble before the first heading (often the abstract/title).
    first_start = matches[0].start()
    if first_start > 40:
        sections.insert(
            0, Section(title="Preamble", body=text[:first_start].strip(), start_char=0)
        )
    return sections


# --------------------------------------------------------------------------- #
# PDF extraction backends
# --------------------------------------------------------------------------- #
def _extract_pdf_pymupdf(path: str) -> Optional[List[str]]:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return None
    pages: List[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            # "text" mode keeps a reasonable reading order for most PDFs.
            pages.append(page.get_text("text"))
    return pages


def _extract_pdf_pdfplumber(path: str) -> Optional[List[str]]:
    try:
        import pdfplumber
    except ImportError:
        return None
    pages: List[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return pages


def _extract_pdf(path: str) -> List[str]:
    """Try PyMuPDF first, then pdfplumber. Raise if neither is available."""
    for backend in (_extract_pdf_pymupdf, _extract_pdf_pdfplumber):
        pages = backend(path)
        if pages is not None:
            return pages
    raise RuntimeError(
        "No PDF backend available. Install 'pymupdf' or 'pdfplumber'."
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_document(
    path: str,
    doc_type: str = "general",
) -> Document:
    """
    Load a file from disk into a normalized ``Document``.

    Parameters
    ----------
    path : str
        Path to a .pdf or .txt file.
    doc_type : str
        Logical document type ("general", "medical", "scientific"). This is a
        hint used downstream for chunker / embedding selection.
    """
    ext = os.path.splitext(path)[1].lower()
    source = os.path.basename(path)

    if ext == ".pdf":
        pages = _extract_pdf(path)
        full_text = "\n\n".join(pages)
    elif ext in (".txt", ".tex", ".md"):
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            full_text = fh.read()
        pages = [full_text]
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    full_text = _normalize_whitespace(full_text)
    sections = detect_sections(full_text)
    has_math = detect_math(full_text)

    return Document(
        text=full_text,
        source=source,
        doc_type=doc_type,
        pages=pages,
        sections=sections,
        metadata={
            "has_math": has_math,
            "n_pages": len(pages),
            "n_sections": len(sections),
            "ext": ext,
        },
    )


def load_text(text: str, source: str = "pasted_text", doc_type: str = "general") -> Document:
    """Build a ``Document`` directly from a string (e.g. pasted into the UI)."""
    text = _normalize_whitespace(text)
    return Document(
        text=text,
        source=source,
        doc_type=doc_type,
        pages=[text],
        sections=detect_sections(text),
        metadata={"has_math": detect_math(text), "n_pages": 1},
    )


def _normalize_whitespace(text: str) -> str:
    """Collapse excessive blank lines and trailing spaces, keep paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
