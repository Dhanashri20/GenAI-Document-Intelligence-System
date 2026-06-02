"""
utils.prompt_templates
=======================

All prompt engineering lives here. Templates are plain ``str.format`` strings so
they are easy to read, version, and audit. Every grounded template instructs the
model to answer ONLY from the supplied context and to abstain when the context
is insufficient — this is the first line of defense against hallucination.
"""

from __future__ import annotations

from typing import List

from config.settings import MEDICAL_DISCLAIMER
from core.schema import Chunk


# --------------------------------------------------------------------------- #
# Context assembly
# --------------------------------------------------------------------------- #
def build_context(chunks: List[Chunk], max_chars: int = 3500) -> str:
    """Concatenate retrieved chunks with lightweight provenance labels."""
    parts: List[str] = []
    used = 0
    for i, c in enumerate(chunks, 1):
        loc = []
        if c.section:
            loc.append(c.section)
        if c.page is not None:
            loc.append(f"p.{c.page}")
        tag = f"[{i}{' | ' + ', '.join(loc) if loc else ''}]"
        piece = f"{tag} {c.text}"
        if used + len(piece) > max_chars:
            break
        parts.append(piece)
        used += len(piece)
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# General QA
# --------------------------------------------------------------------------- #
QA_TEMPLATE = (
    "You are a careful document analysis assistant. Answer the question using "
    "ONLY the context below. If the answer is not contained in the context, "
    "reply exactly: \"Not stated in the document.\" Do not use outside "
    "knowledge and do not guess.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Grounded answer:"
)

# --------------------------------------------------------------------------- #
# General summarization
# --------------------------------------------------------------------------- #
SUMMARY_TEMPLATE = (
    "Summarize the following document content in clear, plain language. "
    "Simplify complex wording while preserving the original meaning. Keep any "
    "hedging or uncertainty (e.g. 'may', 'suggests', 'is associated with') "
    "exactly as expressed. Do not add facts that are not present.\n\n"
    "Content:\n{context}\n\n"
    "Plain-language summary:"
)

# --------------------------------------------------------------------------- #
# Reduce step for map-reduce, full-document summarization
# --------------------------------------------------------------------------- #
REDUCE_SUMMARY_TEMPLATE = (
    "Below are partial summaries of different sections of a single document. "
    "Combine them into one coherent, non-repetitive plain-language summary that "
    "covers the whole document. Preserve any uncertainty or qualifying language "
    "(e.g. 'may', 'suggests'). Do not introduce facts that are not present in "
    "the partial summaries.\n\n"
    "Partial summaries:\n{partial_summaries}\n\n"
    "Unified summary:"
)

# --------------------------------------------------------------------------- #
# Medical QA (grounded + safety framed)
# --------------------------------------------------------------------------- #
MEDICAL_QA_TEMPLATE = (
    f"{MEDICAL_DISCLAIMER}\n\n"
    "Answer the question using ONLY the medical document context below. "
    "Do NOT diagnose, do NOT recommend treatment, and do NOT express medical "
    "certainty. Preserve uncertainty language from the source. If the context "
    "does not contain the answer, reply exactly: \"Not stated in the "
    "document.\"\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Educational, grounded answer:"
)

# --------------------------------------------------------------------------- #
# Medical summarization (jargon simplification)
# --------------------------------------------------------------------------- #
MEDICAL_SUMMARY_TEMPLATE = (
    f"{MEDICAL_DISCLAIMER}\n\n"
    "Summarize the following clinical/medical text for a non-specialist. "
    "Explain medical jargon in parentheses the first time it appears. Preserve "
    "all uncertainty and qualifying language. Do not infer diagnoses or "
    "outcomes that are not explicitly stated.\n\n"
    "Content:\n{context}\n\n"
    "Plain-language medical summary:"
)

# --------------------------------------------------------------------------- #
# LaTeX / equation explanation
# --------------------------------------------------------------------------- #
LATEX_EXPLAIN_TEMPLATE = (
    "You explain mathematics in plain English. Using ONLY the context below, "
    "explain the requested equation or concept step by step. Define each symbol, "
    "state what the equation computes, and give the intuition. If the context "
    "does not define a symbol, say so rather than guessing.\n\n"
    "Context:\n{context}\n\n"
    "Request: {question}\n\n"
    "Step-by-step plain-English explanation:"
)

LATEX_SUMMARY_TEMPLATE = (
    "Summarize the following scientific/mathematical content in plain English. "
    "Keep equation references intact (do not rewrite the math), but explain what "
    "each result means conceptually. Preserve any stated assumptions and "
    "limitations.\n\n"
    "Content:\n{context}\n\n"
    "Conceptual summary:"
)