"""
pipelines.medical_pipeline.medical_summarizer
=============================================

Standalone helper for plain-language medical summarization, decoupled from the
pipeline so it can be reused (e.g. batch jobs, tests). The ``MedicalPipeline``
uses the same template via its ``summarize`` method; this module exposes a
function for direct use.
"""

from __future__ import annotations

from typing import List

from core.generator import Generator
from core.schema import Chunk
from utils import prompt_templates as pt
from utils.formatting import clean_answer


def summarize_medical(chunks: List[Chunk], generator: Generator) -> str:
    """Produce a jargon-simplified, uncertainty-preserving medical summary."""
    context = pt.build_context(chunks, max_chars=4000)
    prompt = pt.MEDICAL_SUMMARY_TEMPLATE.format(context=context)
    return clean_answer(generator.generate(prompt))
