"""
pipelines.router
================

Factory that builds the right pipeline by name, plus a heuristic auto-detector
that guesses a document's domain from its text so the UI can offer "Auto".
"""

from __future__ import annotations

import re
from typing import Optional

from config.settings import AppConfig, DEFAULT_CONFIG
from core.generator import Generator
from pipelines.base_pipeline.pipeline import BasePipeline
from pipelines.latex_pipeline.latex_pipeline import LatexPipeline
from pipelines.medical_pipeline.medical_qa import MedicalPipeline


_MEDICAL_HINTS = re.compile(
    r"\b(patient|clinical|diagnos|symptom|mg/dl|dosage|treatment|disease|"
    r"prognosis|histolog|biopsy|carcinoma|mmhg|comorbid)\b",
    re.IGNORECASE,
)
_MATH_HINTS = re.compile(
    r"\$.+?\$|\\begin\{(equation|align)|\\frac|\\sum|theorem|lemma|proof",
    re.IGNORECASE | re.S,
)


def detect_doc_type(text: str) -> str:
    med = len(_MEDICAL_HINTS.findall(text))
    math = len(_MATH_HINTS.findall(text))
    if med >= 3 and med >= math:
        return "medical"
    if math >= 2:
        return "scientific"
    return "general"


def build_pipeline(
    name: str,
    config: AppConfig = DEFAULT_CONFIG,
    generator: Optional[Generator] = None,
) -> BasePipeline:
    name = name.lower()
    if name == "medical":
        return MedicalPipeline(config=config, generator=generator)
    if name in ("scientific", "latex"):
        return LatexPipeline(config=config, generator=generator)
    return BasePipeline(config=config, generator=generator)
