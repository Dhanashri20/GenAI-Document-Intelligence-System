"""
evaluation.hallucination_checker
================================

The safety gate that sits between generation and the user. It produces a
``GroundednessReport`` and decides whether the answer may be shown or must be
replaced with the safe-refusal message.

Three signals are combined:

1. **Retrieval verification** — was anything sufficiently similar to the
   question actually retrieved? (uses the top chunk's similarity score)
2. **Answer groundedness** — how much of the answer is semantically supported by
   the retrieved context? We embed the answer sentences and the context chunks
   and measure max cosine similarity per answer sentence, then average.
3. **Confidence score** — a weighted blend of the two, plus penalties for
   refusal-style or empty answers.

If confidence falls below ``SafetyConfig.min_confidence`` (or either hard
threshold is breached) the answer is suppressed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

import numpy as np

from config.settings import SafetyConfig
from core.embeddings import EmbeddingManager
from core.schema import Chunk


_ABSTAIN_PATTERNS = re.compile(
    r"not stated in the document|cannot be (confidently )?derived|no .*found|"
    r"i (don't|do not) know|insufficient (context|information)",
    re.IGNORECASE,
)


@dataclass
class GroundednessReport:
    grounded: bool
    confidence: float
    retrieval_score: float
    groundedness_score: float
    per_sentence: List[float] = field(default_factory=list)
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "grounded": self.grounded,
            "confidence": round(self.confidence, 3),
            "retrieval_score": round(self.retrieval_score, 3),
            "groundedness_score": round(self.groundedness_score, 3),
            "reason": self.reason,
        }


class HallucinationChecker:
    def __init__(self, embedder: EmbeddingManager, config: SafetyConfig | None = None):
        self.embedder = embedder
        self.config = config or SafetyConfig()

    # ------------------------------------------------------------------ #
    def check(self, answer: str, chunks: List[Chunk]) -> GroundednessReport:
        retrieval_score = max((c.score for c in chunks), default=0.0)

        # If the model itself abstained, treat as grounded refusal (safe).
        if _ABSTAIN_PATTERNS.search(answer or ""):
            return GroundednessReport(
                grounded=False,
                confidence=0.0,
                retrieval_score=retrieval_score,
                groundedness_score=0.0,
                reason="Model abstained or no answer present.",
            )

        # Hard gate 1: nothing relevant retrieved.
        if not chunks or retrieval_score < self.config.min_retrieval_score:
            return GroundednessReport(
                grounded=False,
                confidence=0.0,
                retrieval_score=retrieval_score,
                groundedness_score=0.0,
                reason="No sufficiently relevant context retrieved.",
            )

        groundedness, per_sent = self._groundedness(answer, chunks)

        # Composite confidence: retrieval quality gates, groundedness dominates.
        confidence = float(np.clip(0.4 * retrieval_score + 0.6 * groundedness, 0, 1))

        grounded = (
            groundedness >= self.config.min_groundedness
            and confidence >= self.config.min_confidence
        )
        reason = (
            "Answer is supported by retrieved context."
            if grounded
            else "Answer not sufficiently supported by retrieved context."
        )
        return GroundednessReport(
            grounded=grounded,
            confidence=confidence,
            retrieval_score=retrieval_score,
            groundedness_score=groundedness,
            per_sentence=per_sent,
            reason=reason,
        )

    # ------------------------------------------------------------------ #
    def _groundedness(self, answer: str, chunks: List[Chunk]):
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 8]
        if not sentences:
            return 0.0, []

        ans_vecs = self.embedder.embed(sentences)
        ctx_vecs = self.embedder.embed([c.text for c in chunks])

        # cosine sims (vectors already L2-normalized) -> max support per sentence
        sims = ans_vecs @ ctx_vecs.T            # (n_sent, n_ctx)
        per_sentence = sims.max(axis=1).tolist()
        return float(np.mean(per_sentence)), per_sentence

    # ------------------------------------------------------------------ #
    def gate(self, answer: str, chunks: List[Chunk]):
        """Return (final_answer, report). Substitutes refusal text if ungrounded."""
        report = self.check(answer, chunks)
        if not report.grounded:
            return self.config.refusal_message, report
        return answer, report
