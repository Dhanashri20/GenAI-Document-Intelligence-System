"""
Central configuration for the Document Intelligence System.

Everything tunable lives here so the rest of the codebase reads from a single
source of truth. All models chosen below are CPU-friendly and download from the
public HuggingFace hub on first use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


# --------------------------------------------------------------------------- #
# Embedding model registry
# --------------------------------------------------------------------------- #
# Keyed by a short logical name used throughout the app. Each maps to a
# sentence-transformers compatible model id on the HuggingFace hub.
EMBEDDING_MODELS: Dict[str, str] = {
    # Small, fast, strong general-purpose model (384-dim).
    "general": "sentence-transformers/all-MiniLM-L6-v2",
    # Biomedical sentence embeddings tuned on PubMed.
    "biomedical": "pritamdeka/S-PubMedBert-MS-MARCO",
    # Scientific paper embeddings (title+abstract trained).
    "scientific": "allenai/specter",
}

# Maps a pipeline name -> default embedding key above.
PIPELINE_DEFAULT_EMBEDDING: Dict[str, str] = {
    "general": "general",
    "medical": "biomedical",
    "scientific": "scientific",
}


# --------------------------------------------------------------------------- #
# Generation (LLM) configuration
# --------------------------------------------------------------------------- #
# FLAN-T5 is an instruction-tuned seq2seq model that runs comfortably on CPU
# and is well suited to grounded QA and summarization. Swap to flan-t5-large
# if you have the RAM/time budget.
# Default generator. FLAN-T5-large is the best practical instruction-tuned
# seq2seq model for grounded RAG QA on CPU: markedly more accurate than -base,
# ungated, and reliable (it stays close to the context and rarely rambles).
#
# For even higher QA accuracy on CPU you can switch to a modern decoder-only
# instruct model below — the Generator auto-detects architecture and applies the
# chat template, so only this string needs to change (expect slower CPU speed):
#   GENERATOR_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # strong, ungated
#   GENERATOR_MODEL = "Qwen/Qwen2.5-3B-Instruct"     # stronger, heavier
GENERATOR_MODEL: str = "google/flan-t5-large"

# Appended when the extractive fallback generator is in use (no LLM weights).
GENERATION_FALLBACK_NOTE: str = (
    "(Extractive mode: language model unavailable; showing the most relevant "
    "source sentence.)"
)


@dataclass
class GenerationConfig:
    """Decoding parameters passed to the HuggingFace pipeline."""

    max_new_tokens: int = 256
    # Low temperature keeps answers grounded and reduces invention.
    temperature: float = 0.3
    do_sample: bool = False        # greedy by default -> deterministic, factual
    top_p: float = 0.95
    repetition_penalty: float = 1.15
    num_beams: int = 1


# --------------------------------------------------------------------------- #
# Chunking configuration
# --------------------------------------------------------------------------- #
@dataclass
class ChunkConfig:
    chunk_size: int = 900          # target characters per chunk
    chunk_overlap: int = 150       # overlap to preserve context across cuts
    min_chunk_size: int = 120      # drop fragments smaller than this


# --------------------------------------------------------------------------- #
# Retrieval configuration
# --------------------------------------------------------------------------- #
@dataclass
class RetrievalConfig:
    top_k: int = 5                 # chunks returned to the generator
    fetch_k: int = 20              # candidates pulled before re-ranking
    use_hybrid: bool = True        # blend semantic + BM25 keyword scores
    semantic_weight: float = 0.7   # weight of dense score in hybrid fusion
    keyword_weight: float = 0.3


# --------------------------------------------------------------------------- #
# Hallucination / groundedness thresholds
# --------------------------------------------------------------------------- #
@dataclass
class SafetyConfig:
    # Minimum top retrieval similarity for the question to be "answerable".
    min_retrieval_score: float = 0.25
    # Minimum semantic overlap between the answer and the retrieved context.
    min_groundedness: float = 0.35
    # Composite confidence below this triggers the safe-refusal message.
    min_confidence: float = 0.40
    refusal_message: str = (
        "The answer cannot be confidently derived from the provided document."
    )


# --------------------------------------------------------------------------- #
# Medical safety framing (injected into every medical prompt)
# --------------------------------------------------------------------------- #
MEDICAL_DISCLAIMER: str = (
    "This is an educational document-analysis assistant. It does not provide "
    "diagnosis, treatment advice, or medical certainty. Always consult a "
    "qualified healthcare professional."
)


@dataclass
class AppConfig:
    """Top-level bundle wired together at startup."""

    generation: GenerationConfig = field(default_factory=GenerationConfig)
    chunking: ChunkConfig = field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    generator_model: str = GENERATOR_MODEL


# A ready-to-use default instance.
DEFAULT_CONFIG = AppConfig()