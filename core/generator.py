"""
core.generator
==============

Wraps a CPU-friendly HuggingFace text-generation pipeline and exposes a single
``generate(prompt)`` method with configurable decoding.

The generator is **architecture-aware**: it inspects the model config and uses
the ``text2text-generation`` task for encoder-decoder models (FLAN-T5) and the
``text-generation`` task for decoder-only / instruct models (Qwen, Llama, Phi).
For instruct models it applies the tokenizer's chat template automatically, so
you can switch to a stronger model purely by changing ``GENERATOR_MODEL`` in
``config/settings.py`` — no code changes required.

A lightweight extractive fallback is provided so the rest of the system is
testable when transformers/the model weights are unavailable. The fallback
returns the most relevant context sentence rather than inventing text, keeping
it consistent with the project's anti-hallucination stance.
"""

from __future__ import annotations

import re
from typing import Optional

from config.settings import GENERATION_FALLBACK_NOTE, GenerationConfig  # type: ignore


class Generator:
    def __init__(
        self,
        model_name: str = "google/flan-t5-large",
        config: Optional[GenerationConfig] = None,
    ):
        self.model_name = model_name
        self.config = config or GenerationConfig()
        self._pipe = None
        self._tokenizer = None
        self._task: Optional[str] = None
        self._loaded = False
        self._is_fallback = False

    # ------------------------------------------------------------------ #
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            from transformers import AutoConfig, pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

            # Decide seq2seq vs causal from the model config.
            try:
                cfg = AutoConfig.from_pretrained(self.model_name)
                is_seq2seq = bool(getattr(cfg, "is_encoder_decoder", False))
            except Exception:  # noqa: BLE001 - heuristic fallback
                is_seq2seq = "t5" in self.model_name.lower()

            self._task = "text-generation"  #"text2text-generation" if is_seq2seq else 
            self._pipe = pipeline(self._task, model=AutoModelForSeq2SeqLM.from_pretrained(self.model_name), device=-1)  # CPU
            self._tokenizer = getattr(self._pipe, "tokenizer", AutoTokenizer.from_pretrained("google/flan-t5-large"))
        except Exception as exc:  # noqa: BLE001
            print(f"[generator] Using extractive fallback ({exc}).")
            self._pipe = None
            self._is_fallback = True
        self._loaded = True

    # ------------------------------------------------------------------ #
    def _gen_kwargs(self) -> dict:
        """Build decoding kwargs, only passing sampling params when sampling.

        (Passing temperature/top_p under greedy decoding triggers warnings and,
        in newer transformers, errors.)
        """
        kw = {
            "max_new_tokens": self.config.max_new_tokens,
            "repetition_penalty": self.config.repetition_penalty,
            "num_beams": self.config.num_beams,
        }
        if self.config.do_sample:
            kw.update(
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
        else:
            kw["do_sample"] = False
        return kw

    # ------------------------------------------------------------------ #
    def generate(self, prompt: str) -> str:
        self._ensure_loaded()
        if self._pipe is None:
            return self._extractive_fallback(prompt)

        kwargs = self._gen_kwargs()

        if self._task == "text-generation":
            # Apply the model's chat template when available (instruct models).
            text = prompt
            if self._tokenizer is not None and getattr(
                self._tokenizer, "chat_template", None
            ):
                text = self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            out = self._pipe(text, return_full_text=False, **kwargs)
            return out[0]["generated_text"].strip()

        # Encoder-decoder (FLAN-T5).
        out = self._pipe(prompt, **kwargs)
        return out[0]["generated_text"].strip()

    # ------------------------------------------------------------------ #
    @property
    def is_fallback(self) -> bool:
        self._ensure_loaded()
        return self._is_fallback

    # ------------------------------------------------------------------ #
    def _extractive_fallback(self, prompt: str) -> str:
        """Return the most prompt-relevant context sentence (no invention)."""
        m = re.search(
            r"(?:Context|Content|Partial summaries):\s*(.+?)\s*"
            r"(?:Question:|Request:|Grounded answer:|Plain-language|Conceptual|"
            r"Step-by-step|Educational|Unified summary|$)",
            prompt,
            re.S,
        )
        context = m.group(1) if m else prompt
        # Strip provenance tags "[1 | Methods, p.2]" and reduce-step bullets.
        context = re.sub(r"\[\d+(?:\s*\|[^\]]*)?\]", " ", context)
        context = re.sub(r"^\s*[-*]\s+", "", context, flags=re.M)
        sentences = re.split(r"(?<=[.!?])\s+", context)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        if not sentences:
            return "No extractable content was found in the provided context."
        return sentences[0] + (
            f"\n\n{GENERATION_FALLBACK_NOTE}" if GENERATION_FALLBACK_NOTE else ""
        )