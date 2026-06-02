# Developer Guide

This document explains the system in depth. It is intentionally **separate from
the Streamlit app** and is never rendered in the UI.

---

## 1. System architecture

The system is a layered RAG application. Layers depend only on the layers below
them, which keeps modules swappable and testable.

```
            ┌──────────────────────────────┐
            │      app/streamlit_app.py     │   presentation (UI only)
            └──────────────┬───────────────┘
                           │
            ┌──────────────▼───────────────┐
            │          pipelines/           │   orchestration + domain logic
            │  base · medical · latex       │
            │          router.py            │
            └──────────────┬───────────────┘
                           │
   ┌───────────────────────┼─────────────────────────┐
   │                       │                          │
┌──▼──────┐  ┌─────────────▼─────────┐  ┌─────────────▼──────────┐
│  core/  │  │       utils/          │  │     evaluation/         │
│ loader  │  │  prompt_templates     │  │ hallucination_checker   │
│ chunker │  │  formatting           │  └─────────────────────────┘
│ embed.  │  └───────────────────────┘
│ vstore  │
│ retr.   │            ┌──────────────────────┐
│ gen.    │            │     config/settings   │  single source of truth
│ schema  │            └──────────────────────┘
└─────────┘
```

Two design principles drive the structure:

1. **One source of truth for configuration** (`config/settings.py`) — models,
   chunk sizes, retrieval weights and safety thresholds are all declared once.
2. **Graceful degradation** — every heavy dependency (sentence-transformers,
   transformers, FAISS, scispaCy, BM25, PDF backends) has a fallback so the
   pipeline always runs end to end, which makes it demoable and testable
   offline. Fallbacks are clearly logged.

---

## 2. Each module explained

### `config/settings.py`
Dataclasses for generation, chunking, retrieval and safety, plus the embedding
model registry and the medical disclaimer. Change a number here and the whole
system follows.

### `core/schema.py`
The shared vocabulary: `Document` (normalized text + pages + sections +
metadata), `Section`, and `Chunk` (text + provenance metadata + query-time
score). Every stage produces or consumes these.

### `core/loader.py`
Turns a file into a `Document`. PDF text is extracted with PyMuPDF (fallback
pdfplumber). It detects **math/LaTeX** content heuristically and splits text
into **sections** via heading regexes (numbered headings, ALL-CAPS headings,
known scientific/clinical section names). Whitespace is normalized so chunking
is stable.

### `core/chunker.py`
`BaseChunker` performs sentence-aware, **overlap-preserving** packing: text is
segmented into sentences, then greedily packed toward `chunk_size`, seeding each
new chunk with a trailing overlap from the previous one so context survives cut
points. Oversized atomic segments become their own chunk rather than being
split. Domain chunkers subclass this.

### `core/embeddings.py`
`EmbeddingManager` lazily loads and caches sentence-transformers models by
logical name (`general`/`biomedical`/`scientific`), normalizes vectors, and lets
you switch models at runtime. A deterministic hashing embedder is the offline
fallback.

### `core/vectorstore.py`
A FAISS `IndexFlatIP` over normalized vectors (so inner product == cosine), with
chunks stored in parallel for full provenance on search. Includes save/load and
a numpy brute-force fallback with identical semantics.

### `core/retriever.py`
Dense top-k search plus optional **hybrid fusion**: BM25 keyword scores and
dense similarities are min-max normalized and combined with configurable
weights, then re-ranked. Hybrid helps on rare tokens (drug names, gene symbols,
equation labels) that dense models can under-weight.

### `core/generator.py`
Wraps a CPU `text2text-generation` pipeline (FLAN-T5). Decoding params come from
config; defaults are greedy/low-temperature for factuality. The offline fallback
is **extractive** (returns the most relevant context sentence) — it never
invents text, consistent with the anti-hallucination stance.

### `utils/prompt_templates.py`
All prompt engineering, versioned in one place. Every grounded template tells
the model to answer **only from context** and to abstain otherwise. Includes
medical (safety-framed) and LaTeX (step-by-step) variants, plus `build_context`
which assembles retrieved chunks with `[n | section, p.X]` provenance labels.

### `utils/formatting.py`
Presentation helpers: confidence → label/color, chunk → source card, text
truncation and answer cleanup.

### `evaluation/hallucination_checker.py`
The safety gate (see §7).

### `pipelines/`
`BasePipeline` orchestrates everything; `MedicalPipeline` and `LatexPipeline`
override the chunker, embeddings and prompts; `router.py` builds pipelines by
name and auto-detects document type.

---

## 3. Data flow

**Index time** (when a document is processed):

```
file ─▶ loader.load_document ─▶ Document(text, pages, sections, has_math)
     ─▶ pipeline._preprocess (e.g. medical NER tagging)
     ─▶ chunker.chunk ─▶ [Chunk, …]
     ─▶ embeddings.embed ─▶ vectors
     ─▶ VectorStore.add(chunks, vectors)
     ─▶ Retriever(store, embedder)
```

**Query time** (Ask):

```
question ─▶ retriever.retrieve ─▶ top-k Chunks (scored)
        ─▶ build_context ─▶ prompt_template.format
        ─▶ generator.generate ─▶ raw answer
        ─▶ checker.gate(answer, chunks) ─▶ (final answer | refusal, report)
        ─▶ UI shows answer + confidence + sources
```

**Summarize** skips retrieval and feeds the leading chunks straight into the
summarization template (bounded for CPU cost).

---

## 4. RAG pipeline breakdown

1. **Chunking** decides retrieval granularity. Overlap preserves cross-boundary
   context; section metadata gives provenance.
2. **Embedding** maps chunks and the query into the same vector space. Choosing a
   domain-matched model materially improves recall.
3. **Vector store** does approximate/exact nearest-neighbor search. `fetch_k`
   candidates are pulled, then trimmed to `top_k` after (optional) re-ranking.
4. **Hybrid fusion** blends semantic and keyword evidence.
5. **Context assembly** packs the top chunks under a character budget with
   provenance tags, so the model (and the user) can trace claims.
6. **Generation** is constrained by templates to stay inside the context.
7. **Verification** gates the output before display.

Key knobs in `config/settings.py`: `chunk_size`, `chunk_overlap`, `top_k`,
`fetch_k`, `semantic_weight`/`keyword_weight`, `temperature`, `max_new_tokens`.

---

## 5. Medical pipeline design

`MedicalPipeline` differs from the base pipeline in four ways:

- **Biomedical embeddings** (`S-PubMedBERT`) so clinical synonyms cluster.
- **NER preprocessing** (`medical_ner.py`): scispaCy extracts diseases,
  symptoms, medications and procedures; results are stored on the document and
  surfaced in the UI. A lexicon/regex fallback runs if scispaCy is unavailable.
- **Entity-aware chunking** (`medical_chunker.py`): a sentence segmenter that
  respects medical abbreviations (`q.d.`, `mg/dL`, `Dr.`) so entities aren't
  split, and each chunk is tagged with the entities it contains.
- **Safety-framed prompts**: the medical QA/summary templates inject the
  disclaimer and forbid diagnosis, treatment advice and certainty, while
  instructing the model to **preserve uncertainty language** from the source.

The net effect: the medical mode behaves as an educational analysis tool, not a
clinician.

---

## 6. LaTeX handling strategy

`LatexChunker` guarantees equations are never broken:

1. **Mask** equations (`$…$`, `$$…$$`, `\[…\]`, `\begin{equation}…`) with
   placeholder tokens before sentence segmentation.
2. **Segment** the masked text normally.
3. **Split** segments around equation tokens and **restore** the real math, so
   each equation becomes its own atomic segment tagged `type="equation"`.

`LatexPipeline` adds:
- **Scientific embeddings** (SPECTER),
- `list_equations()` to enumerate detected formulas,
- `explain_equation()` which prioritizes equation-bearing chunks and uses the
  step-by-step explanation template (define each symbol, state what it computes,
  give intuition; admit when a symbol is undefined in context).

---

## 7. How the hallucination checker works

`HallucinationChecker.check(answer, chunks)` combines three signals:

1. **Retrieval verification** — the top chunk similarity must clear
   `min_retrieval_score`; otherwise nothing relevant was found and the system
   refuses immediately.
2. **Answer groundedness** — the answer is split into sentences; each sentence is
   embedded and compared (cosine) against all retrieved chunks; the per-sentence
   **max support** is averaged. This measures how much of the answer is actually
   backed by the context.
3. **Confidence** — a weighted blend (`0.4·retrieval + 0.6·groundedness`),
   clipped to [0, 1].

The answer is shown only if groundedness ≥ `min_groundedness` **and** confidence
≥ `min_confidence`. Otherwise `gate()` substitutes:

> "The answer cannot be confidently derived from the provided document."

Abstention phrases produced by the model (e.g. "Not stated in the document") are
detected and treated as safe refusals. The UI renders the confidence bar, the
reason, and the source chunks so the user can verify independently.

---

## 8. How to extend the system

- **New document type / pipeline**: subclass `BasePipeline`, override
  `_build_chunker`, `_preprocess`, and set `qa_template`/`summary_template`;
  register it in `pipelines/router.py` and add a default embedding in
  `PIPELINE_DEFAULT_EMBEDDING`.
- **New embedding model**: add an entry to `EMBEDDING_MODELS`; switch with
  `embedder.set_active(name)`.
- **Different LLM**: change `GENERATOR_MODEL` (any HF `text2text` or adapt
  `Generator` for causal models). Tune `GenerationConfig`.
- **Persistent index**: `VectorStore.save(prefix)` / `VectorStore.load(prefix)`
  already exist — wire them into the pipeline to cache embeddings between runs.
- **Reranker**: add a cross-encoder step in `Retriever._fuse` for higher
  precision at the cost of latency.
- **Stronger NER**: swap the scispaCy model in `medical_ner.py`, or add UMLS
  linking via `scispacy.linking`.
- **Evaluation**: extend `evaluation/` with faithfulness/answer-relevance metrics
  (e.g. RAGAS-style) for offline benchmarking.

---

## Testing

A dependency-free smoke test exercises the full flow with offline fallbacks:
load → chunk → embed → store → retrieve → generate → gate, across all three
pipelines, plus medical NER and equation extraction. Run any module's logic
directly by importing from the package root (`PYTHONPATH=.`).
