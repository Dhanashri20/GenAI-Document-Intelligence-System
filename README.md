# 📄 GenAI Document Intelligence System

A production-style, modular Retrieval-Augmented Generation (RAG) system that
ingests medical, scientific (LaTeX-heavy), and general documents, then
**summarizes** them and answers **grounded questions** — with a built-in
**hallucination checker** and a clean Streamlit UI. Everything runs on **CPU**.

> ⚕️ The medical mode is an **educational document-analysis assistant only**.
> It does not diagnose, recommend treatment, or claim medical certainty.

---

## Features

- **Multi-format ingestion** — PDF (PyMuPDF → pdfplumber fallback), TXT, LaTeX, Markdown, with math detection and section detection.
- **Three chunkers** — general semantic packer, entity-aware medical chunker, equation-atomic LaTeX chunker.
- **Switchable embeddings** — general (MiniLM), biomedical (S-PubMedBERT), scientific (SPECTER).
- **Vector store** — FAISS inner-product index (cosine) with a numpy brute-force fallback and metadata.
- **Hybrid retrieval** — dense + BM25 keyword fusion with re-ranking.
- **Grounded generation** — FLAN-T5 with anti-hallucination prompt templates.
- **Hallucination checker** — retrieval verification + answer groundedness + confidence scoring, with safe refusal.
- **Medical NER** — scispaCy (diseases, symptoms, medications, procedures) with lexicon fallback.
- **LaTeX explainer** — plain-English, step-by-step equation breakdowns.

Every external dependency degrades gracefully: if a model or library is missing,
a lightweight fallback keeps the full pipeline runnable.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# CPU torch (if not already installed):
pip install torch --index-url https://download.pytorch.org/whl/cpu

# (optional) scispaCy medical models:
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz

streamlit run app/streamlit_app.py
```

Open the local URL Streamlit prints, upload a document, pick a pipeline
(or **Auto-detect**), then **Summarize** or **Ask**.

---

## Project layout

```
project_root/
├── app/streamlit_app.py        # UI only (no dev docs rendered here)
├── config/settings.py          # models, RAG params, safety thresholds
├── core/                       # loader, chunker, embeddings, vectorstore, retriever, generator, schema
├── pipelines/
│   ├── base_pipeline/          # orchestration
│   ├── medical_pipeline/       # NER, chunker, embeddings, QA, summarizer
│   ├── latex_pipeline/         # equation-aware chunking + explainer
│   └── router.py               # factory + auto-detection
├── evaluation/hallucination_checker.py
├── utils/                      # prompt_templates, formatting
├── requirements.txt
└── DEVELOPER_GUIDE.md          # full architecture write-up
```

See **DEVELOPER_GUIDE.md** for the detailed architecture and extension guide.
