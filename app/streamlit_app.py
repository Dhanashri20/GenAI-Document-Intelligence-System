"""
app.streamlit_app
=================

Aesthetic Streamlit front-end for the Document Intelligence System.

Run with:
    streamlit run app/streamlit_app.py

Note: this file is intentionally UI-only. All intelligence lives in the
``core`` / ``pipelines`` / ``evaluation`` packages. No developer documentation
is rendered here (see DEVELOPER_GUIDE.md).
"""

from __future__ import annotations

import os
import sys
import tempfile

# Make project root importable when launched via `streamlit run app/...`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st  # noqa: E402

from config.settings import MEDICAL_DISCLAIMER  # noqa: E402
from core.loader import load_document, load_text  # noqa: E402
from pipelines.router import build_pipeline, detect_doc_type  # noqa: E402
from utils.formatting import confidence_label, format_sources  # noqa: E402


# --------------------------------------------------------------------------- #
# Page + theme
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Document Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
  --bg: #f7f7fb;
  --card: #ffffff;
  --ink: #2b2d42;
  --muted: #8a8d9f;
  --accent: #6c63ff;
  --accent-soft: #eceaff;
  --ok: #2e7d6f;
  --radius: 18px;
}
html, body, [class*="css"] { font-family: 'Inter','Segoe UI',sans-serif; color: var(--ink); }
.stApp { background: var(--bg); }
.block-container { padding-top: 2rem; max-width: 1100px; }
h1, h2, h3 { letter-spacing: -0.02em; }

.di-hero {
  background: linear-gradient(135deg, #6c63ff 0%, #8e85ff 100%);
  color: white; padding: 1.6rem 2rem; border-radius: var(--radius);
  margin-bottom: 1.2rem; box-shadow: 0 10px 30px rgba(108,99,255,.25);
}
.di-hero h1 { margin: 0; font-size: 1.7rem; }
.di-hero p { margin: .35rem 0 0; opacity: .9; font-size: .95rem; }

.di-card {
  background: var(--card); border-radius: var(--radius); padding: 1.3rem 1.5rem;
  box-shadow: 0 4px 18px rgba(43,45,66,.06); margin-bottom: 1rem;
  border: 1px solid #eef0f6;
}
.di-answer { font-size: 1.05rem; line-height: 1.65; }
.di-pill {
  display:inline-block; padding:.25rem .8rem; border-radius:999px;
  font-size:.78rem; font-weight:600; background:var(--accent-soft); color:var(--accent);
}
.di-source {
  background:#fafbff; border:1px solid #eef0f6; border-radius:14px;
  padding:.8rem 1rem; margin-bottom:.6rem; font-size:.88rem; color:#444;
}
.di-source-meta { color: var(--muted); font-size:.76rem; margin-bottom:.3rem; }
.di-conf-bar { height:10px; border-radius:999px; background:#eceff5; overflow:hidden; }
.di-conf-fill { height:100%; border-radius:999px; }
.di-disclaimer {
  background:#fff8e6; border:1px solid #ffe6a3; color:#8a6d1f;
  border-radius:14px; padding:.7rem 1rem; font-size:.82rem; margin-bottom:1rem;
}
.stButton>button {
  border-radius:14px; border:none; padding:.6rem 1.1rem; font-weight:600;
  background:var(--accent); color:white; transition:all .15s ease;
}
.stButton>button:hover { transform:translateY(-1px); box-shadow:0 6px 16px rgba(108,99,255,.3); }
.di-entity { display:inline-block; background:var(--accent-soft); color:var(--accent);
  padding:.15rem .6rem; border-radius:999px; font-size:.75rem; margin:.15rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def _init_state():
    st.session_state.setdefault("pipeline", None)
    st.session_state.setdefault("indexed", False)
    st.session_state.setdefault("pipeline_name", None)
    st.session_state.setdefault("doc_name", None)
    st.session_state.setdefault("n_chunks", 0)


_init_state()


def render_confidence(score: float):
    info = confidence_label(score)
    pct = int(round(score * 100))
    st.markdown(
        f"<div style='display:flex;justify-content:space-between;"
        f"margin-bottom:.3rem;'><span style='font-weight:600'>Confidence</span>"
        f"<span style='color:{info['color']};font-weight:700'>{info['emoji']} "
        f"{info['label']} · {pct}%</span></div>"
        f"<div class='di-conf-bar'><div class='di-conf-fill' "
        f"style='width:{pct}%;background:{info['color']}'></div></div>",
        unsafe_allow_html=True,
    )


def render_sources(sources):
    if not sources:
        st.caption("No sources retrieved.")
        return
    st.markdown("##### 📚 Sources")
    for s in format_sources(sources):
        st.markdown(
            f"<div class='di-source'><div class='di-source-meta'>"
            f"Source {s['index']} · {s['locator']} · similarity {s['score']}</div>"
            f"{s['preview']}</div>",
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# Sidebar — configuration
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    pipeline_choice = st.selectbox(
        "Pipeline",
        ["Auto-detect", "General", "Medical", "Scientific / LaTeX"],
        help="Choose how the document is processed, or let the system detect it.",
    )
    top_k = st.slider("Chunks to retrieve (top-k)", 2, 10, 5)
    st.divider()
    st.caption(
        "Embeddings, chunking and the LLM run on CPU. First run downloads models "
        "from the HuggingFace hub; offline, the app uses lightweight fallbacks."
    )


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown(
    "<div class='di-hero'><h1>📄 Document Intelligence</h1>"
    "<p>Understand, summarize and ask grounded questions about medical, "
    "scientific and general documents — with hallucination checking.</p></div>",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Upload + index
# --------------------------------------------------------------------------- #
with st.container():
    st.markdown("<div class='di-card'>", unsafe_allow_html=True)
    st.markdown("#### 1 · Upload a document")
    uploaded = st.file_uploader(
        "PDF, TXT, or LaTeX/Markdown", type=["pdf", "txt", "tex", "md"],
        label_visibility="collapsed",
    )
    pasted = st.text_area("…or paste text", height=120, placeholder="Paste document text here")

    if st.button("Process document", use_container_width=True):
        if not uploaded and not pasted.strip():
            st.warning("Please upload a file or paste some text first.")
        else:
            with st.spinner("Loading, chunking and embedding…"):
                # Build the document
                if uploaded:
                    suffix = os.path.splitext(uploaded.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded.read())
                        tmp_path = tmp.name
                    raw_for_detect = ""
                    try:
                        document = load_document(tmp_path)
                        raw_for_detect = document.text
                        doc_name = uploaded.name
                    finally:
                        os.unlink(tmp_path)
                else:
                    document = load_text(pasted)
                    raw_for_detect = pasted
                    doc_name = "pasted_text"

                # Resolve pipeline
                mapping = {
                    "General": "general", "Medical": "medical",
                    "Scientific / LaTeX": "scientific",
                }
                if pipeline_choice == "Auto-detect":
                    resolved = detect_doc_type(raw_for_detect)
                else:
                    resolved = mapping[pipeline_choice]

                pipe = build_pipeline(resolved)
                n = pipe.index(document)

                st.session_state.update(
                    pipeline=pipe, indexed=True, pipeline_name=resolved,
                    doc_name=doc_name, n_chunks=n,
                )
            st.success(
                f"Indexed **{doc_name}** with the **{resolved}** pipeline "
                f"({n} chunks)."
            )
    st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Interaction
# --------------------------------------------------------------------------- #
if st.session_state.indexed:
    pipe = st.session_state.pipeline
    name = st.session_state.pipeline_name

    st.markdown(
        f"<span class='di-pill'>Active pipeline: {name}</span> "
        f"<span class='di-pill'>{st.session_state.n_chunks} chunks</span>",
        unsafe_allow_html=True,
    )

    if name == "medical":
        st.markdown(f"<div class='di-disclaimer'>⚠️ {MEDICAL_DISCLAIMER}</div>",
                    unsafe_allow_html=True)
        ents = getattr(pipe, "entities", None)
        if ents and ents.all_terms():
            with st.expander("🧬 Extracted medical entities", expanded=False):
                for cat, terms in ents.as_dict().items():
                    if terms:
                        chips = "".join(f"<span class='di-entity'>{t}</span>" for t in terms)
                        st.markdown(f"**{cat.title()}** {chips}", unsafe_allow_html=True)

    if name == "scientific":
        eqs = pipe.list_equations() if hasattr(pipe, "list_equations") else []
        if eqs:
            with st.expander("📐 Detected equations", expanded=False):
                for e in eqs:
                    st.code(e, language="latex")

    col_a, col_b = st.columns(2)

    # ---- Summarize ----
    with col_a:
        st.markdown("<div class='di-card'>", unsafe_allow_html=True)
        st.markdown("#### 2 · Summarize")
        if st.button("Summarize document", use_container_width=True):
            with st.spinner("Generating plain-language summary…"):
                result = pipe.summarize()
            st.markdown(f"<div class='di-answer'>{result.summary}</div>",
                        unsafe_allow_html=True)
            st.caption(f"Based on {result.sections_covered} leading chunks.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- Ask ----
    with col_b:
        st.markdown("<div class='di-card'>", unsafe_allow_html=True)
        st.markdown("#### 3 · Ask a question")
        question = st.text_input("Your question", placeholder="e.g. What were the main findings?")
        explain_mode = (
            name == "scientific"
            and st.checkbox("Explain as an equation (step by step)")
        )
        if st.button("Ask", use_container_width=True):
            if not question.strip():
                st.warning("Type a question first.")
            else:
                with st.spinner("Retrieving context and generating a grounded answer…"):
                    if explain_mode and hasattr(pipe, "explain_equation"):
                        result = pipe.explain_equation(question, top_k=top_k)
                    else:
                        result = pipe.ask(question, top_k=top_k)
                st.markdown(f"<div class='di-answer'>{result.answer}</div>",
                            unsafe_allow_html=True)
                st.write("")
                render_confidence(result.report.confidence)
                st.caption(result.report.reason)
                st.write("")
                render_sources(result.sources)
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Upload or paste a document above, then choose **Process document** to begin.")
