"""End-to-end smoke test using offline fallbacks (no model downloads)."""
import sys
from core.loader import load_text
from pipelines.router import build_pipeline, detect_doc_type

SAMPLE = """ABSTRACT
This study examines the effect of metformin on patients with type 2 diabetes.
The drug reduced fasting glucose levels. Some patients reported nausea and fatigue.

METHODS
We enrolled 120 patients. A biopsy was performed in select cases. Results may
suggest an association, but causation was not established.

RESULTS
The treatment group showed lower glucose. The equation $E = mc^2$ is unrelated
but tests math detection. We also use $$\\int_0^1 x^2 dx = \\frac{1}{3}$$ here.
"""

print("detected type:", detect_doc_type(SAMPLE))

for name in ["general", "medical", "scientific"]:
    print(f"\n=== pipeline: {name} ===")
    doc = load_text(SAMPLE, source="sample.txt", doc_type=name)
    pipe = build_pipeline(name)
    n = pipe.index(doc)
    print(f"indexed {n} chunks; store size={pipe.store.size}; dim={pipe.embedder.dimension}")
    res = pipe.ask("What drug was studied and what side effects were reported?")
    print("answer:", res.answer[:160])
    print("confidence:", round(res.report.confidence, 3),
          "| grounded:", res.report.grounded,
          "| retrieval:", round(res.report.retrieval_score, 3))
    print("n_sources:", len(res.sources))
    summ = pipe.summarize()
    print("summary:", summ.summary[:120])

# medical entities + latex equation listing
med = build_pipeline("medical"); med.index(load_text(SAMPLE, doc_type="medical"))
print("\nmedical entities:", med.entities.as_dict(), "| backend:", med.ner_backend)

sci = build_pipeline("scientific"); sci.index(load_text(SAMPLE, doc_type="scientific"))
print("equations found:", sci.list_equations())
eq = sci.explain_equation("Explain the integral equation")
print("equation explanation:", eq.answer[:160])

# Test refusal path on an off-topic question
off = pipe.ask("What is the capital of France according to this document?")
print("\noff-topic answer:", off.answer[:120], "| grounded:", off.report.grounded)
print("\nSMOKE TEST PASSED")
