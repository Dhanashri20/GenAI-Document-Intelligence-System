"""
pipelines.medical_pipeline.medical_ner
======================================

Biomedical named-entity recognition. Primary backend is scispaCy
(``en_ner_bc5cdr_md`` for diseases/chemicals plus the base ``en_core_sci_sm``
parser). If scispaCy or its models are not installed, a regex/keyword fallback
returns coarse matches so the pipeline still annotates chunks.

Entities are grouped into: diseases, symptoms, medications, procedures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class MedicalEntities:
    diseases: List[str] = field(default_factory=list)
    symptoms: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    procedures: List[str] = field(default_factory=list)

    def all_terms(self) -> List[str]:
        return self.diseases + self.symptoms + self.medications + self.procedures

    def as_dict(self) -> Dict[str, List[str]]:
        return {
            "diseases": self.diseases,
            "symptoms": self.symptoms,
            "medications": self.medications,
            "procedures": self.procedures,
        }


# Lightweight fallback lexicons (illustrative, not exhaustive).
_FALLBACK = {
    "procedures": [
        "biopsy", "mri", "ct scan", "ultrasound", "x-ray", "endoscopy",
        "surgery", "resection", "angiography", "echocardiogram", "ecg", "eeg",
    ],
    "medications": [
        "aspirin", "metformin", "ibuprofen", "insulin", "warfarin", "statin",
        "amoxicillin", "paracetamol", "acetaminophen", "lisinopril",
    ],
    "symptoms": [
        "fever", "pain", "nausea", "fatigue", "cough", "headache", "dyspnea",
        "edema", "rash", "dizziness", "vomiting", "weight loss",
    ],
    "diseases": [
        "diabetes", "hypertension", "cancer", "carcinoma", "pneumonia",
        "asthma", "anemia", "sepsis", "stroke", "infarction", "tumor",
    ],
}


class MedicalNER:
    def __init__(self):
        self._nlp = None
        self._is_fallback = False
        self._load()

    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        try:
            import spacy

            # Prefer the disease/chemical NER model; fall back to base parser.
            for model in ("en_ner_bc5cdr_md", "en_core_sci_sm"):
                try:
                    self._nlp = spacy.load(model)
                    break
                except OSError:
                    continue
            if self._nlp is None:
                raise OSError("no scispaCy model installed")
        except Exception:  # noqa: BLE001
            self._is_fallback = True

    @property
    def is_fallback(self) -> bool:
        return self._is_fallback

    # ------------------------------------------------------------------ #
    def extract(self, text: str) -> MedicalEntities:
        if not self._is_fallback and self._nlp is not None:
            return self._extract_spacy(text)
        return self._extract_fallback(text)

    # ------------------------------------------------------------------ #
    def _extract_spacy(self, text: str) -> MedicalEntities:
        # Cap length to keep CPU inference snappy.
        doc = self._nlp(text[:100_000])
        ents = MedicalEntities()
        seen = set()
        for ent in doc.ents:
            term = ent.text.strip()
            key = term.lower()
            if key in seen or len(term) < 3:
                continue
            seen.add(key)
            label = ent.label_.upper()
            if label in ("DISEASE", "DISEASES"):
                ents.diseases.append(term)
            elif label in ("CHEMICAL", "DRUG"):
                ents.medications.append(term)
            else:
                # Base scientific model uses generic ENTITY; route by lexicon.
                self._route_fallback(term, ents)
        return ents

    # ------------------------------------------------------------------ #
    def _extract_fallback(self, text: str) -> MedicalEntities:
        lowered = text.lower()
        ents = MedicalEntities()
        for category, terms in _FALLBACK.items():
            found = sorted({t for t in terms if t in lowered})
            getattr(ents, category).extend(found)
        return ents

    @staticmethod
    def _route_fallback(term: str, ents: MedicalEntities) -> None:
        t = term.lower()
        for category, terms in _FALLBACK.items():
            if any(k in t for k in terms):
                getattr(ents, category).append(term)
                return
