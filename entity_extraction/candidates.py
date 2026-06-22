from __future__ import annotations

import re


STOPWORDS = {
    "the",
    "and",
    "with",
    "from",
    "that",
    "this",
    "these",
    "those",
    "into",
    "onto",
    "over",
    "under",
    "between",
    "within",
    "without",
    "for",
    "of",
    "to",
    "in",
    "on",
    "at",
    "by",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "a",
    "an",
    "or",
    "it",
    "its",
    "if",
    "then",
    "when",
    "while",
    "may",
    "might",
    "can",
    "could",
    "should",
    "would",
    "patient",
    "patients",
    "clinical",
    "medicine",
    "medical",
    "treatment",
    "care",
    "history",
    "case",
    "report",
}

MEDICAL_SUFFIXES = (
    "itis",
    "emia",
    "osis",
    "opathy",
    "algia",
    "uria",
    "ectomy",
    "plasty",
    "graphy",
    "scopy",
    "meter",
    "gram",
    "ology",
    "ase",
    "mab",
    "pril",
    "sartan",
    "statin",
    "azole",
    "idine",
    "olol",
    "ine",
    "amide",
)

CLINICAL_CUES = {
    "disease",
    "disorder",
    "syndrome",
    "symptom",
    "sign",
    "diagnosis",
    "medication",
    "drug",
    "therapy",
    "treatment",
    "procedure",
    "lab",
    "laboratory",
    "test",
    "result",
    "finding",
    "risk",
    "contraindication",
    "adverse",
    "recommendation",
    "goal",
    "renal",
    "cardiac",
    "pulmonary",
    "infection",
    "hypertension",
    "diabetes",
    "failure",
    "dysfunction",
}

BAD_TOKENS = {
    "patient",
    "patients",
    "has",
    "have",
    "had",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "with",
    "without",
    "treated",
    "treat",
}

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9/-]*")


def _score_term(term: str) -> int:
    words = [word.lower().strip(".,;:()[]{}") for word in term.split()]
    score = len(words)
    if any(any(word.endswith(suffix) for suffix in MEDICAL_SUFFIXES) for word in words):
        score += 3
    if any(word in CLINICAL_CUES for word in words):
        score += 3
    if any(char.isdigit() for char in term):
        score += 2
    if "-" in term or "/" in term:
        score += 1
    if term[:1].isupper():
        score += 1
    if any(word not in STOPWORDS for word in words):
        score += 1
    return score


def extract_candidate_terms(text: str, limit: int = 120) -> list[str]:
    candidates: dict[str, int] = {}

    tokens = TOKEN_RE.findall(text)
    max_span = 4
    for start in range(len(tokens)):
        for span in range(1, max_span + 1):
            window = tokens[start : start + span]
            if len(window) != span:
                continue
            lowered = [word.lower().strip(".,;:()[]{}") for word in window]
            if any(word in STOPWORDS for word in lowered):
                continue
            if lowered[0] in BAD_TOKENS or lowered[-1] in BAD_TOKENS:
                continue
            normalized = " ".join(window)
            if len(normalized) < 4:
                continue
            if len(lowered) == 1 and len(lowered[0]) < 5 and not any(
                lowered[0].endswith(suffix) for suffix in MEDICAL_SUFFIXES
            ):
                continue

            score = _score_term(normalized)
            if score < 3:
                continue
            existing = candidates.get(normalized)
            if existing is None or score > existing:
                candidates[normalized] = score

    ordered = sorted(candidates.items(), key=lambda item: (-item[1], -len(item[0]), item[0].lower()))
    return [term for term, _ in ordered[:limit]]
