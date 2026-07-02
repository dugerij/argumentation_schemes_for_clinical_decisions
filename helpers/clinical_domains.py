from __future__ import annotations

import re
from typing import Protocol

from retrieval.concepts.extractor import UMLSConceptExtractor
from retrieval.concepts.umls import UMLSClient


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
NEGATION_RE = r"(?:no|without|denies)\s+"

STOPWORDS = {
    "about", "after", "again", "against", "also", "among", "because", "before", "between",
    "could", "does", "during", "each", "from", "have", "into", "most", "next", "other",
    "patient", "patients", "should", "that", "their", "there", "these", "they", "this",
    "those", "through", "under", "what", "which", "while", "with", "without", "would",
    "years", "year", "woman", "women", "male", "female", "best", "following", "likely",
}

DOMAIN_TERMS: dict[str, tuple[str, ...]] = {
    "renal_metabolic": (
        "renal",
        "kidney",
        "nephro",
        "nephritic",
        "nephrotic",
        "glomerular",
        "glomerulonephritis",
        "proteinuria",
        "albuminuria",
        "hematuria",
        "pyelonephritis",
        "aki",
        "ckd",
        "esrd",
        "dialysis",
        "hemodialysis",
        "peritoneal",
        "creatinine",
        "bun",
        "uremia",
        "hyperkalemia",
        "hypokalemia",
        "hyponatremia",
        "hypernatremia",
        "acidosis",
        "alkalosis",
        "electrolyte",
        "metabolic",
        "diabetic",
        "diabetes",
        "insulin",
    ),
}

DOMAIN_PHRASES: dict[str, tuple[str, ...]] = {
    "renal_metabolic": (
        "acute kidney injury",
        "chronic kidney disease",
        "end stage renal disease",
        "end-stage renal disease",
        "renal replacement therapy",
        "metabolic acidosis",
        "metabolic alkalosis",
        "diabetic ketoacidosis",
        "hyperosmolar hyperglycemic state",
        "renal failure",
        "kidney failure",
    ),
}


def normalize_domain_name(domain: str) -> str:
    normalized = domain.strip().lower().replace("-", "_")
    if normalized not in DOMAIN_TERMS:
        raise ValueError(f"Unsupported clinical domain: {domain}")
    return normalized


def tokenize(text: str) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_RE.findall(text)
        if token and token.lower() not in STOPWORDS
    }


def domain_hit_terms(text: str, domain: str) -> set[str]:
    normalized_domain = normalize_domain_name(domain)
    lowered = text.lower()
    tokenized = tokenize(text)
    hits = set()
    for term in DOMAIN_TERMS[normalized_domain]:
        if term not in lowered and term not in tokenized:
            continue
        if re.search(rf"{NEGATION_RE}{re.escape(term)}\b", lowered):
            continue
        hits.add(term)
    for phrase in DOMAIN_PHRASES.get(normalized_domain, ()):
        if phrase in lowered and not re.search(rf"{NEGATION_RE}{re.escape(phrase)}\b", lowered):
            hits.add(phrase)
    return hits


def domain_matches(text: str, domain: str, *, min_hits: int = 1) -> bool:
    return len(domain_hit_terms(text, domain)) >= min_hits


def domain_vocabulary(domain: str) -> set[str]:
    normalized_domain = normalize_domain_name(domain)
    vocabulary = set(DOMAIN_TERMS[normalized_domain])
    for phrase in DOMAIN_PHRASES.get(normalized_domain, ()):
        vocabulary.update(tokenize(phrase))
    return vocabulary


def count_domain_hits(text: str, domain: str) -> int:
    return len(domain_hit_terms(text, domain))


class DomainMatcher(Protocol):
    def match_details(self, text: str) -> tuple[int, list[str]]: ...


class UMLSDomainMatcher:
    def __init__(self, client: UMLSClient, domain: str, *, candidate_limit: int = 120):
        self.domain = normalize_domain_name(domain)
        self.candidate_limit = candidate_limit
        self.extractor = UMLSConceptExtractor(client)
        self.seed_cuis = {
            concept.cui
            for term in DOMAIN_TERMS[self.domain] + DOMAIN_PHRASES.get(self.domain, ())
            if (concept := client.best_match(term)) is not None and concept.cui
        }

    def match_details(self, text: str) -> tuple[int, list[str]]:
        mentions = self.extractor.extract_from_text(text, limit=self.candidate_limit)
        matched: list[str] = []
        seen: set[str] = set()
        for mention in mentions:
            concept = mention.concept
            concept_text = " ".join(
                part
                for part in (
                    mention.text or "",
                    concept.preferred_term if concept else "",
                    concept.semantic_type if concept else "",
                )
                if part
            )
            if concept and concept.cui and concept.cui in self.seed_cuis:
                key = concept.cui
                if key not in seen:
                    seen.add(key)
                    matched.append(concept.preferred_term or mention.text)
                continue
            if domain_matches(concept_text, self.domain, min_hits=1):
                key = (concept.preferred_term if concept and concept.preferred_term else mention.text).lower()
                if key not in seen:
                    seen.add(key)
                    matched.append(concept.preferred_term if concept and concept.preferred_term else mention.text)
        return len(matched), matched
