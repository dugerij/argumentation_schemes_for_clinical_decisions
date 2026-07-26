from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from clinical_cds.direct import label_key
from clinical_cds.terminology.candidates import extract_candidate_terms
from clinical_cds.terminology.local_umls import LocalUMLSClient
from clinical_cds.terminology.schema import UMLSConcept


LEXICAL_DIAGNOSIS_ALIASES = {
    "stemi": "stemiacs",
    "copdasthma": "asthmacopd",
    "severeasthmaexacerbation": "severeasthma",
    "pituitarymacroadenoma": "pituitarymacroadenomas",
    "pituitarymicroadenoma": "pituitarymicroadenomas",
    "gastroesophagealrefluxdisease": "gastrooesophagealrefluxdisease",
}
DIAGNOSIS_CATEGORIES = {"diagnosis"}
RETRIEVAL_CATEGORIES = {
    "diagnosis",
    "clinical_finding",
    "lab_or_measurement",
    "therapy_or_procedure",
}


def lexical_diagnosis_key(value: str) -> str:
    key = label_key(value)
    return LEXICAL_DIAGNOSIS_ALIASES.get(key, key)


@dataclass
class UMLSNormalizer:
    client: LocalUMLSClient
    candidate_limit: int = 12
    aliases_per_concept: int = 3

    @classmethod
    def from_path(cls, db_path: Path) -> "UMLSNormalizer":
        return cls(client=LocalUMLSClient(db_path=Path(db_path)))

    @property
    def normalizer_id(self) -> str:
        sources = ",".join(self.client.source_vocabularies)
        alias_mode = (
            "full-aliases"
            if self.client.supports_full_alias_lookup
            else "preferred-terms"
        )
        return (
            f"umls-local-v2-{alias_mode}:"
            f"{self.client.database_id}:{sources}"
        )

    def concept(
        self,
        term: str,
        *,
        categories: set[str] | None = None,
    ) -> UMLSConcept | None:
        match = self.client.best_match(term)
        if match is None:
            return None
        if categories is not None and match.category not in categories:
            return None
        return match

    def diagnosis_key(self, term: str) -> str:
        match = self.concept(term, categories=DIAGNOSIS_CATEGORIES)
        if match is not None:
            return f"cui:{match.cui.casefold()}"
        return lexical_diagnosis_key(term)

    def diagnosis_keys(self, term: str) -> tuple[str, ...]:
        lexical_key = lexical_diagnosis_key(term)
        concept_key = self.diagnosis_key(term)
        return tuple(dict.fromkeys((lexical_key, concept_key)))

    def match_diagnosis(
        self,
        term: str,
        labels: Iterable[str],
    ) -> str | None:
        label_list = tuple(labels)
        lexical_key = lexical_diagnosis_key(term)
        for label in label_list:
            if lexical_diagnosis_key(label) == lexical_key:
                return label

        concept_key = self.diagnosis_key(term)
        if not concept_key.startswith("cui:"):
            return None
        for label in label_list:
            if self.diagnosis_key(label) == concept_key:
                return label
        return None

    def expand_text(self, text: str) -> tuple[str, ...]:
        expansions: list[str] = []
        seen_terms: set[str] = set()
        seen_cuis: set[str] = set()
        for candidate in extract_candidate_terms(text, limit=self.candidate_limit):
            match = self.concept(candidate, categories=RETRIEVAL_CATEGORIES)
            if match is None or match.cui in seen_cuis:
                continue
            seen_cuis.add(match.cui)
            terms = (
                match.preferred_term,
                *self.client.concept_terms(
                    match.cui,
                    limit=self.aliases_per_concept,
                ),
            )
            for term in terms:
                normalized = " ".join(term.split()).strip()
                key = normalized.casefold()
                if not normalized or key in seen_terms:
                    continue
                seen_terms.add(key)
                expansions.append(normalized)
        return tuple(expansions)
