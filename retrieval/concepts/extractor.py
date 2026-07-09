import re
from collections.abc import Iterable

from retrieval.concepts.candidates import extract_candidate_terms
from retrieval.concepts.schema import ClinicalEntityMention
from retrieval.concepts.umls import UMLSClient


class UMLSConceptExtractor:
    """Maps candidate clinical terms to UMLS concepts and text spans.

    This is intentionally term-based. MIMIC ingestion can later plug in a
    stronger mention detector while reusing the same UMLS normalization layer.
    """

    def __init__(self, client: UMLSClient):
        self.client = client

    def extract_from_terms(
        self,
        text: str,
        candidate_terms: Iterable[str],
        *,
        max_mentions: int | None = None,
    ) -> list[ClinicalEntityMention]:
        mentions: list[ClinicalEntityMention] = []
        seen: set[tuple[str, int, int]] = set()

        for term in sorted(set(candidate_terms), key=len, reverse=True):
            if not term.strip():
                continue

            concept = self.client.best_match(term)
            if concept is None:
                continue
            pattern = re.compile(rf"\b{re.escape(term)}\b", flags=re.IGNORECASE)

            for match in pattern.finditer(text):
                key = (term.lower(), match.start(), match.end())
                if key in seen:
                    continue
                seen.add(key)
                mentions.append(
                    ClinicalEntityMention(
                        text=match.group(0),
                        concept=concept,
                        start_char=match.start(),
                        end_char=match.end(),
                        category=concept.category if concept else None,
                    )
                )
                if max_mentions is not None and len(mentions) >= max_mentions:
                    return mentions

        return mentions

    def extract_from_text(self, text: str, limit: int = 120, *, max_mentions: int | None = None) -> list[ClinicalEntityMention]:
        candidate_terms = extract_candidate_terms(text, limit=limit)
        return self.extract_from_terms(text, candidate_terms, max_mentions=max_mentions)
