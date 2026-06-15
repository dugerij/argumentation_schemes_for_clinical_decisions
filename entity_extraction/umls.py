import os
from dataclasses import dataclass
from typing import Any

import requests

from entity_extraction.schema import UMLSConcept
from entity_extraction.vocabularies import SOURCE_PRIORITY, category_for


@dataclass(frozen=True)
class UMLSConfig:
    api_key: str
    version: str = "current"
    base_url: str = "https://uts-ws.nlm.nih.gov/rest"
    source_vocabularies: tuple[str, ...] = SOURCE_PRIORITY
    page_size: int = 10

    @classmethod
    def from_env(cls) -> "UMLSConfig":
        api_key = os.environ.get("UMLS_API_KEY")
        if not api_key:
            raise ValueError("UMLS_API_KEY must be set to use UMLS concept lookup.")

        source_vocabularies = tuple(
            item.strip()
            for item in os.environ.get("UMLS_SOURCE_VOCABS", ",".join(SOURCE_PRIORITY)).split(",")
            if item.strip()
        )

        return cls(
            api_key=api_key,
            version=os.environ.get("UMLS_VERSION", "current"),
            base_url=os.environ.get("UMLS_BASE_URL", "https://uts-ws.nlm.nih.gov/rest"),
            source_vocabularies=source_vocabularies,
            page_size=int(os.environ.get("UMLS_PAGE_SIZE", "10")),
        )


class UMLSClient:
    def __init__(self, config: UMLSConfig):
        self.config = config

    def search(
        self,
        term: str,
        source_vocabularies: tuple[str, ...] | None = None,
        page_size: int | None = None,
        search_type: str = "words",
    ) -> list[UMLSConcept]:
        params: dict[str, Any] = {
            "apiKey": self.config.api_key,
            "string": term,
            "searchType": search_type,
            "pageSize": page_size or self.config.page_size,
        }

        sources = source_vocabularies or self.config.source_vocabularies
        if sources:
            params["sabs"] = ",".join(sources)

        response = requests.get(
            f"{self.config.base_url}/search/{self.config.version}",
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()

        return [self._concept_from_search_result(item) for item in payload.get("result", {}).get("results", [])]

    def best_match(
        self,
        term: str,
        source_vocabularies: tuple[str, ...] | None = None,
    ) -> UMLSConcept | None:
        matches = self.search(term, source_vocabularies=source_vocabularies)
        if not matches:
            return None

        priority = {source: index for index, source in enumerate(self.config.source_vocabularies)}
        return sorted(
            matches,
            key=lambda concept: priority.get(concept.source_vocabulary, len(priority)),
        )[0]

    def _concept_from_search_result(self, item: dict[str, Any]) -> UMLSConcept:
        source_vocabulary = item.get("rootSource") or item.get("source") or "UMLS"
        semantic_type = item.get("semanticType") or item.get("semanticTypes") or ""
        if isinstance(semantic_type, list):
            semantic_type = semantic_type[0] if semantic_type else ""

        return UMLSConcept(
            cui=item.get("ui", ""),
            preferred_term=item.get("name", ""),
            semantic_type=semantic_type,
            source_vocabulary=source_vocabulary,
            source_code=item.get("code"),
            category=category_for(source_vocabulary, semantic_type),
            metadata={
                "uri": item.get("uri"),
                "raw": item,
            },
        )
