import os
import logging
import time
from dataclasses import dataclass
from typing import Any

import requests

from retrieval.concepts.local_umls import DEFAULT_LOCAL_UMLS_DB_PATH, LocalUMLSClient
from retrieval.concepts.schema import UMLSConcept
from retrieval.concepts.vocabularies import SOURCE_PRIORITY, category_for

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class UMLSConfig:
    api_key: str | None = None
    version: str = "current"
    base_url: str = "https://uts-ws.nlm.nih.gov/rest"
    source_vocabularies: tuple[str, ...] = SOURCE_PRIORITY
    page_size: int = 10
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0
    backend: str = "local"
    local_db_path: str = str(DEFAULT_LOCAL_UMLS_DB_PATH)

    @classmethod
    def from_env(cls) -> "UMLSConfig":
        backend = os.environ.get("UMLS_BACKEND", "local").strip().lower()
        api_key = os.environ.get("UMLS_API_KEY")
        if backend == "api" and not api_key:
            raise ValueError("UMLS_API_KEY must be set when UMLS_BACKEND=api.")

        source_vocabularies = parse_source_vocabularies(
            os.environ.get("UMLS_SOURCE_VOCABS", ",".join(SOURCE_PRIORITY))
        )

        return cls(
            api_key=api_key,
            version=os.environ.get("UMLS_VERSION", "current"),
            base_url=os.environ.get("UMLS_BASE_URL", "https://uts-ws.nlm.nih.gov/rest"),
            source_vocabularies=source_vocabularies,
            page_size=int(os.environ.get("UMLS_PAGE_SIZE", "10")),
            max_retries=int(os.environ.get("UMLS_MAX_RETRIES", "3")),
            retry_backoff_seconds=float(os.environ.get("UMLS_RETRY_BACKOFF_SECONDS", "1.0")),
            backend=backend,
            local_db_path=os.environ.get("UMLS_LOCAL_DB_PATH", str(DEFAULT_LOCAL_UMLS_DB_PATH)),
        )


def parse_source_vocabularies(value: str) -> tuple[str, ...]:
    """Parse comma-separated UMLS vocabularies, allowing inline comments."""
    vocabularies: list[str] = []
    for item in value.split(","):
        token = item.split("#", 1)[0].strip()
        if token:
            vocabularies.append(token)
    return tuple(dict.fromkeys(vocabularies))


class UMLSClient:
    def __init__(self, config: UMLSConfig):
        self.config = config
        self._search_cache: dict[tuple[str, tuple[str, ...] | None, int | None, str], list[UMLSConcept]] = {}
        self._best_match_cache: dict[tuple[str, tuple[str, ...] | None], UMLSConcept | None] = {}
        self.failed_request_count = 0

    def search(
        self,
        term: str,
        source_vocabularies: tuple[str, ...] | None = None,
        page_size: int | None = None,
        search_type: str = "words",
    ) -> list[UMLSConcept]:
        cache_key = (term.strip().lower(), source_vocabularies, page_size, search_type)
        if cache_key in self._search_cache:
            return self._search_cache[cache_key]

        params: dict[str, Any] = {
            "apiKey": self.config.api_key,
            "string": term,
            "searchType": search_type,
            "pageSize": page_size or self.config.page_size,
        }

        sources = source_vocabularies or self.config.source_vocabularies
        if sources:
            params["sabs"] = ",".join(sources)

        response = self._get_with_retries(f"{self.config.base_url}/search/{self.config.version}", params=params)
        if response is None:
            self._search_cache[cache_key] = []
            return []
        payload = response.json()

        matches = [self._concept_from_search_result(item) for item in payload.get("result", {}).get("results", [])]
        self._search_cache[cache_key] = matches
        return matches

    def best_match(
        self,
        term: str,
        source_vocabularies: tuple[str, ...] | None = None,
    ) -> UMLSConcept | None:
        cache_key = (term.strip().lower(), source_vocabularies)
        if cache_key in self._best_match_cache:
            return self._best_match_cache[cache_key]

        matches = self.search(term, source_vocabularies=source_vocabularies)
        if not matches:
            self._best_match_cache[cache_key] = None
            return None

        priority = {source: index for index, source in enumerate(self.config.source_vocabularies)}
        best = sorted(
            matches,
            key=lambda concept: priority.get(concept.source_vocabulary, len(priority)),
        )[0]
        self._best_match_cache[cache_key] = best
        return best

    def _get_with_retries(self, url: str, params: dict[str, Any]) -> requests.Response | None:
        attempts = max(1, self.config.max_retries + 1)
        last_error: requests.RequestException | None = None

        for attempt in range(1, attempts + 1):
            response = None
            try:
                response = requests.get(url, params=params, timeout=30)
                if response.status_code not in RETRYABLE_STATUS_CODES:
                    response.raise_for_status()
                    return response

                last_error = requests.HTTPError(
                    f"{response.status_code} retryable UMLS response for {url}",
                    response=response,
                )
                if attempt == attempts:
                    break
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = exc
                if attempt == attempts:
                    break

            self._sleep_before_retry(attempt, response)

        self.failed_request_count += 1
        logger.warning("UMLS lookup failed after %s attempts: %s", attempts, last_error)
        return None

    def _sleep_before_retry(self, attempt: int, response: requests.Response | None = None) -> None:
        retry_after = response.headers.get("Retry-After") if response is not None else None
        if retry_after:
            try:
                delay = float(retry_after)
            except ValueError:
                delay = self.config.retry_backoff_seconds * attempt
        else:
            delay = self.config.retry_backoff_seconds * attempt
        if delay > 0:
            time.sleep(delay)

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


def create_umls_client(config: UMLSConfig) -> UMLSClient | LocalUMLSClient:
    if config.backend == "local":
        return LocalUMLSClient(
            db_path=config.local_db_path,
            source_vocabularies=config.source_vocabularies,
        )
    if config.backend != "api":
        raise ValueError(f"Unsupported UMLS_BACKEND: {config.backend}")
    return UMLSClient(config)
