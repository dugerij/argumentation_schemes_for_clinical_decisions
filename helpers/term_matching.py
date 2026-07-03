from __future__ import annotations

"""Generic term matching utilities for specialty subset selection."""

import re
from typing import Protocol

from retrieval.concepts.umls import UMLSClient


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
NEGATION_RE = r"(?:no|without|denies)\s+"
DIAGNOSIS_BOOST = 2

STOPWORDS = {
    "about", "after", "again", "against", "also", "among", "because", "before", "between",
    "could", "does", "during", "each", "from", "have", "into", "most", "next", "other",
    "patient", "patients", "should", "that", "their", "there", "these", "they", "this",
    "those", "through", "under", "what", "which", "while", "with", "without", "would",
    "years", "year", "woman", "women", "male", "female", "best", "following", "likely",
}


def tokenize(text: str) -> set[str]:
    """Tokenize free text into lowercase matching terms."""

    return {
        token.lower()
        for token in TOKEN_RE.findall(text)
        if token and token.lower() not in STOPWORDS
    }


class TermMatcher(Protocol):
    """Protocol for text matchers used by note and question subset filters."""

    def match_details(self, text: str) -> tuple[int, list[str]]: ...


class KeywordSeedMatcher:
    """Local term matcher built only from the provided seed terms."""

    def __init__(self, seed_terms: tuple[str, ...]):
        self.seed_phrases = tuple(sorted({term.strip().lower() for term in seed_terms if term.strip()}))
        self.seed_tokens = {token for phrase in self.seed_phrases for token in tokenize(phrase)}

    def match_details(self, text: str) -> tuple[int, list[str]]:
        lowered = text.lower()
        tokenized = tokenize(text)
        diagnosis_hits: list[str] = []
        token_hits: list[str] = []
        seen: set[str] = set()

        for phrase in self.seed_phrases:
            if phrase in lowered and not re.search(rf"{NEGATION_RE}{re.escape(phrase)}\b", lowered):
                if phrase not in seen:
                    seen.add(phrase)
                    diagnosis_hits.append(phrase)

        for token in sorted(self.seed_tokens):
            if token not in lowered and token not in tokenized:
                continue
            if re.search(rf"{NEGATION_RE}{re.escape(token)}\b", lowered):
                continue
            if token not in seen:
                seen.add(token)
                token_hits.append(token)

        count = len(token_hits) + DIAGNOSIS_BOOST * len(diagnosis_hits)
        return count, diagnosis_hits + token_hits


def _expand_seed_terms_with_umls(client: UMLSClient, seed_terms: tuple[str, ...]) -> tuple[set[str], set[str]]:
    """Expand seed terms once through UMLS and return phrase and token vocabularies."""

    phrases = {term.strip().lower() for term in seed_terms if term.strip()}
    tokens = {token for phrase in phrases for token in tokenize(phrase)}

    for seed in seed_terms:
        for concept in client.search(seed, page_size=5):
            preferred = (concept.preferred_term or "").strip().lower()
            if not preferred:
                continue
            phrases.add(preferred)
            tokens.update(tokenize(preferred))

    return phrases, tokens


class UMLSSeedVocabularyMatcher:
    """Matcher built from a one-time UMLS expansion of diagnosis seed terms.

    A note is strongly favored when an expanded diagnosis phrase appears. Plain
    specialty-term matches still count, but they need several hits to reach the
    usual inclusion threshold.
    """

    def __init__(self, client: UMLSClient, seed_terms: tuple[str, ...]):
        self.seed_terms = seed_terms
        self.phrases, self.tokens = _expand_seed_terms_with_umls(client, seed_terms)

    def match_details(self, text: str) -> tuple[int, list[str]]:
        lowered = text.lower()
        tokenized = tokenize(text)
        diagnosis_hits: list[str] = []
        token_hits: list[str] = []
        seen: set[str] = set()

        for phrase in sorted(self.phrases):
            if phrase in lowered and not re.search(rf"{NEGATION_RE}{re.escape(phrase)}\b", lowered):
                if phrase not in seen:
                    seen.add(phrase)
                    diagnosis_hits.append(phrase)

        for token in sorted(self.tokens):
            if token not in lowered and token not in tokenized:
                continue
            if re.search(rf"{NEGATION_RE}{re.escape(token)}\b", lowered):
                continue
            if token not in seen:
                seen.add(token)
                token_hits.append(token)

        count = len(token_hits) + DIAGNOSIS_BOOST * len(diagnosis_hits)
        return count, diagnosis_hits + token_hits
