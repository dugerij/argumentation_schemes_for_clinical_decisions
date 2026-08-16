from __future__ import annotations

import math
import json
import re
import urllib.request
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from clinical_cds.retrieval import tokenize
from clinical_cds.direct import label_key
from clinical_cds.schema import (
    ClinicalCase,
    FamilyChildFact,
    FamilyDiagnosisAlternative,
    RetrievalBundle,
    RetrievedFact,
    RetrievedFamilyRoute,
)
from clinical_cds.normalization import UMLSNormalizer
from clinical_cds.terminology.candidates import extract_candidate_terms

from .corpus import ControlledPremise
from .retrieval_text import atomic_retrieval_segments
from .provenance_contract import (
    canonical_candidate_id,
    canonical_sha256,
    validate_bundle_citation_allowlist,
)


RETRIEVER_ID = "microsoft-graphrag-3.1.0-structured-family-coverage-version-i"
FLAT_RETRIEVER_ID = "controlled-premise-bm25-flat-v1"
MAX_GRAPH_ROUTES = 8
MAX_GRAPH_FACTS = 12
MAX_RETRIEVED_CANDIDATES = 32
HYBRID_RRF_CONSTANT = 20.0
DEFAULT_DENSE_NEIGHBORS = 64
NUMERIC_THRESHOLD_RE = re.compile(
    r"(?P<metric>[A-Za-z][A-Za-z0-9 /_()'-]{0,48}?)\s*"
    r"(?P<operator>>=|<=|≥|≤|>|<)\s*(?P<value>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
HISTORICAL_CONTEXT_RE = re.compile(
    r"\b(?:history of|historical|previous(?:ly)?|prior|remote|resolved)\b",
    re.IGNORECASE,
)
POSITIVE_RESULT_RE = re.compile(
    r"\b(?:positive|detect(?:s|ed|ing)?|demonstrat(?:e|es|ed|ing)|show(?:s|ed|ing)?|"
    r"confirm(?:s|ed|ing)?|reveals?|present)\b",
    re.IGNORECASE,
)
NEGATIVE_RESULT_RE = re.compile(
    r"\b(?:negative|normal|absent|excluded|ruled out|without|no evidence of)\b",
    re.IGNORECASE,
)
STRUCTURED_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "criteria",
    "criterion", "diagnostic", "evidence", "for", "in", "is", "of",
    "or", "patient", "result", "results", "supports", "test", "the",
    "to", "with",
}
DIAGNOSTIC_MODALITY_PATTERNS = (
    ("ventilation-perfusion-scan", re.compile(r"\b(?:v\s*/\s*q|ventilation[- ]perfusion)(?:\s+lung)?\s+scan\b", re.I)),
    ("ct-angiography", re.compile(r"\b(?:ctpa|cta|ct\s+angiograph\w*|computed\s+tomography\s+angiograph\w*)\b", re.I)),
    ("computed-tomography", re.compile(r"\b(?:ct(?:\s+scan)?|computed\s+tomography)\b", re.I)),
    ("magnetic-resonance", re.compile(r"\b(?:mri|magnetic\s+resonance\s+imag\w*)\b", re.I)),
    ("echocardiography", re.compile(r"\b(?:echocardiogra\w*|echo)\b", re.I)),
    ("ultrasound", re.compile(r"\b(?:ultrasound|ultrasonogra\w*)\b", re.I)),
    ("angiography", re.compile(r"\bangiogra\w*\b", re.I)),
    ("endoscopy", re.compile(r"\bendoscop\w*\b", re.I)),
    ("gastroscopy", re.compile(r"\bgastroscop\w*\b", re.I)),
    ("colonoscopy", re.compile(r"\bcolonoscop\w*\b", re.I)),
    ("biopsy", re.compile(r"\bbiops\w*\b", re.I)),
    ("microscopy", re.compile(r"\bmicroscop\w*\b", re.I)),
    ("pcr", re.compile(r"\b(?:pcr|polymerase\s+chain\s+reaction)\b", re.I)),
    ("culture", re.compile(r"\bcultur\w*\b", re.I)),
    ("radiograph", re.compile(r"\b(?:x[- ]?ray|radiograph\w*)\b", re.I)),
    ("electrocardiography", re.compile(r"\b(?:ecg|ekg|electrocardiogra\w*)\b", re.I)),
    ("electroencephalography", re.compile(r"\b(?:eeg|electroencephalogra\w*)\b", re.I)),
)

PREMISE_TYPE_WEIGHTS = {
    "diagnostic criteria": 1.25,
    "criteria": 1.25,
    "confirmatory test": 1.25,
    "laboratory finding": 1.2,
    "signs": 1.1,
    "sign": 1.1,
    "symptoms": 1.0,
    "symptom": 1.0,
    "risk factors": 0.8,
    "risk factor": 0.8,
}


@dataclass(frozen=True)
class DensePremiseNeighbor:
    source_chunk_id: str
    score: float
    rank: int
    adjusted_score: float = 0.0


class DensePremiseIndex:
    """Normalized premise vectors for finding-level nearest-neighbour search."""

    def __init__(
        self,
        corpus: Iterable[ControlledPremise],
        vectors: Sequence[Sequence[float]],
        *,
        source_indices: Sequence[int] | None = None,
    ) -> None:
        self.corpus = tuple(corpus)
        matrix = np.asarray(vectors, dtype=np.float64)
        indices = tuple(
            range(len(self.corpus)) if source_indices is None else source_indices
        )
        if (
            not self.corpus
            or matrix.ndim != 2
            or len(matrix) != len(indices)
            or any(index < 0 or index >= len(self.corpus) for index in indices)
        ):
            raise ValueError("Dense premise vectors do not match the corpus.")
        if not np.isfinite(matrix).all():
            raise ValueError("Dense premise vectors contain non-finite values.")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms <= 0.0):
            raise ValueError("Dense premise vectors must be non-zero.")
        self.matrix = matrix / norms
        self._source_indices = np.asarray(indices, dtype=np.int64)
        centroid = np.mean(self.matrix, axis=0)
        centroid_norm = float(np.linalg.norm(centroid))
        self._hubness = (
            self.matrix @ (centroid / centroid_norm)
            if centroid_norm > 0.0
            else np.zeros(len(self.corpus), dtype=np.float64)
        )

    @classmethod
    def from_openai_endpoint(
        cls,
        corpus: Iterable[ControlledPremise],
        *,
        api_base: str,
        model: str,
        batch_size: int = 64,
    ) -> "DensePremiseIndex":
        records = tuple(corpus)
        texts: list[str] = []
        source_indices: list[int] = []
        for index, record in enumerate(records):
            segments = atomic_retrieval_segments(record.text) or (record.text,)
            for segment in segments:
                texts.append(" ".join((
                    record.diagnosis_label,
                    record.premise_type,
                    segment,
                )))
                source_indices.append(index)
        vectors = _openai_embeddings(
            tuple(texts),
            api_base=api_base,
            model=model,
            batch_size=batch_size,
        )
        return cls(records, vectors, source_indices=source_indices)

    def nearest(
        self,
        query_vectors: Sequence[Sequence[float]],
        *,
        maximum_neighbors: int = DEFAULT_DENSE_NEIGHBORS,
    ) -> tuple[DensePremiseNeighbor, ...]:
        if maximum_neighbors < 1:
            raise ValueError("maximum_neighbors must be positive.")
        queries = np.asarray(query_vectors, dtype=np.float64)
        if queries.ndim != 2 or queries.shape[1] != self.matrix.shape[1]:
            raise ValueError("Dense query vectors have the wrong dimensions.")
        if not np.isfinite(queries).all():
            raise ValueError("Dense query vectors contain non-finite values.")
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        if np.any(norms <= 0.0):
            raise ValueError("Dense query vectors must be non-zero.")
        queries = queries / norms
        # A premise may match any patient finding; max pooling avoids diluting a
        # discriminating ECG or laboratory finding inside the complete record.
        similarities = queries @ self.matrix.T
        passage_scores = np.max(similarities, axis=0)
        finding_margin = passage_scores - np.mean(similarities, axis=0)
        # Dense hubs sit close to the corpus centroid and otherwise rank for
        # many unrelated cases. Reward a premise that is unusually close to a
        # particular finding, while retaining the raw cosine for auditing.
        passage_adjusted = (
            passage_scores - 0.15 * self._hubness + 0.10 * finding_margin
        )
        scores = np.full(len(self.corpus), -np.inf, dtype=np.float64)
        adjusted = np.full(len(self.corpus), -np.inf, dtype=np.float64)
        for passage_index, source_index in enumerate(self._source_indices):
            scores[source_index] = max(
                scores[source_index], passage_scores[passage_index]
            )
            adjusted[source_index] = max(
                adjusted[source_index], passage_adjusted[passage_index]
            )
        ranked = sorted(
            range(len(self.corpus)),
            key=lambda index: (
                -float(adjusted[index]),
                -float(scores[index]),
                self.corpus[index].source_chunk_id,
            ),
        )[:maximum_neighbors]
        return tuple(
            DensePremiseNeighbor(
                source_chunk_id=self.corpus[index].source_chunk_id,
                score=round(float(scores[index]), 8),
                rank=rank,
                adjusted_score=round(float(adjusted[index]), 8),
            )
            for rank, index in enumerate(ranked, 1)
            if float(scores[index]) > 0.0
        )

    def embed_queries(
        self,
        texts: Iterable[str],
        *,
        api_base: str,
        model: str,
        batch_size: int = 64,
    ) -> tuple[tuple[float, ...], ...]:
        values = tuple(" ".join(text.split()) for text in texts if text.strip())
        if not values:
            raise ValueError("Dense retrieval requires patient finding text.")
        return _openai_embeddings(
            values,
            api_base=api_base,
            model=model,
            batch_size=batch_size,
        )


def _openai_embeddings(
    texts: Sequence[str],
    *,
    api_base: str,
    model: str,
    batch_size: int,
) -> tuple[tuple[float, ...], ...]:
    if not texts or batch_size < 1:
        raise ValueError("Embedding request requires text and a positive batch size.")
    endpoint = api_base.rstrip("/") + "/embeddings"
    output: list[tuple[float, ...]] = []
    for start in range(0, len(texts), batch_size):
        payload = json.dumps({
            "model": model,
            "input": list(texts[start:start + batch_size]),
            "encoding_format": "float",
        }).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            body = json.loads(response.read().decode("utf-8"))
        rows = sorted(body.get("data") or (), key=lambda row: int(row["index"]))
        if len(rows) != len(texts[start:start + batch_size]):
            raise RuntimeError("Embedding endpoint returned the wrong cardinality.")
        output.extend(tuple(float(value) for value in row["embedding"]) for row in rows)
    if len({len(vector) for vector in output}) != 1:
        raise RuntimeError("Embedding endpoint returned inconsistent dimensions.")
    return tuple(output)


def _generic_route_label(value: str) -> bool:
    normalized = " ".join(value.casefold().replace("-", " ").split())
    return normalized.startswith(("suspected ", "strongly suspected "))


def _meaningless_premise(value: str) -> bool:
    normalized = " ".join(value.casefold().strip(" .;,:").split())
    return normalized in {"etc", "and so on", "other", "others"}


def _premise_evidence_weight(record: ControlledPremise) -> float:
    premise_type = " ".join(record.premise_type.casefold().split())
    role_weight = PREMISE_TYPE_WEIGHTS.get(premise_type, 1.0)
    # Very long multi-topic passages behave as dense hubs. They remain
    # retrievable, but concise diagnostic statements receive more weight.
    content_length = max(len(tokenize(record.text)), 1)
    length_weight = 1.0 / math.sqrt(max(content_length / 64.0, 1.0))
    return role_weight * length_weight


def _multi_evidence_score(values: Iterable[float]) -> float:
    strongest = sorted((value for value in values if value > 0.0), reverse=True)[:3]
    return sum(weight * value for weight, value in zip((1.0, 0.5, 0.25), strongest))


def _comparison_holds(observed: float, operator: str, threshold: float) -> bool:
    return {
        ">": observed > threshold,
        ">=": observed >= threshold,
        "≥": observed >= threshold,
        "<": observed < threshold,
        "<=": observed <= threshold,
        "≤": observed <= threshold,
    }[operator]


def _numeric_threshold_match(query_text: str, premise_text: str) -> bool:
    """Evaluate named measurements against explicit premise thresholds."""
    query = " ".join(query_text.casefold().split())
    for threshold in NUMERIC_THRESHOLD_RE.finditer(premise_text):
        metric_tokens = tokenize(threshold.group("metric"))
        if not metric_tokens:
            continue
        metric = metric_tokens[-1]
        threshold_value = float(threshold.group("value"))
        observed_values: list[float] = []
        metric_pattern = re.escape(metric.casefold())
        for observed in re.finditer(
            rf"\b{metric_pattern}\b\s*(?:=|:|is|of)?\s*(\d+(?:\.\d+)?)",
            query,
        ):
            observed_values.append(float(observed.group(1)))
        if metric in {"sbp", "dbp"}:
            for blood_pressure in re.finditer(
                r"\b(?:bp|blood pressure)\b\s*(?:=|:)?\s*"
                r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)",
                query,
            ):
                position = 1 if metric == "sbp" else 2
                observed_values.append(float(blood_pressure.group(position)))
        if any(
            _comparison_holds(observed, threshold.group("operator"), threshold_value)
            for observed in observed_values
        ):
            return True
    return False


def _current_query_clauses(query_text: str) -> tuple[str, ...]:
    """Exclude explicitly historical clauses from executable retrieval boosts."""
    clauses = tuple(
        clause.strip()
        for clause in re.split(r"[\n.;]+", query_text)
        if clause.strip()
    )
    return tuple(
        clause for clause in clauses if not HISTORICAL_CONTEXT_RE.search(clause)
    )


def _event_subject_cuis(
    value: str,
    normalizer: UMLSNormalizer | object | None,
) -> frozenset[str]:
    """Resolve event subjects to exact UMLS concepts; never infer relatedness."""
    if normalizer is None:
        return frozenset()
    words = tuple(
        term for term in tokenize(value) if term not in STRUCTURED_STOPWORDS
    )
    terms = list(extract_candidate_terms(value, limit=16))
    terms.extend(
        " ".join(words[index:index + size])
        for size in (4, 3, 2, 1)
        for index in range(len(words) - size + 1)
    )
    cuis: set[str] = set()
    for term in dict.fromkeys(terms):
        concept = normalizer.concept(term)
        if concept is not None and getattr(concept, "cui", ""):
            cuis.add(str(concept.cui).casefold())
    return frozenset(cuis)


def _positive_diagnostic_event_match(
    query_text: str,
    premise_text: str,
    *,
    normalizer: UMLSNormalizer | object | None = None,
) -> bool:
    """Match modality, positive polarity, and event subject independently."""

    def events(text: str, *, current_only: bool) -> tuple[tuple[str, frozenset[str]], ...]:
        clauses = (
            _current_query_clauses(text)
            if current_only
            else tuple(
                clause.strip()
                for clause in re.split(r"[\n.;]+", text)
                if clause.strip()
            )
        )
        output: list[tuple[str, frozenset[str]]] = []
        for clause in clauses:
            normalized = " ".join(clause.casefold().split())
            if (
                not POSITIVE_RESULT_RE.search(normalized)
                or NEGATIVE_RESULT_RE.search(normalized)
            ):
                continue
            for modality, pattern in DIAGNOSTIC_MODALITY_PATTERNS:
                if not pattern.search(normalized):
                    continue
                without_modality = pattern.sub(" ", normalized)
                subject = _event_subject_cuis(without_modality, normalizer)
                if subject:
                    output.append((modality, subject))
                break
        return tuple(output)

    patient_events = events(query_text, current_only=True)
    warrant_events = events(premise_text, current_only=False)
    for patient_modality, patient_subject in patient_events:
        for warrant_modality, warrant_subject in warrant_events:
            if (
                patient_modality == warrant_modality
                and patient_subject & warrant_subject
            ):
                return True
    return False


def _structured_evidence_strength(
    query_text: str,
    premise_text: str,
    *,
    normalizer: UMLSNormalizer | object | None = None,
) -> float:
    current_text = ". ".join(_current_query_clauses(query_text))
    if current_text and _numeric_threshold_match(current_text, premise_text):
        return 1.0
    if _positive_diagnostic_event_match(
        query_text, premise_text, normalizer=normalizer
    ):
        return 0.8
    return 0.0


def rank_specific_candidate_choices(
    candidate_choices: Iterable[tuple[str, str, Iterable[str]]],
    corpus: Iterable[ControlledPremise],
    *,
    query_text: str = "",
    maximum_candidates: int = MAX_RETRIEVED_CANDIDATES,
) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """Rank routes using retrieval prior plus length-normalized clinical overlap."""
    corpus_list = tuple(corpus)
    corpus_by_source = {record.source_chunk_id: record for record in corpus_list}
    corpus_tokens = {
        record.source_chunk_id: tuple(tokenize(record.text))
        for record in corpus_list
    }
    document_frequency: Counter[str] = Counter()
    for values in corpus_tokens.values():
        document_frequency.update(set(values))
    query = Counter(tokenize(query_text))
    document_count = max(len(corpus_list), 1)

    def source_compatibility(source: str) -> float:
        terms = Counter(corpus_tokens[source])
        if not terms or not query:
            return 0.0
        overlap = 0.0
        for term, query_frequency in query.items():
            if term not in terms:
                continue
            frequency = document_frequency.get(term, document_count)
            inverse = math.log(
                1.0 + (document_count - frequency + 0.5) / (frequency + 0.5)
            )
            overlap += inverse * min(query_frequency, 2) * min(terms[term], 2)
        # Long prose accumulates incidental matches. Square-root normalization
        # preserves concise criteria while avoiding a hard short-text bias.
        return overlap / math.sqrt(max(len(terms), 1))
    ranked: list[tuple[float, int, str, str, tuple[str, ...]]] = []
    for position, (candidate_id, raw_label, raw_sources) in enumerate(
        candidate_choices, start=1
    ):
        label = " ".join(str(raw_label).split())
        if _generic_route_label(label):
            continue
        sources = tuple(
            source
            for source in dict.fromkeys(str(value) for value in raw_sources)
            if source in corpus_by_source
            and not _meaningless_premise(corpus_by_source[source].text)
        )
        if not sources:
            continue
        compatibility = [
            source_compatibility(source) * PREMISE_TYPE_WEIGHTS.get(
                " ".join(corpus_by_source[source].premise_type.casefold().split()),
                1.0,
            )
            for source in sources
        ]
        label_terms = set(tokenize(label))
        label_overlap = len(label_terms & set(query)) / max(len(label_terms), 1)
        # The embedding rank remains a prior. Lexical IDF compatibility then
        # distinguishes concise case-specific evidence from verbose boilerplate.
        score = (
            0.35 / (5.0 + position)
            + max(compatibility, default=0.0)
            + 0.35 * label_overlap
            + 0.015 * min(len(sources), 4)
        )
        ranked.append((score, position, str(candidate_id), label, sources))
    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    return tuple(
        (candidate_id, label, sources)
        for _, _, candidate_id, label, sources in ranked[:maximum_candidates]
    )


def flat_seeded_graph_candidate_choices(
    query_text: str,
    corpus: Iterable[ControlledPremise],
    *,
    dense_neighbors: Iterable[DensePremiseNeighbor] = (),
    normalizer: UMLSNormalizer | None = None,
    maximum_seed_facts: int = 24,
    maximum_seed_graphs: int = 12,
    maximum_candidates: int = MAX_RETRIEVED_CANDIDATES,
) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """Seed graphs with independently fused BM25 and dense premise rankings."""
    records = tuple(corpus)
    if not query_text.strip() or not records:
        raise ValueError("Hybrid retrieval requires a query and controlled corpus.")
    tokens = tuple(Counter(tokenize(record.text)) for record in records)
    lengths = tuple(max(sum(value.values()), 1) for value in tokens)
    average_length = sum(lengths) / len(lengths)
    document_frequency: Counter[str] = Counter()
    for value in tokens:
        document_frequency.update(value.keys())
    query = Counter(tokenize(query_text))
    document_count = len(records)
    scored: list[tuple[float, ControlledPremise]] = []
    for record, terms, length in zip(records, tokens, lengths, strict=True):
        bm25_length_normalizer = 1.2 * (0.25 + 0.75 * length / average_length)
        score = 0.0
        for term, query_frequency in query.items():
            term_frequency = terms.get(term, 0)
            if not term_frequency:
                continue
            frequency = document_frequency.get(term, 0)
            inverse = math.log(
                1.0 + (document_count - frequency + 0.5) / (frequency + 0.5)
            )
            score += (
                inverse * term_frequency * 2.2
                / (term_frequency + bm25_length_normalizer)
                * min(query_frequency, 2)
            )
        if score > 0 and not _meaningless_premise(record.text):
            scored.append((score, record))
    scored.sort(key=lambda item: (-item[0], item[1].source_chunk_id))
    dense = tuple(dense_neighbors)
    if dense:
        known_sources = {record.source_chunk_id for record in records}
        if (
            len({item.source_chunk_id for item in dense}) != len(dense)
            or any(item.source_chunk_id not in known_sources for item in dense)
            or any(
                item.rank < 1
                or not math.isfinite(item.score)
                or not math.isfinite(item.adjusted_score)
                for item in dense
            )
        ):
            raise ValueError("Dense neighbours are not a valid corpus ranking.")
        sparse_rrf = {
            record.source_chunk_id: 1.0 / (HYBRID_RRF_CONSTANT + rank)
            for rank, (_, record) in enumerate(scored, 1)
        }
        dense_rrf = {
            item.source_chunk_id: 1.0 / (HYBRID_RRF_CONSTANT + item.rank)
            for item in dense
        }
        records_by_source = {record.source_chunk_id: record for record in records}
        hybrid_scores = {
            source: (
                sparse_rrf.get(source, 0.0) + dense_rrf.get(source, 0.0)
            ) * _premise_evidence_weight(records_by_source[source])
            + (
                0.08
                if _numeric_threshold_match(query_text, records_by_source[source].text)
                else 0.0
            )
            for source in sparse_rrf.keys() | dense_rrf.keys()
        }
        fused = sorted(
            (
                (score, records_by_source[source])
                for source, score in hybrid_scores.items()
                if score > 0.0 and not _meaningless_premise(records_by_source[source].text)
            ),
            key=lambda item: (-item[0], item[1].source_chunk_id),
        )
        seeds = fused[:maximum_seed_facts]
        score_by_source = {record.source_chunk_id: score for score, record in fused}
    else:
        # Preserve the lexical-only comparator and existing deterministic tests.
        seeds = scored[:maximum_seed_facts]
        score_by_source = {record.source_chunk_id: score for score, record in scored}
    # Executable, current structured evidence is a separate retrieval channel.
    # It cannot be averaged away by verbose lexical or dense neighbours.
    structured = {
        record.source_chunk_id: _structured_evidence_strength(
            query_text, record.text, normalizer=normalizer
        )
        for record in records
    }
    structured = {source: score for source, score in structured.items() if score > 0.0}
    if structured:
        decisive_floor = max(score_by_source.values(), default=0.0) + 1.0
        records_by_source = {record.source_chunk_id: record for record in records}
        for source, strength in structured.items():
            score_by_source[source] = max(
                score_by_source.get(source, 0.0), decisive_floor * strength
            )
        seed_sources = {
            record.source_chunk_id for _, record in seeds
        } | structured.keys()
        seeds = sorted(
            (
                (score_by_source[source], records_by_source[source])
                for source in seed_sources
            ),
            key=lambda item: (-item[0], item[1].source_chunk_id),
        )[:maximum_seed_facts]
    graph_evidence: dict[str, list[float]] = {}
    for score, record in seeds:
        graph_evidence.setdefault(record.graph_id, []).append(score)
    graph_scores = {
        graph_id: _multi_evidence_score(scores)
        for graph_id, scores in graph_evidence.items()
    }
    seed_graphs = tuple(
        graph_id for graph_id, _ in sorted(
            graph_scores.items(), key=lambda item: (-item[1], item[0])
        )[:maximum_seed_graphs]
    )
    grouped: dict[tuple[str, str], list[ControlledPremise]] = {}
    for record in records:
        if record.graph_id not in seed_graphs or _generic_route_label(record.diagnosis_label):
            continue
        if _meaningless_premise(record.text):
            continue
        grouped.setdefault((record.graph_id, record.diagnosis_label), []).append(record)
    route_rows: list[dict[str, object]] = []
    for (graph_id, label), premises in grouped.items():
        ordered = sorted(
            premises,
            key=lambda record: (
                -score_by_source.get(record.source_chunk_id, 0.0),
                record.source_chunk_id,
            ),
        )
        own_score = _multi_evidence_score(
            score_by_source.get(record.source_chunk_id, 0.0)
            for record in premises
        )
        label_terms = set(tokenize(label))
        label_overlap = len(label_terms & set(query)) / max(len(label_terms), 1)
        route_rows.append({
            "graph_id": graph_id,
            "candidate_id": canonical_candidate_id(label),
            "label": label,
            "sources": tuple(record.source_chunk_id for record in ordered[:4]),
            "own_score": own_score,
            "label_overlap": label_overlap,
        })

    candidates: list[tuple[float, str, str, tuple[str, ...]]] = []
    for row in route_rows:
        graph_id = str(row["graph_id"])
        own_score = float(row["own_score"])
        strongest_other = max(
            (
                float(other["own_score"])
                for other in route_rows
                if other is not row and str(other["graph_id"]) == graph_id
            ),
            default=0.0,
        )
        contrastive_margin = own_score - strongest_other
        # Family evidence gets a route into consideration. Route-specific
        # evidence and its margin over sibling subtypes decide which child is
        # ranked first; a weak sibling is penalized rather than inheriting all
        # of its parent's evidence.
        score = (
            own_score
            + 0.35 * graph_scores[graph_id]
            + 0.10 * float(row["label_overlap"])
            + 0.40 * contrastive_margin
        )
        candidates.append((
            score,
            str(row["candidate_id"]),
            str(row["label"]),
            tuple(str(value) for value in row["sources"]),
        ))
    candidates.sort(key=lambda item: (-item[0], item[2].casefold(), item[1]))
    return tuple(
        (candidate_id, label, sources)
        for _, candidate_id, label, sources in candidates[:maximum_candidates]
    )


def select_diverse_candidate_routes(
    candidate_choices: Iterable[tuple[str, str, Iterable[str]]],
    corpus: Iterable[ControlledPremise],
    *,
    query_text: str = "",
    normalizer: UMLSNormalizer | None = None,
    maximum_routes: int = MAX_GRAPH_ROUTES,
    maximum_facts: int = MAX_GRAPH_FACTS,
) -> tuple[dict[str, object], ...]:
    """Select several provenance-bound diagnosis routes without an LLM router.

    Retrieval order remains the relevance signal within each pass. The first
    pass admits the highest-ranked route from each provenance-bound graph;
    duplicate siblings are deferred until every available graph has had an
    opportunity to occupy the eight-route budget. Generic ancestor labels are
    suppressed when the same graph has a more specific retrieved diagnosis.
    Every route receives one fact before extra facts are assigned to candidates
    in sibling-rich families, where discriminating evidence is most useful.
    """
    if maximum_routes < 1 or maximum_facts < maximum_routes:
        raise ValueError("Graph route and fact budgets are inconsistent.")
    corpus_by_source = {record.source_chunk_id: record for record in corpus}
    choices: list[dict[str, object]] = []
    for position, (candidate_id, label, raw_sources) in enumerate(
        candidate_choices,
        start=1,
    ):
        sources = tuple(dict.fromkeys(str(value) for value in raw_sources))
        if not sources or any(source not in corpus_by_source for source in sources):
            raise ValueError("Candidate route cites a source outside the corpus.")
        graph_ids = tuple(dict.fromkeys(
            corpus_by_source[source].graph_id for source in sources
        ))
        choices.append({
            "candidate_id": str(candidate_id),
            "diagnosis_label": " ".join(str(label).split()),
            "sources": sources,
            "graph_ids": graph_ids,
            "family_key": (
                "graph:" + ",".join(sorted(graph_ids))
                if any(graph_ids)
                else "unresolved:" + str(candidate_id)
            ),
            "family_assignment_method": (
                "graph_membership"
                if any(graph_ids)
                else "graph_membership_unavailable"
            ),
            "position": position,
        })

    graphs_with_specific_candidates = {
        graph_id
        for choice in choices
        if not _generic_route_label(str(choice["diagnosis_label"]))
        for graph_id in choice["graph_ids"]
    }
    viable = [
        choice
        for choice in choices
        if not (
            _generic_route_label(str(choice["diagnosis_label"]))
            and any(
                graph_id in graphs_with_specific_candidates
                for graph_id in choice["graph_ids"]
            )
        )
    ]

    selected: list[dict[str, object]] = []
    deferred: list[dict[str, object]] = []
    seen_families: set[str] = set()
    if query_text.strip():
        # Maximum-coverage ordering: retain upstream relevance, but first cover
        # families carrying distinct executable evidence. This only reorders
        # server-owned routes and never creates a candidate or warrant.
        viable.sort(key=lambda choice: (
            -max(
                (_structured_evidence_strength(
                    query_text,
                    corpus_by_source[source].text,
                    normalizer=normalizer,
                ) for source in choice["sources"]),
                default=0.0,
            ),
            int(choice["position"]),
        ))
    for choice in viable:
        family_key = str(choice["family_key"])
        if family_key in seen_families:
            deferred.append(choice)
            continue
        seen_families.add(family_key)
        selected.append(choice)
        if len(selected) == maximum_routes:
            break
    if len(selected) < maximum_routes:
        selected.extend(deferred[:maximum_routes - len(selected)])

    allocated: list[list[str]] = [[] for _ in selected]
    seen_sources: set[str] = set()
    source_offsets = [0] * len(selected)
    selected_graph_counts: Counter[str] = Counter(
        graph_id for choice in selected for graph_id in choice["graph_ids"]
    )

    def allocate_one(index: int) -> bool:
        sources = selected[index]["sources"]
        while source_offsets[index] < len(sources):
            source = sources[source_offsets[index]]
            source_offsets[index] += 1
            if source in seen_sources:
                continue
            seen_sources.add(source)
            allocated[index].append(source)
            return True
        return False

    # Coverage first: no diagnosis route is starved of its primary warrant.
    for index in range(len(selected)):
        allocate_one(index)

    while sum(map(len, allocated)) < maximum_facts:
        eligible = [
            index for index in range(len(selected))
            if source_offsets[index] < len(selected[index]["sources"])
        ]
        if not eligible:
            break
        eligible.sort(key=lambda index: (
            len(allocated[index]),
            -max(
                (
                    selected_graph_counts[graph_id]
                    for graph_id in selected[index]["graph_ids"]
                ),
                default=1,
            ),
            int(selected[index]["position"]),
        ))
        if not any(allocate_one(index) for index in eligible):
            break

    alternatives_by_family: dict[str, list[dict[str, object]]] = {}
    for choice in viable:
        family_key = str(choice["family_key"])
        source = str(choice["sources"][0])
        premise = corpus_by_source[source]
        alternatives_by_family.setdefault(family_key, []).append({
            "candidate_id": choice["candidate_id"],
            "diagnosis_label": choice["diagnosis_label"],
            "graph_id": premise.graph_id,
            "diagnostic_path": list(premise.diagnostic_path),
            "source_chunk_ids": list(choice["sources"]),
            "original_candidate_rank": choice["position"],
        })

    return tuple(
        {
            "rank": rank,
            "candidate_id": choice["candidate_id"],
            "diagnosis_label": choice["diagnosis_label"],
            "source_chunk_ids": sources,
            "family_key": choice["family_key"],
            "family_assignment_method": choice["family_assignment_method"],
            "original_candidate_rank": choice["position"],
            "family_alternatives": [
                {
                    **alternative,
                    "representative": (
                        alternative["candidate_id"] == choice["candidate_id"]
                    ),
                }
                for alternative in alternatives_by_family[
                    str(choice["family_key"])
                ]
            ],
        }
        for rank, (choice, sources) in enumerate(zip(selected, allocated), start=1)
        if sources
    )


class IndependentFlatPremiseRetriever:
    """Independent premise-level BM25 comparator over the controlled corpus."""

    retriever_id = FLAT_RETRIEVER_ID
    normalizer = None

    def __init__(self, corpus: Iterable[ControlledPremise]) -> None:
        self._corpus = tuple(corpus)
        self._tokens = tuple(
            Counter(tokenize(" ".join((record.text, record.diagnosis_label))))
            for record in self._corpus
        )
        self._lengths = tuple(max(sum(values.values()), 1) for values in self._tokens)
        self._average_length = (
            sum(self._lengths) / len(self._lengths) if self._lengths else 1.0
        )
        self._document_frequency: Counter[str] = Counter()
        for values in self._tokens:
            self._document_frequency.update(values.keys())

    def retrieve(self, case: ClinicalCase, *, top_k: int) -> RetrievalBundle:
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        query_tokens = tokenize(case.text)
        query = Counter(query_tokens)
        document_count = max(len(self._corpus), 1)
        ranked: list[tuple[float, ControlledPremise]] = []
        for record, terms, length in zip(
            self._corpus, self._tokens, self._lengths, strict=True
        ):
            score = 0.0
            normalizer = 1.2 * (0.25 + 0.75 * length / self._average_length)
            for term, query_frequency in query.items():
                frequency = terms.get(term, 0)
                if not frequency:
                    continue
                document_frequency = self._document_frequency.get(term, 0)
                inverse = math.log(
                    1.0
                    + (document_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                score += (
                    inverse
                    * frequency
                    * 2.2
                    / (frequency + normalizer)
                    * min(query_frequency, 2)
                )
            if score > 0.0:
                ranked.append((score, record))
        ranked.sort(key=lambda item: (-item[0], item[1].source_chunk_id))
        facts = tuple(
            RetrievedFact(
                evidence_id=f"K{index}",
                node_id=record.node_id,
                category=record.category,
                diagnosis_label=record.diagnosis_label,
                premise_type=record.premise_type,
                text=record.text,
                score=round(score, 6),
                diagnostic_path=record.diagnostic_path,
                knowledge_source_ids=record.knowledge_source_ids,
                source_chunk_id=record.source_chunk_id,
            )
            for index, (score, record) in enumerate(ranked[:top_k], 1)
        )
        return RetrievalBundle(facts=facts, query_tokens=query_tokens)


class FixedGraphRagKnowledgeRetriever:
    """Adapt audited GraphRAG results to the repository's immutable bundle."""

    retriever_id = RETRIEVER_ID
    def __init__(
        self,
        corpus: Iterable[ControlledPremise],
        case_outputs: Iterable[Mapping[str, object]],
        *,
        normalizer: UMLSNormalizer | None = None,
    ) -> None:
        self.normalizer = normalizer
        self._corpus = {
            record.source_chunk_id: record for record in corpus
        }
        self._outputs = {
            str(output.get("case_id") or ""): output
            for output in case_outputs
        }
        self._allowlists: dict[str, tuple[str, ...]] = {}
        for case_id, output in self._outputs.items():
            allowlist = validate_bundle_citation_allowlist(
                (),
                output.get("citation_allowlist") or (),
            )
            if str(output.get("citation_allowlist_sha256") or "") != (
                canonical_sha256(list(allowlist))
            ):
                raise ValueError("GraphRAG citation allowlist hash changed.")
            self._allowlists[case_id] = allowlist

    def retrieve(self, case: ClinicalCase, *, top_k: int) -> RetrievalBundle:
        output = self._outputs.get(case.case_id)
        if output is None:
            raise KeyError(f"No frozen GraphRAG output for {case.case_id}.")
        if output.get("error"):
            raise RuntimeError(str(output["error"]))
        selected: list[tuple[int, ControlledPremise]] = []
        seen: set[str] = set()
        allowlist = self._allowlists[case.case_id]
        allowed = set(allowlist)
        family_routes: list[RetrievedFamilyRoute] = []
        for raw_route in output.get("family_routes") or ():
            alternatives: list[FamilyDiagnosisAlternative] = []
            graph_id = str(raw_route.get("graph_id") or "")
            for raw_alternative in raw_route.get("alternatives") or ():
                source_ids = tuple(
                    str(value)
                    for value in raw_alternative.get("source_chunk_ids") or ()
                )
                if (
                    not source_ids
                    or any(source not in allowed for source in source_ids)
                    or any(source not in self._corpus for source in source_ids)
                    or any(
                        self._corpus[source].graph_id != graph_id
                        for source in source_ids
                    )
                ):
                    raise ValueError("Family alternative escaped its selected graph.")
                alternatives.append(FamilyDiagnosisAlternative(
                    candidate_id=str(raw_alternative.get("candidate_id") or ""),
                    diagnosis_label=str(
                        raw_alternative.get("diagnosis_label") or ""
                    ),
                    graph_id=graph_id,
                    diagnostic_path=tuple(
                        str(value)
                        for value in raw_alternative.get("diagnostic_path") or ()
                    ),
                    source_chunk_ids=source_ids,
                    original_candidate_rank=int(
                        raw_alternative.get("original_candidate_rank") or 0
                    ),
                    representative=bool(raw_alternative.get("representative")),
                ))
            if not graph_id or not alternatives or sum(
                alternative.representative for alternative in alternatives
            ) != 1:
                raise ValueError("Family route has no unique representative.")
            family_routes.append(RetrievedFamilyRoute(
                family_rank=int(raw_route.get("family_rank") or 0),
                graph_id=graph_id,
                family_key=str(raw_route.get("family_key") or ""),
                representative_diagnosis=str(
                    raw_route.get("representative_diagnosis") or ""
                ),
                alternatives=tuple(alternatives),
            ))

        ranked_bindings = tuple(output.get("ranked_candidates") or ())
        # One warrant per selected graph first, preserving the eight-family
        # contract independently of the downstream evidence budget.
        for candidate in ranked_bindings:
            rank = int(candidate.get("rank") or len(selected) + 1)
            bindings = tuple(candidate.get("kg_bindings") or ())
            for binding in bindings[:1]:
                source_id = str(binding.get("source_chunk_id") or "")
                premise = self._corpus.get(source_id)
                if premise is None:
                    raise ValueError(
                        f"GraphRAG cited an unknown source chunk: {source_id}"
                    )
                if source_id not in allowed:
                    raise ValueError(
                        "GraphRAG binding escaped the case citation allowlist."
                    )
                candidate_key = label_key(
                    str(candidate.get("diagnosis_label") or "")
                )
                # Candidate choices are frozen from diagnosis-bearing records on
                # the source's controlled KG path.  The source's focal document
                # label is therefore only one valid path option; requiring focal
                # equality here incorrectly rejects an already provenance-bound
                # ancestor/path candidate.  Keep this exact and fail closed: no
                # fuzzy, semantic, nearest-label, or gold-assisted matching.
                if not candidate_key or not any(
                    label_key(path_label) == candidate_key
                    for path_label in premise.diagnostic_path
                ):
                    raise ValueError(
                        "GraphRAG binding does not support its candidate."
                    )
                if source_id in seen:
                    continue
                seen.add(source_id)
                selected.append((rank, premise))
        # Then expose the best retrieved sibling diagnoses from those same
        # families. They consume evidence capacity, never retrieval-family slots.
        sibling_groups = []
        for route in family_routes:
            representative_rank = min(
                alternative.original_candidate_rank
                for alternative in route.alternatives
                if alternative.representative
            )
            siblings = tuple(sorted(
                (
                    alternative
                    for alternative in route.alternatives
                    if not alternative.representative
                ),
                key=lambda alternative: (
                    alternative.original_candidate_rank,
                    alternative.candidate_id,
                ),
            ))
            if siblings:
                sibling_groups.append((
                    siblings[0].original_candidate_rank - representative_rank,
                    route.family_rank,
                    siblings,
                ))
        sibling_groups.sort(key=lambda item: (item[0], item[1]))
        # Ambiguity-balanced round robin: give several close families one
        # discriminating sibling before giving any family a second sibling.
        sibling_alternatives = []
        for depth in range(max((len(item[2]) for item in sibling_groups), default=0)):
            sibling_alternatives.extend(
                (family_rank, siblings[depth])
                for _, family_rank, siblings in sibling_groups
                if depth < len(siblings)
            )
        for family_rank, alternative in sibling_alternatives:
            if len(selected) == top_k:
                break
            source_id = next(
                (source for source in alternative.source_chunk_ids if source not in seen),
                "",
            )
            if source_id:
                seen.add(source_id)
                selected.append((family_rank, self._corpus[source_id]))
        # Any remaining capacity returns to additional warrants on the selected
        # representative routes.
        for candidate in ranked_bindings:
            if len(selected) == top_k:
                break
            rank = int(candidate.get("rank") or len(selected) + 1)
            for binding in tuple(candidate.get("kg_bindings") or ())[1:]:
                source_id = str(binding.get("source_chunk_id") or "")
                if source_id in seen:
                    continue
                if source_id not in allowed or source_id not in self._corpus:
                    raise ValueError("GraphRAG extra warrant escaped its allowlist.")
                seen.add(source_id)
                selected.append((rank, self._corpus[source_id]))
                if len(selected) == top_k:
                    break
        facts = tuple(
            RetrievedFact(
                evidence_id=f"K{index}",
                node_id=premise.node_id,
                category=premise.category,
                diagnosis_label=premise.diagnosis_label,
                premise_type=premise.premise_type,
                text=premise.text,
                score=round(1.0 / rank, 6),
                diagnostic_path=premise.diagnostic_path,
                knowledge_source_ids=premise.knowledge_source_ids,
                source_chunk_id=premise.source_chunk_id,
            )
            for index, (rank, premise) in enumerate(selected, 1)
        )
        selected_graph_ids = {route.graph_id for route in family_routes}
        child_records = tuple(sorted(
            (
                premise
                for premise in self._corpus.values()
                if premise.graph_id in selected_graph_ids
                and premise.source_chunk_id in allowed
            ),
            key=lambda premise: (
                premise.graph_id,
                premise.diagnostic_path,
                premise.node_id,
                premise.source_chunk_id,
            ),
        ))
        family_child_facts = tuple(
            FamilyChildFact(
                graph_id=premise.graph_id,
                fact=RetrievedFact(
                    evidence_id=f"KF{index}",
                    node_id=premise.node_id,
                    category=premise.category,
                    diagnosis_label=premise.diagnosis_label,
                    premise_type=premise.premise_type,
                    text=premise.text,
                    score=0.0,
                    diagnostic_path=premise.diagnostic_path,
                    knowledge_source_ids=premise.knowledge_source_ids,
                    source_chunk_id=premise.source_chunk_id,
                ),
            )
            for index, premise in enumerate(child_records, 1)
        )
        if not facts:
            return RetrievalBundle(
                facts=(),
                query_tokens=tokenize(case.text),
                citation_allowlist=allowlist,
                family_routes=tuple(family_routes),
                family_child_facts=family_child_facts,
            )
        validate_bundle_citation_allowlist(
            (fact.source_chunk_id for fact in facts),
            allowlist,
        )
        return RetrievalBundle(
            facts=facts,
            query_tokens=tokenize(case.text),
            citation_allowlist=allowlist,
            family_routes=tuple(family_routes),
            family_child_facts=family_child_facts,
        )
