from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from clinical_cds.direct import label_key
from clinical_cds.normalization import UMLSNormalizer
from clinical_cds.schema import (
    ClinicalCase,
    DiagnosticGraph,
    GraphNode,
    RetrievalBundle,
    RetrievedFact,
)


TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "patient",
    "she",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "with",
}
SECTION_EVIDENCE_IDS = {
    "chief_complaint": "S-CC",
    "history_of_present_illness": "S-HPI",
    "past_medical_history": "S-PMH",
    "family_history": "S-FH",
    "physical_exam": "S-PE",
    "pertinent_results": "S-RESULTS",
    "patient_info": "S-DEMOGRAPHICS",
    "initial_vitals": "S-VITALS",
    "tests": "S-TESTS",
    "question": "S-QUESTION",
}
RETRIEVER_VERSION = "bm25-diagnosis-route-v1"


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in TOKEN_RE.findall(text.casefold())
        if len(token) > 1 and token not in STOPWORDS
    )


@dataclass(frozen=True)
class _IndexedPremise:
    node: GraphNode
    term_counts: Counter[str]
    length: int


@dataclass(frozen=True)
class _IndexedRoute:
    category: str
    diagnosis_label: str
    path: tuple[str, ...]
    premises: tuple[_IndexedPremise, ...]
    term_counts: Counter[str]
    length: int


class KnowledgeRetriever:
    def __init__(
        self,
        graphs: Iterable[DiagnosticGraph],
        *,
        k1: float = 1.2,
        b: float = 0.75,
        normalizer: UMLSNormalizer | None = None,
    ):
        self.graphs = tuple(graphs)
        self.k1 = k1
        self.b = b
        self.normalizer = normalizer
        self._documents: tuple[_IndexedPremise, ...] = self._build_documents()
        self._document_frequency = self._term_document_frequency(self._documents)
        self._average_length = (
            sum(document.length for document in self._documents) / len(self._documents)
            if self._documents
            else 1.0
        )
        self._routes: tuple[_IndexedRoute, ...] = self._build_routes()
        self._route_document_frequency = self._term_document_frequency(self._routes)
        self._average_route_length = (
            sum(route.length for route in self._routes) / len(self._routes)
            if self._routes
            else 1.0
        )

    @property
    def retriever_id(self) -> str:
        if self.normalizer is None:
            return RETRIEVER_VERSION
        return f"{RETRIEVER_VERSION}+{self.normalizer.normalizer_id}"

    @lru_cache(maxsize=8192)
    def _tokens(self, text: str) -> tuple[str, ...]:
        if self.normalizer is None:
            return tokenize(text)
        expansions = self.normalizer.expand_text(text)
        return tokenize(" ".join((text, *expansions)))

    def _build_documents(self) -> tuple[_IndexedPremise, ...]:
        documents: list[_IndexedPremise] = []
        for graph in self.graphs:
            for node in graph.nodes:
                if node.kind != "premise" or not node.text or not node.diagnosis_label:
                    continue
                tokens = self._tokens(node.text)
                documents.append(
                    _IndexedPremise(
                        node=node,
                        term_counts=Counter(tokens),
                        length=max(len(tokens), 1),
                    )
                )
        return tuple(documents)

    def _build_routes(self) -> tuple[_IndexedRoute, ...]:
        documents_by_node_id = {
            document.node.node_id: document
            for document in self._documents
        }
        routes: list[_IndexedRoute] = []
        for graph in self.graphs:
            premises_by_diagnosis: defaultdict[
                str,
                list[_IndexedPremise],
            ] = defaultdict(list)
            for node in graph.nodes:
                if node.kind != "premise" or not node.diagnosis_label:
                    continue
                document = documents_by_node_id.get(node.node_id)
                if document is not None:
                    premises_by_diagnosis[label_key(node.diagnosis_label)].append(
                        document
                    )

            leaf_keys = {label_key(label) for label in graph.leaf_labels}
            for node in graph.nodes:
                if node.kind != "diagnosis" or label_key(node.label) not in leaf_keys:
                    continue
                path = graph.diagnostic_paths.get(
                    label_key(node.label),
                    (node.label,),
                )
                premises = tuple(
                    premise
                    for diagnosis in path
                    for premise in premises_by_diagnosis[label_key(diagnosis)]
                )
                if not premises:
                    continue
                term_counts: Counter[str] = Counter()
                for premise in premises:
                    term_counts.update(premise.term_counts)
                routes.append(
                    _IndexedRoute(
                        category=graph.category,
                        diagnosis_label=node.label,
                        path=path,
                        premises=premises,
                        term_counts=term_counts,
                        length=max(sum(premise.length for premise in premises), 1),
                    )
                )
        return tuple(routes)

    @staticmethod
    def _term_document_frequency(
        documents: Iterable[_IndexedPremise | _IndexedRoute],
    ) -> Counter[str]:
        frequencies: Counter[str] = Counter()
        for document in documents:
            frequencies.update(document.term_counts.keys())
        return frequencies

    @staticmethod
    def _inverse_document_frequency(
        term: str,
        *,
        document_count: int,
        document_frequency: Counter[str],
    ) -> float:
        frequency = document_frequency.get(term, 0)
        return math.log(1.0 + (document_count - frequency + 0.5) / (frequency + 0.5))

    def _score(
        self,
        query_terms: Counter[str],
        document: _IndexedPremise | _IndexedRoute,
        *,
        document_count: int,
        document_frequency: Counter[str],
        average_length: float,
    ) -> float:
        score = 0.0
        length_normalizer = self.k1 * (
            1.0 - self.b + self.b * document.length / average_length
        )
        for term, query_frequency in query_terms.items():
            term_frequency = document.term_counts.get(term, 0)
            if not term_frequency:
                continue
            score += (
                self._inverse_document_frequency(
                    term,
                    document_count=document_count,
                    document_frequency=document_frequency,
                )
                * (term_frequency * (self.k1 + 1.0))
                / (term_frequency + length_normalizer)
                * min(query_frequency, 2)
            )
        return score

    def retrieve(
        self,
        case: ClinicalCase,
        *,
        top_k: int = 6,
        max_per_diagnosis: int | None = None,
    ) -> RetrievalBundle:
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        query_tokens = self._tokens(case.text)
        query_terms = Counter(query_tokens)
        ranked_routes = [
            (
                self._score(
                    query_terms,
                    route,
                    document_count=len(self._routes),
                    document_frequency=self._route_document_frequency,
                    average_length=self._average_route_length,
                ),
                route,
            )
            for route in self._routes
        ]
        ranked_routes = [
            item for item in ranked_routes if item[0] > 0.0
        ]
        ranked_routes.sort(
            key=lambda item: (
                -item[0],
                item[1].category.casefold(),
                item[1].diagnosis_label.casefold(),
            )
        )

        selected_routes = ranked_routes[:top_k]
        selected: list[
            tuple[float, _IndexedRoute, _IndexedPremise]
        ] = []
        remaining = top_k
        for route_index, (route_score, route) in enumerate(selected_routes):
            routes_left = len(selected_routes) - route_index
            quota = max(1, math.ceil(remaining / routes_left))
            if max_per_diagnosis is not None:
                quota = min(quota, max_per_diagnosis)
            ranked_premises = [
                (
                    self._score(
                        query_terms,
                        premise,
                        document_count=len(self._documents),
                        document_frequency=self._document_frequency,
                        average_length=self._average_length,
                    ),
                    premise,
                )
                for premise in route.premises
            ]
            ranked_premises.sort(
                key=lambda item: (
                    -item[0],
                    item[1].node.node_id,
                )
            )
            for premise_score, premise in ranked_premises[:quota]:
                selected.append(
                    (route_score + premise_score, route, premise)
                )
                remaining -= 1
            if remaining <= 0:
                break

        facts = tuple(
            RetrievedFact(
                evidence_id=f"K{index}",
                node_id=premise.node.node_id,
                category=route.category,
                diagnosis_label=route.diagnosis_label,
                premise_type=premise.node.premise_type or "Clinical premise",
                text=premise.node.text or premise.node.label,
                score=round(score, 6),
                diagnostic_path=route.path,
            )
            for index, (score, route, premise) in enumerate(selected, start=1)
        )
        return RetrievalBundle(
            facts=facts,
            query_tokens=tuple(sorted(set(query_tokens))),
        )


def section_evidence(case: ClinicalCase) -> tuple[tuple[str, str, str], ...]:
    output: list[tuple[str, str, str]] = []
    used_ids: set[str] = set()
    for index, (section_name, text) in enumerate(case.sections.items(), start=1):
        if not text.strip():
            continue
        evidence_id = SECTION_EVIDENCE_IDS.get(section_name, f"S-OTHER-{index}")
        if evidence_id in used_ids:
            evidence_id = f"{evidence_id}-{index}"
        used_ids.add(evidence_id)
        output.append((evidence_id, section_name, text))
    return tuple(output)


def render_case(case: ClinicalCase) -> str:
    lines = [
        f"{evidence_id} | {section_name.replace('_', ' ').title()} | {text}"
        for evidence_id, section_name, text in section_evidence(case)
    ]
    if case.options:
        lines.append("Answer options:")
        lines.extend(f"{key}. {value}" for key, value in case.options.items())
    return "\n".join(lines)


def render_flat_retrieval(bundle: RetrievalBundle) -> str:
    if not bundle.facts:
        return "No guideline premises were retrieved."
    return "\n".join(
        (
            f"{fact.evidence_id} | diagnosis={fact.diagnosis_label} | "
            f"premise_type={fact.premise_type} | premise={fact.text}"
        )
        for fact in bundle.facts
    )


def render_graph_retrieval(bundle: RetrievalBundle) -> str:
    if not bundle.facts:
        return "No guideline subgraph was retrieved."
    lines: list[str] = []
    for fact in bundle.facts:
        path = " -> ".join(fact.diagnostic_path)
        lines.append(
            f"{fact.evidence_id} --supports--> {fact.diagnosis_label} "
            f"| type={fact.premise_type} | premise={fact.text}"
        )
        lines.append(f"PATH {fact.evidence_id}: {path}")
    return "\n".join(lines)
