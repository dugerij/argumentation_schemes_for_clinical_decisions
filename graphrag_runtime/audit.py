from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from typing import Any, Iterable, Mapping

from clinical_cds.direct import label_key
from clinical_cds.evaluation import (
    diagnosis_family_match,
    exact_label_match,
)
from clinical_cds.normalization import UMLSNormalizer

from .corpus import ControlledPremise
from .queries import FORBIDDEN_RETRIEVAL_KEYS


AUDIT_ID = "microsoft-graphrag-retrieval-context-validation-v2"
MINIMUM_FAMILY_RECALL_AT_3 = 0.75
MINIMUM_CANDIDATE_RECALL_AT_8 = 0.80
# The independent leakage gate below must recognise every key the query
# builder (graphrag_runtime.queries) already refuses to forward, plus a few
# audit-only synonyms -- otherwise this "second check" role is illusory and
# it only ever catches what the builder already caught.
FORBIDDEN_QUERY_KEYS = FORBIDDEN_RETRIEVAL_KEYS | {"directory_label"}


def audit_retrieval_checkpoint(
    cases: Iterable[Mapping[str, Any]],
    *,
    expected_case_ids: Iterable[str],
) -> dict[str, Any]:
    """Fail-fast structural checkpoint without consulting evaluation labels."""
    case_list = tuple(cases)
    expected_ids = tuple(expected_case_ids)
    exact_ids = tuple(
        str(case.get("case_id") or "") for case in case_list
    ) == expected_ids
    zero_errors = all(not case.get("error") for case in case_list)
    complete_provenance = all(
        bool(case.get("citation_allowlist"))
        and (
            bool(case.get("ranked_candidates"))
            or bool(case.get("abstain"))
        )
        and all(
            binding.get("patient_evidence_ids")
            and binding.get("kg_bindings")
            and all(
                item.get("node_id")
                and item.get("diagnostic_path")
                and item.get("source_chunk_id")
                and item.get("knowledge_source_ids")
                for item in binding["kg_bindings"]
            )
            for binding in case.get("ranked_candidates") or ()
        )
        for case in case_list
    )
    zero_leakage = all(
        not _contains_forbidden_query_key(case.get("saved_query_payload") or {})
        for case in case_list
    )
    gates = {
        "exact_checkpoint_case_sequence": exact_ids and bool(expected_ids),
        "zero_errors": zero_errors,
        "complete_patient_kg_path_source_provenance": complete_provenance,
        "zero_label_leakage": zero_leakage,
    }
    return {
        "audit_id": "graphrag-query-only-five-case-structural-checkpoint-v1",
        "case_count": len(case_list),
        "case_ids": list(expected_ids),
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "evaluation_labels_consulted": False,
    }

RANKED_RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ranked_candidates": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "diagnosis_label": {"type": "string", "minLength": 1},
                    "source_chunk_ids": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {"type": "string", "minLength": 1},
                    },
                },
                "required": ["diagnosis_label", "source_chunk_ids"],
                "additionalProperties": False,
            },
        },
        "abstain": {"type": "boolean"},
    },
    "required": ["ranked_candidates", "abstain"],
    "additionalProperties": False,
}
VLLM_RANKED_RESPONSE_JSON_SCHEMA = deepcopy(RANKED_RESPONSE_JSON_SCHEMA)
# vLLM 0.18.1's structured-output grammar rejects JSON Schema's
# ``uniqueItems`` keyword. Uniqueness remains mandatory and is enforced by
# ``parse_ranked_response`` after generation; the decoder schema only omits
# this unsupported optimization hint.
del VLLM_RANKED_RESPONSE_JSON_SCHEMA["properties"]["ranked_candidates"][
    "items"
]["properties"]["source_chunk_ids"]["uniqueItems"]

RANKED_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "graphrag_ranked_response",
        "strict": True,
        "schema": VLLM_RANKED_RESPONSE_JSON_SCHEMA,
    },
}
GRAPHRAG_RANKED_RESPONSE_CONTRACT_ID = (
    "graphrag-ranked-json-schema-v2-vllm-compatible"
)
GRAPHRAG_RANKED_RESPONSE_MARKER = (
    f"OUTPUT CONTRACT ID: {GRAPHRAG_RANKED_RESPONSE_CONTRACT_ID}"
)

CANDIDATE_CHOICE_RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ranked_candidates": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "candidate_id": {
                        "type": "string",
                        "pattern": r"^candidate:[0-9a-f]{20}$",
                    },
                    "source_chunk_ids": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {
                            "type": "string",
                            "pattern": r"^source-chunk:[0-9a-f]{20}$",
                        },
                    },
                },
                "required": ["candidate_id", "source_chunk_ids"],
                "additionalProperties": False,
            },
        },
        "abstain": {"type": "boolean"},
    },
    "required": ["ranked_candidates", "abstain"],
    "additionalProperties": False,
}
VLLM_CANDIDATE_CHOICE_RESPONSE_JSON_SCHEMA = deepcopy(
    CANDIDATE_CHOICE_RESPONSE_JSON_SCHEMA
)
del VLLM_CANDIDATE_CHOICE_RESPONSE_JSON_SCHEMA["properties"][
    "ranked_candidates"
]["items"]["properties"]["source_chunk_ids"]["uniqueItems"]
CANDIDATE_CHOICE_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "graphrag_candidate_choice_response",
        "strict": True,
        "schema": VLLM_CANDIDATE_CHOICE_RESPONSE_JSON_SCHEMA,
    },
}
GRAPHRAG_CANDIDATE_CHOICE_CONTRACT_ID = (
    "graphrag-canonical-kg-path-candidate-choice-v4-dynamic-case-enums"
)
GRAPHRAG_CANDIDATE_CHOICE_MARKER = (
    f"OUTPUT CONTRACT ID: {GRAPHRAG_CANDIDATE_CHOICE_CONTRACT_ID}"
)


def candidate_choice_response_format(
    candidate_sources: Mapping[str, Iterable[str]],
) -> dict[str, Any]:
    """Build a decoder schema containing only this rendered case's choices.

    The response deliberately contains at most one candidate. Candidate Recall@8
    is a retrieval-context measure and is computed from the rendered candidate
    allowlist, not by asking the completion model to reproduce that allowlist.
    """
    from graphrag_runtime.provenance_contract import (
        require_canonical_candidate_id,
        require_canonical_source_id,
    )

    branches: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_candidate, raw_sources in candidate_sources.items():
        candidate_id = require_canonical_candidate_id(raw_candidate)
        sources = tuple(require_canonical_source_id(value) for value in raw_sources)
        if (
            candidate_id in seen
            or not sources
            or len(sources) != len(set(sources))
        ):
            raise ValueError("Dynamic candidate/source choices are invalid.")
        seen.add(candidate_id)
        branches.append({
            "type": "object",
            "properties": {
                "ranked_candidates": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "candidate_id": {
                                "type": "string",
                                "enum": [candidate_id],
                            },
                            "source_chunk_ids": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": len(sources),
                                "items": {
                                    "type": "string",
                                    "enum": list(sources),
                                },
                            },
                        },
                        "required": ["candidate_id", "source_chunk_ids"],
                        "additionalProperties": False,
                    },
                },
                "abstain": {"type": "boolean", "enum": [False]},
            },
            "required": ["ranked_candidates", "abstain"],
            "additionalProperties": False,
        })
    if not branches:
        raise ValueError("Dynamic candidate/source choices must be non-empty.")
    branches.append({
        "type": "object",
        "properties": {
            "ranked_candidates": {"type": "array", "maxItems": 0},
            "abstain": {"type": "boolean", "enum": [True]},
        },
        "required": ["ranked_candidates", "abstain"],
        "additionalProperties": False,
    })
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "graphrag_case_candidate_choice_response",
            "strict": True,
            "schema": {"oneOf": branches},
        },
    }


def _parse_one_exact_json_fence(response: str) -> dict[str, Any]:
    """Parse one whole-response Markdown fence without searching or repair."""
    stripped = response.strip()
    if not stripped.startswith("```"):
        raise ValueError("GraphRAG response is neither bare JSON nor one JSON fence.")
    opening_end = stripped.find("\n")
    if opening_end < 0:
        raise ValueError("Fenced GraphRAG response has no content line.")
    language = stripped[3:opening_end]
    if language and language.casefold() != "json":
        raise ValueError("GraphRAG response fence language must be json or absent.")
    remainder = stripped[opening_end + 1 :]
    if not remainder.endswith("\n```"):
        raise ValueError("GraphRAG response must end with one standalone fence.")
    content = remainder[:-4]
    if "```" in content:
        raise ValueError("GraphRAG response contains a nested or second fence.")
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("GraphRAG response must be a JSON object.")
    return payload


def parse_ranked_response(response: str) -> dict[str, Any]:
    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        try:
            payload = _parse_one_exact_json_fence(response)
        except (json.JSONDecodeError, ValueError) as fence_error:
            raise ValueError(
                "GraphRAG response violates the exact JSON response contract."
            ) from fence_error
    if not isinstance(payload, dict):
        raise ValueError("GraphRAG response must be a JSON object.")
    if set(payload) != {"ranked_candidates", "abstain"}:
        raise ValueError("GraphRAG response fields differ from the strict schema.")
    raw_candidates = payload.get("ranked_candidates")
    if not isinstance(raw_candidates, list) or len(raw_candidates) > 8:
        raise ValueError("GraphRAG must return at most eight ranked candidates.")
    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for raw in raw_candidates:
        if not isinstance(raw, dict):
            raise ValueError("Each ranked candidate must be an object.")
        required_candidate_fields = {"diagnosis_label", "source_chunk_ids"}
        if required_candidate_fields - set(raw):
            raise ValueError("Ranked candidates must be unique and source-bound.")
        if set(raw) != required_candidate_fields:
            raise ValueError("Ranked candidate fields differ from the strict schema.")
        raw_diagnosis = raw.get("diagnosis_label")
        raw_source_ids = raw.get("source_chunk_ids")
        if not isinstance(raw_diagnosis, str) or not isinstance(raw_source_ids, list):
            raise ValueError("Ranked candidate field types differ from the strict schema.")
        if any(not isinstance(value, str) for value in raw_source_ids):
            raise ValueError("Source chunk identifiers must be strings.")
        diagnosis = " ".join(raw_diagnosis.split())
        source_chunk_ids = tuple(raw_source_ids)
        key = label_key(diagnosis)
        if (
            not diagnosis
            or not source_chunk_ids
            or len(source_chunk_ids) != len(set(source_chunk_ids))
            or key in seen
        ):
            raise ValueError("Ranked candidates must be unique and source-bound.")
        seen.add(key)
        candidates.append({
            "rank": len(candidates) + 1,
            "diagnosis_label": diagnosis,
            "source_chunk_ids": list(source_chunk_ids),
        })
    abstain = payload.get("abstain")
    if not isinstance(abstain, bool):
        raise ValueError("GraphRAG abstain must be a boolean.")
    if abstain == bool(candidates):
        raise ValueError("Abstention must be true exactly when no candidates exist.")
    return {"ranked_candidates": candidates, "abstain": abstain}


def parse_candidate_choice_response(response: str) -> dict[str, Any]:
    """Parse the new exact-choice contract without changing legacy replay."""
    from graphrag_runtime.provenance_contract import (
        require_canonical_candidate_id,
        require_canonical_source_id,
    )

    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        try:
            payload = _parse_one_exact_json_fence(response)
        except (json.JSONDecodeError, ValueError) as fence_error:
            raise ValueError(
                "GraphRAG response violates the candidate-choice JSON contract."
            ) from fence_error
    if not isinstance(payload, dict) or set(payload) != {
        "ranked_candidates", "abstain"
    }:
        raise ValueError("Candidate-choice response fields differ from the schema.")
    raw_candidates = payload.get("ranked_candidates")
    if not isinstance(raw_candidates, list) or len(raw_candidates) > 8:
        raise ValueError("GraphRAG must return at most eight candidate choices.")
    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for raw in raw_candidates:
        if not isinstance(raw, dict) or set(raw) != {
            "candidate_id", "source_chunk_ids"
        }:
            raise ValueError("Candidate-choice fields differ from the strict schema.")
        candidate_id = require_canonical_candidate_id(raw.get("candidate_id"))
        raw_sources = raw.get("source_chunk_ids")
        if not isinstance(raw_sources, list):
            raise ValueError("Candidate source identifiers must be a list.")
        sources = tuple(require_canonical_source_id(value) for value in raw_sources)
        if not sources or len(sources) != len(set(sources)) or candidate_id in seen:
            raise ValueError("Candidate choices must be unique and source-bound.")
        seen.add(candidate_id)
        candidates.append({
            "rank": len(candidates) + 1,
            "candidate_id": candidate_id,
            "source_chunk_ids": list(sources),
        })
    abstain = payload.get("abstain")
    if not isinstance(abstain, bool) or abstain == bool(candidates):
        raise ValueError("Abstention must be true exactly when no candidates exist.")
    return {"ranked_candidates": candidates, "abstain": abstain}


def _contains_forbidden_query_key(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            str(key).casefold() in FORBIDDEN_QUERY_KEYS
            or _contains_forbidden_query_key(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_query_key(child) for child in value)
    return False


def bind_ranked_candidates(
    parsed: dict[str, Any],
    corpus: Iterable[ControlledPremise],
    patient_evidence_ids: Iterable[str],
) -> tuple[dict[str, Any], ...]:
    corpus_by_source = {record.source_chunk_id: record for record in corpus}
    patient_ids = tuple(str(value) for value in patient_evidence_ids)
    if not patient_ids or any(not value.startswith("S-") for value in patient_ids):
        raise ValueError("Patient evidence IDs must be non-empty stable S-* IDs.")
    bound: list[dict[str, Any]] = []
    for candidate in parsed["ranked_candidates"]:
        source_records = tuple(
            corpus_by_source.get(source_id)
            for source_id in candidate["source_chunk_ids"]
        )
        if any(record is None for record in source_records):
            raise ValueError("Candidate cites a source chunk outside the corpus.")
        diagnosis_key = label_key(candidate["diagnosis_label"])
        matching = tuple(
            record
            for record in source_records
            if record is not None
            and label_key(record.diagnosis_label) == diagnosis_key
        )
        if not matching:
            raise ValueError("Candidate diagnosis is not supported by its cited chunks.")
        bound.append({
            **candidate,
            "patient_evidence_ids": list(patient_ids),
            "kg_bindings": [
                {
                    "graph_id": record.graph_id,
                    "node_id": record.node_id,
                    "diagnostic_path": list(record.diagnostic_path),
                    "source_chunk_id": record.source_chunk_id,
                    "knowledge_source_ids": list(record.knowledge_source_ids),
                }
                for record in matching
            ],
        })
    return tuple(bound)


def bind_ranked_candidates_strict(
    parsed: dict[str, Any],
    corpus: Iterable[ControlledPremise],
    patient_evidence_ids: Iterable[str],
    *,
    allowed_source_chunk_ids: Iterable[str],
) -> tuple[dict[str, Any], ...]:
    """Bind only canonical, allowlisted, diagnosis-supporting citations.

    This is the post-v5 provenance contract. The historical binder above is
    retained solely so preserved failure-analysis artifacts keep their exact
    original semantics.
    """
    from graphrag_runtime.provenance_contract import (
        canonical_sha256,
        require_canonical_source_id,
    )

    corpus_by_source = {record.source_chunk_id: record for record in corpus}
    patient_ids = tuple(str(value) for value in patient_evidence_ids)
    if not patient_ids or any(not value.startswith("S-") for value in patient_ids):
        raise ValueError("Patient evidence IDs must be non-empty stable S-* IDs.")
    allowlist = tuple(
        require_canonical_source_id(value) for value in allowed_source_chunk_ids
    )
    if not allowlist or len(allowlist) != len(set(allowlist)):
        raise ValueError("Per-case citation allowlist must be non-empty and unique.")
    allowed = set(allowlist)
    if any(source_id not in corpus_by_source for source_id in allowlist):
        raise ValueError("Citation allowlist contains a source outside the corpus.")

    bound: list[dict[str, Any]] = []
    for candidate in parsed["ranked_candidates"]:
        cited = tuple(
            require_canonical_source_id(value)
            for value in candidate["source_chunk_ids"]
        )
        if any(source_id not in allowed for source_id in cited):
            raise ValueError("Candidate cites a source outside its case allowlist.")
        source_records = tuple(corpus_by_source[source_id] for source_id in cited)
        diagnosis_key = label_key(candidate["diagnosis_label"])
        if any(
            label_key(record.diagnosis_label) != diagnosis_key
            for record in source_records
        ):
            raise ValueError(
                "Every cited source must substantively support its candidate."
            )
        bound.append({
            **candidate,
            "patient_evidence_ids": list(patient_ids),
            "citation_allowlist_sha256": canonical_sha256(list(allowlist)),
            "kg_bindings": [
                {
                    "graph_id": record.graph_id,
                    "node_id": record.node_id,
                    "diagnostic_path": list(record.diagnostic_path),
                    "source_chunk_id": record.source_chunk_id,
                    "knowledge_source_ids": list(record.knowledge_source_ids),
                }
                for record in source_records
            ],
        })
    return tuple(bound)


def bind_candidate_choices_strict(
    parsed: dict[str, Any],
    corpus: Iterable[ControlledPremise],
    patient_evidence_ids: Iterable[str],
    *,
    candidate_choices: Iterable[tuple[str, str, Iterable[str]]],
    normalizer: UMLSNormalizer | None = None,
) -> tuple[dict[str, Any], ...]:
    """Bind exact candidate IDs; labels are derived, never fuzzy-matched."""
    from graphrag_runtime.provenance_contract import (
        canonical_candidate_id,
        canonical_sha256,
        require_canonical_candidate_id,
        require_canonical_source_id,
    )

    corpus_by_source = {record.source_chunk_id: record for record in corpus}
    patient_ids = tuple(str(value) for value in patient_evidence_ids)
    if not patient_ids or any(not value.startswith("S-") for value in patient_ids):
        raise ValueError("Patient evidence IDs must be non-empty stable S-* IDs.")
    choice_map: dict[str, tuple[str, tuple[str, ...]]] = {}
    for raw_id, raw_label, raw_sources in candidate_choices:
        candidate_id = require_canonical_candidate_id(raw_id)
        label = " ".join(str(raw_label or "").split())
        sources = tuple(require_canonical_source_id(value) for value in raw_sources)
        if (
            candidate_id in choice_map
            or not label
            or candidate_id != canonical_candidate_id(label, normalizer)
            or not sources
            or len(sources) != len(set(sources))
            or any(source not in corpus_by_source for source in sources)
            or any(
                not any(
                    exact_label_match(path_label, label, normalizer)
                    for path_label in corpus_by_source[source].diagnostic_path
                )
                for source in sources
            )
        ):
            raise ValueError("Candidate choice map is invalid or conflicting.")
        choice_map[candidate_id] = (label, sources)
    if not choice_map:
        raise ValueError("Candidate choice map must be non-empty.")

    bound: list[dict[str, Any]] = []
    for candidate in parsed["ranked_candidates"]:
        candidate_id = require_canonical_candidate_id(candidate["candidate_id"])
        if candidate_id not in choice_map:
            raise ValueError("Candidate ID is outside the case choice allowlist.")
        label, allowed_sources = choice_map[candidate_id]
        cited = tuple(
            require_canonical_source_id(value)
            for value in candidate["source_chunk_ids"]
        )
        if any(source not in set(allowed_sources) for source in cited):
            raise ValueError("Candidate cites a source assigned to another choice.")
        records = tuple(corpus_by_source[source] for source in cited)
        bound.append({
            **candidate,
            "diagnosis_label": label,
            "patient_evidence_ids": list(patient_ids),
            "candidate_choice_allowlist_sha256": canonical_sha256([
                [key, value[0], list(value[1])]
                for key, value in choice_map.items()
            ]),
            "kg_bindings": [
                {
                    "graph_id": record.graph_id,
                    "node_id": record.node_id,
                    "diagnostic_path": list(record.diagnostic_path),
                    "source_chunk_id": record.source_chunk_id,
                    "knowledge_source_ids": list(record.knowledge_source_ids),
                }
                for record in records
            ],
        })
    return tuple(bound)


def audit_five_case_retrieval(
    cases: Iterable[Mapping[str, Any]],
    corpus: Iterable[ControlledPremise],
    *,
    expected_case_ids: Iterable[str],
    normalizer: UMLSNormalizer | None = None,
) -> dict[str, Any]:
    corpus_list = tuple(corpus)
    expected_ids = tuple(expected_case_ids)
    case_list = tuple(cases)
    if tuple(str(case.get("case_id") or "") for case in case_list) != expected_ids:
        raise ValueError("Fallback validation case IDs do not match the frozen order.")
    family_hits = 0
    candidate_hits = 0
    semantic_candidate_hits = 0
    retrieved_exact_32_hits = 0
    retrieved_family_32_hits = 0
    selected_exact_hits = 0
    selected_family_hits = 0
    selected_exact_8_hits = 0
    selected_family_8_hits = 0
    graph_family_8_hits = 0
    family_assignment_methods: Counter[str] = Counter()
    family_deduplicated_cases = 0
    selected_route_counts: list[int] = []
    selected_fact_counts: list[int] = []
    selected_graph_counts: list[int] = []
    errors = 0
    abstentions = 0
    complete_provenance = 0
    label_invariant = 0
    results: list[dict[str, Any]] = []
    for case in case_list:
        error = case.get("error")
        bindings = tuple(case.get("ranked_candidates") or ())
        family_trace = tuple(case.get("family_selection_trace") or ())
        family_assignment_methods.update(
            str(item.get("assignment_method") or "untracked")
            for item in family_trace
        )
        if any(
            int(item.get("selected_rank") or 0)
            != int(item.get("original_candidate_rank") or 0)
            for item in family_trace
        ):
            family_deduplicated_cases += 1
        gold_label = str(case.get("evaluation_gold_label") or "")
        exact_gold_records = tuple(
            record
            for record in corpus_list
            if exact_label_match(record.diagnosis_label, gold_label, normalizer)
        )
        family_gold_records = tuple(
            record
            for record in corpus_list
            if diagnosis_family_match(record.diagnosis_label, gold_label)
        )
        gold_records = exact_gold_records or family_gold_records
        gold_graph_ids = {record.graph_id for record in gold_records}
        if error or not gold_graph_ids:
            errors += 1
        retrieved_graph_ids = tuple(case.get("retrieved_candidate_graph_ids") or ())
        retrieved_labels = tuple(case.get("retrieved_candidate_labels") or ())
        family_top3 = (
            {str(value) for value in retrieved_graph_ids[:3]}
            if retrieved_graph_ids
            else {
                str(binding.get("kg_bindings", [{}])[0].get("graph_id") or "")
                for binding in bindings[:3]
                if binding.get("kg_bindings")
            }
        )
        candidate_top8 = (
            tuple(str(value) for value in retrieved_labels[:8])
            if retrieved_labels
            else tuple(
                str(binding.get("diagnosis_label") or "")
                for binding in bindings[:8]
            )
        )
        candidate_top32 = (
            tuple(str(value) for value in retrieved_labels[:32])
            if retrieved_labels
            else tuple(
                str(binding.get("diagnosis_label") or "")
                for binding in bindings[:32]
            )
        )
        family_hit = bool(gold_graph_ids & family_top3)
        candidate_hit = any(
            exact_label_match(label, gold_label, normalizer)
            for label in candidate_top8
        )
        semantic_candidate_hit = any(
            diagnosis_family_match(label, gold_label)
            for label in candidate_top8
        )
        retrieved_exact_32_hit = any(
            exact_label_match(label, gold_label, normalizer)
            for label in candidate_top32
        )
        retrieved_family_32_hit = any(
            diagnosis_family_match(label, gold_label)
            for label in candidate_top32
        )
        selected_labels = tuple(
            str(binding.get("diagnosis_label") or "") for binding in bindings[:4]
        )
        selected_exact_hit = any(
            exact_label_match(label, gold_label, normalizer)
            for label in selected_labels
        )
        selected_family_hit = any(
            diagnosis_family_match(label, gold_label)
            for label in selected_labels
        )
        selected_labels_8 = tuple(
            str(binding.get("diagnosis_label") or "") for binding in bindings[:8]
        )
        selected_exact_8_hit = any(
            exact_label_match(label, gold_label, normalizer)
            for label in selected_labels_8
        )
        selected_family_8_hit = any(
            diagnosis_family_match(label, gold_label)
            for label in selected_labels_8
        )
        selected_sources = {
            str(item.get("source_chunk_id") or "")
            for binding in bindings[:8]
            for item in (binding.get("kg_bindings") or ())
            if item.get("source_chunk_id")
        }
        selected_graphs = {
            str(item.get("graph_id") or "")
            for binding in bindings[:8]
            for item in (binding.get("kg_bindings") or ())
            if item.get("graph_id")
        }
        graph_family_8_hit = bool(gold_graph_ids & selected_graphs)
        provenance_complete = bool(case.get("citation_allowlist")) and (
            bool(case.get("abstain"))
            or (
                bool(bindings)
                and all(
                    binding.get("patient_evidence_ids")
                    and binding.get("kg_bindings")
                    and all(
                        item.get("node_id")
                        and item.get("diagnostic_path")
                        and item.get("source_chunk_id")
                        and item.get("knowledge_source_ids")
                        for item in binding["kg_bindings"]
                    )
                    for binding in bindings
                )
            )
        )
        query_payload = case.get("saved_query_payload") or {}
        invariant = not _contains_forbidden_query_key(query_payload)
        family_hits += family_hit
        candidate_hits += candidate_hit
        semantic_candidate_hits += semantic_candidate_hit
        retrieved_exact_32_hits += retrieved_exact_32_hit
        retrieved_family_32_hits += retrieved_family_32_hit
        selected_exact_hits += selected_exact_hit
        selected_family_hits += selected_family_hit
        selected_exact_8_hits += selected_exact_8_hit
        selected_family_8_hits += selected_family_8_hit
        graph_family_8_hits += graph_family_8_hit
        selected_route_counts.append(len(bindings[:8]))
        selected_fact_counts.append(len(selected_sources))
        selected_graph_counts.append(len(selected_graphs))
        complete_provenance += provenance_complete
        label_invariant += invariant
        abstentions += bool(case.get("abstain"))
        results.append({
            "case_id": case["case_id"],
            "error": error,
            "family_recall_at_3_hit": family_hit,
            "candidate_recall_at_8_hit": candidate_hit,
            "semantic_candidate_recall_at_8_hit": semantic_candidate_hit,
            "retrieved_exact_candidate_recall_at_32_hit": retrieved_exact_32_hit,
            "retrieved_family_candidate_recall_at_32_hit": retrieved_family_32_hit,
            "selected_route_exact_recall_at_4_hit": selected_exact_hit,
            "selected_route_family_recall_at_4_hit": selected_family_hit,
            "selected_route_exact_recall_at_8_hit": selected_exact_8_hit,
            "selected_route_family_recall_at_8_hit": selected_family_8_hit,
            "graph_family_recall_at_8_hit": graph_family_8_hit,
            "selected_route_count": len(bindings[:8]),
            "selected_fact_count": len(selected_sources),
            "selected_graph_count": len(selected_graphs),
            "provenance_complete": provenance_complete,
            "label_invariant": invariant,
            "abstain": bool(case.get("abstain")),
        })
    count = len(case_list)
    family_rate = family_hits / count if count else 0.0
    candidate_rate = candidate_hits / count if count else 0.0
    semantic_candidate_rate = semantic_candidate_hits / count if count else 0.0
    selected_exact_rate = selected_exact_hits / count if count else 0.0
    selected_family_rate = selected_family_hits / count if count else 0.0
    selected_exact_8_rate = selected_exact_8_hits / count if count else 0.0
    selected_family_8_rate = selected_family_8_hits / count if count else 0.0
    graph_family_8_rate = graph_family_8_hits / count if count else 0.0
    retrieved_exact_32_rate = retrieved_exact_32_hits / count if count else 0.0
    retrieved_family_32_rate = retrieved_family_32_hits / count if count else 0.0

    def distribution(values: list[int]) -> dict[str, float | int]:
        return {
            "mean": sum(values) / len(values) if values else 0.0,
            "minimum": min(values) if values else 0,
            "maximum": max(values) if values else 0,
        }
    gates = {
        "exact_frozen_case_set": count == len(expected_ids) and count > 0,
        "zero_errors": errors == 0,
        "semantic_family_recall_at_3": family_rate
        >= MINIMUM_FAMILY_RECALL_AT_3,
        "routed_candidate_recall_at_8": candidate_rate
        >= MINIMUM_CANDIDATE_RECALL_AT_8,
        "semantic_candidate_recall_at_8": semantic_candidate_rate
        >= MINIMUM_CANDIDATE_RECALL_AT_8,
        "graph_family_recall_at_8": graph_family_8_rate
        >= MINIMUM_CANDIDATE_RECALL_AT_8,
        "complete_patient_kg_path_source_provenance": complete_provenance == count,
        "zero_label_leakage": label_invariant == count,
    }
    primary_gate_names = (
        "exact_frozen_case_set",
        "zero_errors",
        "graph_family_recall_at_8",
        "complete_patient_kg_path_source_provenance",
        "zero_label_leakage",
    )
    gates["development_authorized"] = all(
        gates[name] for name in primary_gate_names
    )
    return {
        "audit_id": AUDIT_ID,
        "case_count": count,
        "case_ids": list(expected_ids),
        "error_count": errors,
        "semantic_family_recall_at_3": {
            "count": family_hits,
            "rate": family_rate,
            "minimum": MINIMUM_FAMILY_RECALL_AT_3,
        },
        "routed_candidate_recall_at_8": {
            "count": candidate_hits,
            "rate": candidate_rate,
            "minimum": MINIMUM_CANDIDATE_RECALL_AT_8,
        },
        "retrieved_exact_candidate_recall_at_8": {
            "count": candidate_hits,
            "rate": candidate_rate,
        },
        "retrieved_family_candidate_recall_at_8": {
            "count": semantic_candidate_hits,
            "rate": semantic_candidate_rate,
        },
        "retrieved_exact_candidate_recall_at_32": {
            "count": retrieved_exact_32_hits,
            "rate": retrieved_exact_32_rate,
        },
        "retrieved_family_candidate_recall_at_32": {
            "count": retrieved_family_32_hits,
            "rate": retrieved_family_32_rate,
        },
        "selected_route_exact_recall_at_4": {
            "count": selected_exact_hits,
            "rate": selected_exact_rate,
        },
        "selected_route_family_recall_at_4": {
            "count": selected_family_hits,
            "rate": selected_family_rate,
        },
        "selected_route_exact_recall_at_8": {
            "count": selected_exact_8_hits,
            "rate": selected_exact_8_rate,
        },
        "selected_route_family_recall_at_8": {
            "count": selected_family_8_hits,
            "rate": selected_family_8_rate,
        },
        "family_recall_at_8": {
            "count": graph_family_8_hits,
            "rate": graph_family_8_rate,
            "minimum": MINIMUM_CANDIDATE_RECALL_AT_8,
            "identity_basis": "controlled_graph_membership",
        },
        "family_assignment_audit": {
            "method_counts": dict(sorted(family_assignment_methods.items())),
            "cases_reordered_by_family_deduplication": family_deduplicated_cases,
            "family_identity_policy": "controlled_graph_membership_only",
            "unresolved_policy": (
                "tagged_as_graph_membership_unavailable_and_not_deduplicated"
            ),
        },
        "selected_route_count": distribution(selected_route_counts),
        "selected_fact_count": distribution(selected_fact_counts),
        "selected_graph_count": distribution(selected_graph_counts),
        "semantic_candidate_recall_at_8": {
            "count": semantic_candidate_hits,
            "rate": semantic_candidate_rate,
            "minimum": MINIMUM_CANDIDATE_RECALL_AT_8,
        },
        "metric_roles": {
            "family_recall_at_8": "primary_retrieval_gate",
            "semantic_family_recall_at_3": "primary_retrieval_gate",
            "semantic_candidate_recall_at_8": "legacy_label_family_diagnostic",
            "routed_candidate_recall_at_8": "supplementary_exact_label_metric",
            "retrieved_exact_candidate_recall_at_8": "pre_selection_diagnostic",
            "retrieved_family_candidate_recall_at_8": "pre_selection_diagnostic",
            "retrieved_exact_candidate_recall_at_32": "pre_selection_diagnostic",
            "retrieved_family_candidate_recall_at_32": "pre_selection_diagnostic",
            "selected_route_exact_recall_at_4": "post_selection_diagnostic",
            "selected_route_family_recall_at_4": "post_selection_diagnostic",
            "selected_route_exact_recall_at_8": "post_selection_diagnostic",
            "selected_route_family_recall_at_8": "post_selection_diagnostic",
            "selected_fact_count": "post_selection_evidence_budget_diagnostic",
            "rationale": (
                "Clinical concept/family equivalence is primary; exact wording "
                "is retained for transparency but does not veto an equivalent "
                "diagnosis terminology match."
            ),
        },
        "complete_provenance_count": complete_provenance,
        "label_invariance_count": label_invariant,
        "abstention_count": abstentions,
        "abstention_rate": abstentions / count if count else 0.0,
        "coverage": (count - abstentions) / count if count else 0.0,
        "cases": results,
        "gates": gates,
        "prohibited_partitions_accessed": [],
    }
