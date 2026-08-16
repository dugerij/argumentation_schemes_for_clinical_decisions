from __future__ import annotations

import json
from dataclasses import replace

import pytest

from graphrag_runtime.audit import (
    FORBIDDEN_QUERY_KEYS,
    bind_candidate_choices_strict,
    candidate_choice_response_format,
    RANKED_RESPONSE_FORMAT,
    RANKED_RESPONSE_JSON_SCHEMA,
    VLLM_RANKED_RESPONSE_JSON_SCHEMA,
    audit_five_case_retrieval,
    bind_ranked_candidates,
    bind_ranked_candidates_strict,
    parse_candidate_choice_response,
    parse_ranked_response,
)
from graphrag_runtime.provenance_contract import canonical_candidate_id
from graphrag_runtime.corpus import ControlledPremise
from graphrag_runtime.queries import FORBIDDEN_RETRIEVAL_KEYS


def test_post_hoc_leakage_gate_recognises_every_key_the_query_builder_refuses():
    # The audit gate is meant to be an independent second check on top of
    # graphrag_runtime.queries' own forbidden-key refusal. If its key set
    # were a strict subset of the builder's, a payload containing e.g.
    # "gold" or "target" nested under saved_query_payload would pass this
    # gate silently should the builder's own check ever be bypassed.
    assert FORBIDDEN_RETRIEVAL_KEYS <= FORBIDDEN_QUERY_KEYS


def _premise(*, diagnosis: str = "Diagnosis A") -> ControlledPremise:
    return ControlledPremise(
        id="source-chunk:one",
        title="Category A | Diagnosis A | criteria",
        text="A finding",
        graph_id="direct:category-a",
        category="Category A",
        node_id="premise:a",
        diagnosis_label=diagnosis,
        premise_type="criteria",
        diagnostic_path=("Category A", diagnosis),
        source_chunk_id="source-chunk:one",
        knowledge_source_ids=("direct-supplied-graph:direct:category-a",),
        source_origin="supplied_direct_diagnostic_guideline_graph",
    )


def _bound_case(case_id: str, premise: ControlledPremise) -> dict[str, object]:
    parsed = parse_ranked_response(json.dumps({
        "ranked_candidates": [{
            "diagnosis_label": premise.diagnosis_label,
            "source_chunk_ids": [premise.source_chunk_id],
        }],
        "abstain": False,
    }))
    return {
        "case_id": case_id,
        "evaluation_gold_label": premise.diagnosis_label,
        "saved_query_payload": {
            "patient_evidence": [{"evidence_id": "S-HPI", "text": "finding"}]
        },
        "ranked_candidates": list(
            bind_ranked_candidates(parsed, (premise,), ("S-HPI",))
        ),
        "citation_allowlist": [premise.source_chunk_id],
        "abstain": False,
        "error": None,
    }


def test_response_parser_and_binding_require_complete_sources():
    premise = _premise()
    case = _bound_case("case-1", premise)

    binding = case["ranked_candidates"][0]
    assert binding["patient_evidence_ids"] == ["S-HPI"]
    assert binding["kg_bindings"][0]["node_id"] == premise.node_id
    assert binding["kg_bindings"][0]["diagnostic_path"]
    assert binding["kg_bindings"][0]["knowledge_source_ids"]


def test_response_parser_rejects_unknown_or_unbound_sources():
    premise = _premise()
    parsed = parse_ranked_response(json.dumps({
        "ranked_candidates": [{
            "diagnosis_label": premise.diagnosis_label,
            "source_chunk_ids": ["source-chunk:unknown"],
        }],
        "abstain": False,
    }))

    with pytest.raises(ValueError, match="outside the corpus"):
        bind_ranked_candidates(parsed, (premise,), ("S-HPI",))


def test_five_case_audit_passes_equivalent_gates():
    premise = _premise()
    case_ids = tuple(f"case-{index}" for index in range(5))
    cases = [_bound_case(case_id, premise) for case_id in case_ids]

    report = audit_five_case_retrieval(
        cases,
        (premise,),
        expected_case_ids=case_ids,
    )

    assert report["gates"]["development_authorized"] is True
    assert report["semantic_family_recall_at_3"]["rate"] == 1.0
    assert report["routed_candidate_recall_at_8"]["rate"] == 1.0
    assert report["selected_route_exact_recall_at_4"]["rate"] == 1.0
    assert report["selected_route_family_recall_at_4"]["rate"] == 1.0
    assert report["selected_route_count"] == {
        "mean": 1.0,
        "minimum": 1,
        "maximum": 1,
    }
    assert report["selected_fact_count"]["mean"] == 1.0
    assert report["selected_graph_count"]["mean"] == 1.0
    assert report["complete_provenance_count"] == 5


def test_five_case_audit_fails_label_leakage_and_missing_provenance():
    premise = _premise()
    case_ids = tuple(f"case-{index}" for index in range(5))
    cases = [_bound_case(case_id, premise) for case_id in case_ids]
    cases[0]["saved_query_payload"]["gold_label"] = premise.diagnosis_label
    cases[1]["ranked_candidates"][0]["kg_bindings"] = []

    report = audit_five_case_retrieval(
        cases,
        (premise,),
        expected_case_ids=case_ids,
    )

    assert report["gates"]["zero_label_leakage"] is False
    assert report["gates"]["complete_patient_kg_path_source_provenance"] is False
    assert report["gates"]["development_authorized"] is False


def test_parser_refuses_candidate_without_citation():
    with pytest.raises(ValueError, match="source-bound"):
        parse_ranked_response(json.dumps({
            "ranked_candidates": [{"diagnosis_label": "Diagnosis A"}],
            "abstain": False,
        }))


def _valid_response_object() -> dict[str, object]:
    return {
        "ranked_candidates": [{
            "diagnosis_label": "Diagnosis A",
            "source_chunk_ids": ["source-chunk:one"],
        }],
        "abstain": False,
    }


@pytest.mark.parametrize(
    "render",
    [
        lambda value: value,
        lambda value: f"```json\n{value}\n```",
        lambda value: f"```\n{value}\n```",
        lambda value: f"```JSON\n{value}\n```",
    ],
)
def test_parser_accepts_bare_or_one_exact_json_fence(render):
    raw = json.dumps(_valid_response_object())
    parsed = parse_ranked_response(render(raw))

    assert parsed["ranked_candidates"][0]["source_chunk_ids"] == [
        "source-chunk:one"
    ]
    assert parsed["abstain"] is False


@pytest.mark.parametrize(
    "response",
    [
        'prefix\n```json\n{"ranked_candidates": [], "abstain": true}\n```',
        '```json\n{"ranked_candidates": [], "abstain": true}\n```\nsuffix',
        '```json\n{"ranked_candidates": [], "abstain": true}\n```\n```',
        '```json\n{"ranked_candidates": [], "note": "```", "abstain": true}\n```',
        '```json\n{"ranked_candidates": [], "abstain": true}\n'
        '{"ranked_candidates": [], "abstain": true}\n```',
        '```json\n[]\n```',
        '```json\ntrue\n```',
        '```json\n{"ranked_candidates": [}\n```',
        '{"ranked_candidates": [], "abstain": true} trailing',
        'not-json {"ranked_candidates": [], "abstain": true}',
        '```python\n{"ranked_candidates": [], "abstain": true}\n```',
        '``` json\n{"ranked_candidates": [], "abstain": true}\n```',
    ],
)
def test_parser_rejects_ambiguous_or_non_contract_envelopes(response):
    with pytest.raises(ValueError, match="response contract"):
        parse_ranked_response(response)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop("abstain"),
        lambda value: value.update({"abstain": 0}),
        lambda value: value.update({"evaluation_gold_label": "forbidden"}),
        lambda value: value["ranked_candidates"][0].update(
            {"evaluation_gold_category": "forbidden"}
        ),
        lambda value: value["ranked_candidates"][0].update(
            {"source_chunk_ids": [1]}
        ),
    ],
)
def test_parser_rejects_schema_or_gold_field_injection(mutation):
    value = _valid_response_object()
    mutation(value)
    with pytest.raises(ValueError):
        parse_ranked_response(json.dumps(value))


def test_structured_output_schema_is_strict_and_matches_parser_contract():
    assert RANKED_RESPONSE_FORMAT["type"] == "json_schema"
    assert RANKED_RESPONSE_FORMAT["json_schema"]["strict"] is True
    assert RANKED_RESPONSE_FORMAT["json_schema"]["schema"] is (
        VLLM_RANKED_RESPONSE_JSON_SCHEMA
    )
    assert RANKED_RESPONSE_JSON_SCHEMA["additionalProperties"] is False
    assert set(RANKED_RESPONSE_JSON_SCHEMA["required"]) == {
        "ranked_candidates",
        "abstain",
    }
    candidate_schema = RANKED_RESPONSE_JSON_SCHEMA["properties"][
        "ranked_candidates"
    ]["items"]
    assert candidate_schema["additionalProperties"] is False
    assert candidate_schema["properties"]["source_chunk_ids"]["uniqueItems"] is True
    decoder_sources = VLLM_RANKED_RESPONSE_JSON_SCHEMA["properties"][
        "ranked_candidates"
    ]["items"]["properties"]["source_chunk_ids"]
    assert "uniqueItems" not in decoder_sources


def test_parser_retains_uniqueness_after_decoder_schema_compatibility():
    value = _valid_response_object()
    source_id = value["ranked_candidates"][0]["source_chunk_ids"][0]
    value["ranked_candidates"][0]["source_chunk_ids"] = [source_id, source_id]
    with pytest.raises(ValueError, match="unique and source-bound"):
        parse_ranked_response(json.dumps(value))


def test_parser_and_binding_reject_invalid_source_identifier_provenance():
    premise = _premise()
    parsed = parse_ranked_response(json.dumps({
        "ranked_candidates": [{
            "diagnosis_label": premise.diagnosis_label,
            "source_chunk_ids": ["source-chunk:outside"],
        }],
        "abstain": False,
    }))

    with pytest.raises(ValueError, match="outside the corpus"):
        bind_ranked_candidates(parsed, (premise,), ("S-HPI",))


def _strict_premise(*, diagnosis: str = "Diagnosis A") -> ControlledPremise:
    source_id = "source-chunk:" + "a" * 20
    return ControlledPremise(
        id=source_id,
        title="Category A | Diagnosis A | criteria",
        text="A finding",
        graph_id="direct:category-a",
        category="Category A",
        node_id="premise:a",
        diagnosis_label=diagnosis,
        premise_type="criteria",
        diagnostic_path=("Category A", diagnosis),
        source_chunk_id=source_id,
        knowledge_source_ids=("direct-supplied-graph:direct:category-a",),
        source_origin="supplied_direct_diagnostic_guideline_graph",
    )


def _strict_parsed(source_id: str, *, diagnosis: str = "Diagnosis A"):
    return parse_ranked_response(json.dumps({
        "ranked_candidates": [{
            "diagnosis_label": diagnosis,
            "source_chunk_ids": [source_id],
        }],
        "abstain": False,
    }))


def test_strict_binding_accepts_only_allowlisted_canonical_support():
    premise = _strict_premise()
    result = bind_ranked_candidates_strict(
        _strict_parsed(premise.source_chunk_id),
        (premise,),
        ("S-HPI",),
        allowed_source_chunk_ids=(premise.source_chunk_id,),
    )
    assert result[0]["kg_bindings"][0]["source_chunk_id"] == (
        premise.source_chunk_id
    )
    assert result[0]["citation_allowlist_sha256"]


@pytest.mark.parametrize("citation", ["7", "row:7", "source-chunk:nearest"])
def test_strict_binding_rejects_legacy_malformed_or_nearest_match(citation):
    premise = _strict_premise()
    with pytest.raises(ValueError, match="canonical"):
        bind_ranked_candidates_strict(
            _strict_parsed(citation),
            (premise,),
            ("S-HPI",),
            allowed_source_chunk_ids=(premise.source_chunk_id,),
        )


def test_strict_binding_rejects_canonical_source_absent_from_case_allowlist():
    premise = _strict_premise()
    other = "source-chunk:" + "b" * 20
    other_premise = replace(
        premise,
        id=other,
        source_chunk_id=other,
        diagnosis_label="Diagnosis B",
    )
    with pytest.raises(ValueError, match="case allowlist"):
        bind_ranked_candidates_strict(
            _strict_parsed(premise.source_chunk_id),
            (premise, other_premise),
            ("S-HPI",),
            allowed_source_chunk_ids=(other,),
        )


def test_strict_binding_rejects_every_candidate_source_mismatch():
    premise = _strict_premise(diagnosis="Diagnosis B")
    with pytest.raises(ValueError, match="substantively support"):
        bind_ranked_candidates_strict(
            _strict_parsed(premise.source_chunk_id, diagnosis="Diagnosis A"),
            (premise,),
            ("S-HPI",),
            allowed_source_chunk_ids=(premise.source_chunk_id,),
        )


def test_strict_binding_rejects_empty_or_duplicate_allowlist():
    premise = _strict_premise()
    for allowlist in ((), (premise.source_chunk_id, premise.source_chunk_id)):
        with pytest.raises(ValueError, match="non-empty and unique"):
            bind_ranked_candidates_strict(
                _strict_parsed(premise.source_chunk_id),
                (premise,),
                ("S-HPI",),
                allowed_source_chunk_ids=allowlist,
            )


def test_candidate_choice_derives_exact_label_without_fuzzy_matching():
    premise = _strict_premise(diagnosis="Suspected Aortic Dissection")
    candidate_id = canonical_candidate_id(premise.diagnosis_label)
    parsed = parse_candidate_choice_response(json.dumps({
        "ranked_candidates": [{
            "candidate_id": candidate_id,
            "source_chunk_ids": [premise.source_chunk_id],
        }],
        "abstain": False,
    }))
    result = bind_candidate_choices_strict(
        parsed,
        (premise,),
        ("S-HPI",),
        candidate_choices=((
            candidate_id,
            premise.diagnosis_label,
            (premise.source_chunk_id,),
        ),),
    )
    assert result[0]["diagnosis_label"] == "Suspected Aortic Dissection"
    assert result[0]["candidate_id"] == candidate_id


def test_candidate_choice_rejects_unknown_choice_or_cross_candidate_source():
    premise = _strict_premise(diagnosis="Diagnosis A")
    source_b = "source-chunk:" + "b" * 20
    premise_b = replace(
        premise,
        id=source_b,
        source_chunk_id=source_b,
        diagnosis_label="Diagnosis B",
        diagnostic_path=("Category A", "Diagnosis B"),
    )
    choice_a = canonical_candidate_id("Diagnosis A")
    choice_b = canonical_candidate_id("Diagnosis B")
    choices = (
        (choice_a, "Diagnosis A", (premise.source_chunk_id,)),
        (choice_b, "Diagnosis B", (source_b,)),
    )
    cross_source = parse_candidate_choice_response(json.dumps({
        "ranked_candidates": [{
            "candidate_id": choice_a,
            "source_chunk_ids": [source_b],
        }],
        "abstain": False,
    }))
    with pytest.raises(ValueError, match="another choice"):
        bind_candidate_choices_strict(
            cross_source,
            (premise, premise_b),
            ("S-HPI",),
            candidate_choices=choices,
        )
    unknown = parse_candidate_choice_response(json.dumps({
        "ranked_candidates": [{
            "candidate_id": canonical_candidate_id("Unknown"),
            "source_chunk_ids": [premise.source_chunk_id],
        }],
        "abstain": False,
    }))
    with pytest.raises(ValueError, match="outside"):
        bind_candidate_choices_strict(
            unknown,
            (premise, premise_b),
            ("S-HPI",),
            candidate_choices=choices,
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"ranked_candidates": [{"candidate_id": "candidate:notcanonical", "source_chunk_ids": ["source-chunk:" + "a" * 20]}], "abstain": False},
        {"ranked_candidates": [{"candidate_id": canonical_candidate_id("Diagnosis A"), "source_chunk_ids": ["source-chunk:" + "a" * 20], "diagnosis_label": "Diagnosis A"}], "abstain": False},
        {"ranked_candidates": [{"candidate_id": canonical_candidate_id("Diagnosis A"), "source_chunk_ids": ["source-chunk:" + "a" * 20]}], "abstain": True},
        {"ranked_candidates": [], "abstain": False},
        {"ranked_candidates": [], "abstain": True, "evaluation_gold_label": "forbidden"},
    ],
)
def test_candidate_choice_parser_fails_closed(payload):
    with pytest.raises(ValueError):
        parse_candidate_choice_response(json.dumps(payload))


def test_candidate_choice_decoder_schema_requires_canonical_namespaces():
    from graphrag_runtime.audit import CANDIDATE_CHOICE_RESPONSE_FORMAT

    candidate = CANDIDATE_CHOICE_RESPONSE_FORMAT["json_schema"]["schema"][
        "properties"
    ]["ranked_candidates"]["items"]["properties"]
    assert candidate["candidate_id"]["pattern"] == (
        r"^candidate:[0-9a-f]{20}$"
    )
    assert candidate["source_chunk_ids"]["items"]["pattern"] == (
        r"^source-chunk:[0-9a-f]{20}$"
    )


def test_dynamic_candidate_schema_encodes_only_valid_case_pairs():
    candidate_a = canonical_candidate_id("Diagnosis A")
    candidate_b = canonical_candidate_id("Diagnosis B")
    source_a = "source-chunk:" + "a" * 20
    source_b = "source-chunk:" + "b" * 20
    response_format = candidate_choice_response_format({
        candidate_a: (source_a,),
        candidate_b: (source_b,),
    })
    branches = response_format["json_schema"]["schema"]["oneOf"]
    assert len(branches) == 3
    first = branches[0]["properties"]["ranked_candidates"]["items"][
        "properties"
    ]
    assert first["candidate_id"]["enum"] == [candidate_a]
    assert first["source_chunk_ids"]["items"]["enum"] == [source_a]
    assert source_b not in first["source_chunk_ids"]["items"]["enum"]
    assert branches[-1]["properties"]["ranked_candidates"]["maxItems"] == 0


def test_dynamic_candidate_schema_allows_one_kg_source_on_multiple_path_candidates():
    source = "source-chunk:" + "a" * 20
    response_format = candidate_choice_response_format({
        canonical_candidate_id("Diagnosis A"): (source,),
        canonical_candidate_id("Diagnosis B"): (source,),
    })
    branches = response_format["json_schema"]["schema"]["oneOf"]
    assert len(branches) == 3
    assert all(
        branch["properties"]["ranked_candidates"]["items"]["properties"]
        ["source_chunk_ids"]["items"]["enum"] == [source]
        for branch in branches[:2]
    )


def test_retrieval_metrics_use_context_candidates_even_when_model_abstains():
    premise = _premise()
    case = _bound_case("case-1", premise)
    case["ranked_candidates"] = []
    case["abstain"] = True
    case["retrieved_candidate_labels"] = [premise.diagnosis_label]
    case["retrieved_candidate_graph_ids"] = [premise.graph_id]

    report = audit_five_case_retrieval(
        (case,),
        (premise,),
        expected_case_ids=("case-1",),
    )

    assert report["semantic_family_recall_at_3"]["count"] == 1
    assert report["routed_candidate_recall_at_8"]["count"] == 1
    assert report["complete_provenance_count"] == 1
    assert report["abstention_count"] == 1


def test_retrieval_reports_exact_and_semantic_candidate_recall_separately():
    premise = _premise(diagnosis="Acute Coronary Syndrome")
    case = _bound_case("case-1", premise)
    case["evaluation_gold_label"] = "NSTEMI"
    case["retrieved_candidate_labels"] = ["Acute Coronary Syndrome"]
    case["retrieved_candidate_graph_ids"] = [premise.graph_id]

    report = audit_five_case_retrieval(
        (case,),
        (premise,),
        expected_case_ids=("case-1",),
    )

    assert report["routed_candidate_recall_at_8"]["rate"] == 0.0
    assert report["semantic_candidate_recall_at_8"]["rate"] == 1.0
    assert report["semantic_family_recall_at_3"]["rate"] == 1.0
    assert report["gates"]["development_authorized"] is True
    assert report["metric_roles"]["semantic_family_recall_at_3"] == (
        "primary_retrieval_gate"
    )
    assert report["metric_roles"]["family_recall_at_8"] == (
        "primary_retrieval_gate"
    )
    assert report["metric_roles"]["semantic_candidate_recall_at_8"] == (
        "legacy_label_family_diagnostic"
    )
    assert report["metric_roles"]["routed_candidate_recall_at_8"] == (
        "supplementary_exact_label_metric"
    )


def test_retrieval_exact_recall_uses_frozen_nstemi_alias():
    premise = _premise(
        diagnosis="Non-ST Elevation Myocardial Infarction (NSTEMI)"
    )
    case = _bound_case("case-1", premise)
    case["evaluation_gold_label"] = "NSTEMI"
    case["retrieved_candidate_labels"] = [premise.diagnosis_label]
    case["retrieved_candidate_graph_ids"] = [premise.graph_id]

    report = audit_five_case_retrieval(
        (case,),
        (premise,),
        expected_case_ids=("case-1",),
    )

    assert report["routed_candidate_recall_at_8"]["rate"] == 1.0
    assert report["semantic_candidate_recall_at_8"]["rate"] == 1.0
    parse_candidate_choice_response,
