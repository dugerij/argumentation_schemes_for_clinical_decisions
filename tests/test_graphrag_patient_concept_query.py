from __future__ import annotations

import pytest

from graphrag_runtime.queries import (
    context_filter_queries,
    initial_retrieval_sections,
    patient_concept_retrieval_query,
    patient_finding_retrieval_queries,
    presentation_context_query,
)


def _payload() -> dict[str, object]:
    return {
        "case_id": "aggregate-safe-case",
        "task": "diagnosis",
        "instruction": "Formatting instructions must not affect retrieval.",
        "patient_evidence": [
            {
                "evidence_id": "S-HPI",
                "section": "history of present illness",
                "text": "clinical concept alpha",
            },
            {
                "evidence_id": "S-RESULTS",
                "section": "pertinent results",
                "text": "clinical concept beta",
            },
        ],
    }


def test_patient_concept_query_contains_only_ordered_clinical_evidence():
    query = patient_concept_retrieval_query(_payload())

    assert query == (
        "Clinical evidence for differential diagnosis:\n"
        "history of present illness: clinical concept alpha\n"
        "pertinent results: clinical concept beta"
    )
    assert "aggregate-safe-case" not in query
    assert "Formatting instructions" not in query


@pytest.mark.parametrize(
    "field",
    ["gold_label", "target_label", "evaluation_gold_label", "evaluation_gold_category"],
)
def test_patient_concept_query_rejects_any_evaluation_field(field: str):
    payload = _payload()
    payload[field] = "must-never-enter-retrieval"

    with pytest.raises(ValueError, match="Evaluation-only"):
        patient_concept_retrieval_query(payload)


def test_patient_concept_query_rejects_duplicate_or_missing_evidence():
    duplicate = _payload()
    duplicate["patient_evidence"] = [
        duplicate["patient_evidence"][0],
        duplicate["patient_evidence"][0],
    ]
    with pytest.raises(ValueError, match="stable"):
        patient_concept_retrieval_query(duplicate)
    with pytest.raises(ValueError, match="non-empty"):
        patient_concept_retrieval_query({"patient_evidence": []})


def test_dense_patient_queries_split_long_results_and_preserve_late_finding():
    payload = _payload()
    payload["patient_evidence"][1]["text"] = (
        "routine laboratory value " * 90
        + "Endoscopy: oesophagitis LA grade B. Ejection fraction LVEF=30%."
    )

    queries = patient_finding_retrieval_queries(payload)

    assert len(queries) > 3
    assert any("oesophagitis LA grade B" in query for query in queries)
    assert any("LVEF=30%" in query for query in queries)
    assert all(len(query.split()) <= 62 for query in queries)


def test_broad_query_uses_complete_clinical_record():
    payload = _payload()
    payload["patient_evidence"] = [
        {
            "evidence_id": "S-CC",
            "section": "chief complaint",
            "text": "clinical concept complaint",
        },
        {
            "evidence_id": "S-EXAM",
            "section": "physical exam",
            "text": "clinical concept exam",
        },
        *payload["patient_evidence"],
    ]
    query = presentation_context_query(payload)
    assert "chief_complaint" in query
    assert "history_of_present_illness" in query
    assert "physical_exam" in query
    assert "pertinent_results" in query
    assert initial_retrieval_sections(payload) == (
        "chief_complaint",
        "history_of_present_illness",
        "physical_exam",
    )


def test_lower_layers_become_initial_query_when_primary_layers_are_absent():
    payload = _payload()
    payload["patient_evidence"] = [
        {
            "evidence_id": "S-RESULTS",
            "section": "pertinent results",
            "text": "clinical concept result",
        },
        {
            "evidence_id": "S-PMH",
            "section": "past medical history",
            "text": "clinical concept history",
        },
    ]
    query = presentation_context_query(payload)
    assert "pertinent_results" in query
    assert "past_medical_history" in query
    assert initial_retrieval_sections(payload) == (
        "pertinent_results",
        "past_medical_history",
    )
    assert [item[0] for item in context_filter_queries(payload)] == [
        "pertinent_results",
        "past_medical_history",
    ]


def test_filter_queries_preserve_frozen_section_order():
    payload = _payload()
    payload["patient_evidence"] = [
        *payload["patient_evidence"],
        {
            "evidence_id": "S-EXAM",
            "section": "physical exam",
            "text": "clinical concept exam",
        },
        {
            "evidence_id": "S-PMH",
            "section": "past medical history",
            "text": "clinical concept history",
        },
    ]
    filters = context_filter_queries(payload)
    assert [item[0] for item in filters] == [
        "pertinent_results",
        "physical_exam",
        "history_of_present_illness",
        "past_medical_history",
    ]
    assert all(text.startswith(f"{section}:") for section, _, text in filters)


@pytest.mark.parametrize("field", ["evaluation_gold_label", "evaluation_gold_category"])
def test_section_queries_reject_gold_fields(field: str):
    payload = _payload()
    payload[field] = "forbidden"
    with pytest.raises(ValueError):
        presentation_context_query(payload)
    with pytest.raises(ValueError):
        context_filter_queries(payload)
