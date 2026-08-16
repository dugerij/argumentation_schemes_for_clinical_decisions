from __future__ import annotations

from dataclasses import replace

from graphrag_runtime.queries import (
    build_validation_queries,
    presentation_context_query,
)


def test_validation_queries_keep_gold_out_of_model_payload(direct_root):
    from clinical_cds.direct import load_direct_dataset

    direct = load_direct_dataset(direct_root)
    cases = tuple(
        replace(case, case_id=f"validation-{index}")
        for index, case in enumerate([direct.cases[0]] * 5)
    )
    direct = replace(direct, cases=cases)
    case_ids = tuple(case.case_id for case in cases)

    queries = build_validation_queries(direct, case_ids)

    assert tuple(query["case_id"] for query in queries) == case_ids
    assert all("evaluation_gold_label" in query for query in queries)
    assert all("gold_label" not in query["saved_query_payload"] for query in queries)
    assert all(
        evidence["evidence_id"].startswith("S-")
        for query in queries
        for evidence in query["saved_query_payload"]["patient_evidence"]
    )


def test_presentation_query_uses_every_submitted_clinical_section():
    payload = {
        "patient_evidence": [
            {
                "evidence_id": "S-HPI",
                "section": "history_of_present_illness",
                "text": "Chest discomfort.",
            },
            {
                "evidence_id": "S-RESULTS",
                "section": "pertinent_results",
                "text": "Troponin is elevated.",
            },
            {
                "evidence_id": "S-PMH",
                "section": "past_medical_history",
                "text": "Prior coronary disease.",
            },
        ]
    }

    query = presentation_context_query(payload)

    assert "Chest discomfort" in query
    assert "Troponin is elevated" in query
    assert "Prior coronary disease" in query
