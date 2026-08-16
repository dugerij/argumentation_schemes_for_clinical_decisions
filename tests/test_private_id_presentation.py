import json

import pytest

from clinical_cds.presentation import (
    assert_presentation_safe,
    presentation_argument_trace,
    presentation_prediction,
)


def test_presentation_trace_and_response_hide_internal_references():
    candidate_id = "candidate:" + "a" * 20
    source_id = "source-chunk:" + "b" * 20
    trace = {
        "case_id": "case-private",
        "trace_id": "trace-private",
        "proposal": {
            "preferred_diagnosis": "Diagnosis One",
            "abstain": False,
            "candidates": [{
                "candidate_id": candidate_id,
                "diagnosis": "Diagnosis One",
                "arguments": [{
                    "argument_id": "A1",
                    "patient_finding": "A relevant finding.",
                    "application": "satisfies",
                    "model_rationale": f"Supported by {source_id} and K1.",
                    "evidence_ids": ["S-HPI", "K1"],
                }],
            }],
        },
        "verifier": {
            "abstain": False,
            "reviews": [{
                "argument_id": "A1",
                "verdict": "supported",
                "explanation": f"{candidate_id} is supported.",
                "failed_critical_questions": [],
                "evidence_ids": ["S-HPI", "K1"],
            }],
            "counterarguments": [],
        },
        "argument_graph": {
            "nodes": [{"argument_id": "A1", "conclusion": "Diagnosis One"}]
        },
        "llm_adjudication": {
            "selected_candidate_id": candidate_id,
            "selected_diagnosis": "Diagnosis One",
            "supporting_argument_ids": ["A1"],
            "evidence_ids": ["S-HPI", "K1"],
            "abstained": False,
            "confidence": "high",
            "error": None,
        },
    }
    prediction = {
        "case_id": "case-private",
        "mode": "evidence_grounded_argumentation",
        "predicted_label": "Diagnosis One",
        "reasoning": f"Supported by {candidate_id}, {source_id}, A1 and K1.",
        "citations": ["S-HPI", "K1"],
        "abstained": False,
        "error": None,
    }

    readable_trace = presentation_argument_trace(trace, case_number=1)
    readable_response = presentation_prediction(prediction, case_number=1)
    rendered = json.dumps([readable_trace, readable_response])

    assert "candidate:" not in rendered
    assert "source-chunk:" not in rendered
    assert '"case_id"' not in rendered
    assert '"candidate_id"' not in rendered
    assert '"argument_id"' not in rendered
    assert '"evidence_ids"' not in rendered
    assert_presentation_safe(readable_trace)
    assert_presentation_safe(readable_response)


def test_presentation_safety_rejects_internal_keys_and_values():
    with pytest.raises(ValueError, match="internal key"):
        assert_presentation_safe({"candidate_id": "hidden"})
    with pytest.raises(ValueError, match="opaque reference"):
        assert_presentation_safe({"diagnosis": "candidate:" + "a" * 20})
