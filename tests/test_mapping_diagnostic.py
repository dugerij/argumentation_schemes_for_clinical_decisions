import json

import pytest

from job.mapping_diagnostic import (
    DECISIONS,
    decision_schema,
    diagnostic_plan,
    run_mapping_diagnostic,
)


def test_diagnostic_schema_order_is_controlled_without_changing_choices():
    forward = decision_schema()
    reverse = decision_schema(tuple(reversed(DECISIONS)))
    assert forward["properties"]["decision"]["enum"] == list(DECISIONS)
    assert reverse["properties"]["decision"]["enum"] == list(reversed(DECISIONS))


def test_diagnostic_schema_rejects_missing_or_duplicate_choices():
    with pytest.raises(ValueError, match="each choice once"):
        decision_schema((DECISIONS[0], DECISIONS[0], DECISIONS[2]))


def test_diagnostic_plan_uses_only_generic_labels_and_both_expected_classes():
    plan = diagnostic_plan()
    assert len(plan) == 4
    assert {item["expected"] for item in plan} == {"preserve", "reject"}
    assert all(set(item) == {"base", "candidate", "expected"} for item in plan)


def test_diagnostic_executes_three_conditions_per_pair():
    class FakeModel:
        def complete(self, _system, _user, *, output_schema, **_kwargs):
            if output_schema is None:
                return json.dumps({"decision": "reject_different_family"})
            return json.dumps({
                "decision": output_schema["properties"]["decision"]["enum"][0]
            })

    audit = run_mapping_diagnostic(FakeModel())
    assert audit["request_count"] == 12
    assert audit["gold_or_patient_data_used"] is False
    assert {item["condition"] for item in audit["results"]} == {
        "structured_default_order", "structured_reversed_order", "unconstrained",
    }
