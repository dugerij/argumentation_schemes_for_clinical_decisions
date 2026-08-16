from __future__ import annotations

import json
from typing import Any


DECISIONS = (
    "preserve_candidate_family",
    "reject_different_family",
    "uncertain",
)

GENERIC_PAIRS = (
    ("seropositive inflammatory arthritis", "inflammatory arthritis", "preserve"),
    ("stage 3 chronic kidney disease", "chronic kidney disease", "preserve"),
    ("chest discomfort", "coronary artery disease", "reject"),
    ("bacterial meningitis", "osteoarthritis", "reject"),
)

SYSTEM_PROMPT = """Compare a base diagnosis A with a candidate response B.
If either is only a symptom, sign, cause, complication, risk factor, associated
condition, comorbidity, mimic, or differential of the other, reject immediately
and stop. If they are different diseases, reject immediately and stop. Otherwise
preserve only when "A is a kind of B" or "B is a kind of A" is medically true
because one is a parent, subtype, stage, severity form, within-disease risk form,
or anatomical form of the other. "Kind of" never means symptom of, caused by,
leads to, occurs with, resembles, shares an organ with, or belongs to the same
clinical pathway as. Return uncertain when the labels are insufficient. Do not
diagnose a patient. You may reason before answering, but end with exactly one
JSON object whose only field is `decision` and whose value is exactly
`preserve_candidate_family`, `reject_different_family`, or `uncertain`."""


def decision_schema(order: tuple[str, ...] = DECISIONS) -> dict[str, object]:
    if set(order) != set(DECISIONS) or len(order) != len(DECISIONS):
        raise ValueError("Diagnostic decision order must contain each choice once.")
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {"decision": {"type": "string", "enum": list(order)}},
        "required": ["decision"],
    }


def diagnostic_plan() -> tuple[dict[str, str], ...]:
    return tuple(
        {"base": base, "candidate": candidate, "expected": expected}
        for base, candidate, expected in GENERIC_PAIRS
    )


def run_mapping_diagnostic(model: Any) -> dict[str, object]:
    conditions = (
        ("structured_default_order", decision_schema()),
        ("structured_reversed_order", decision_schema(tuple(reversed(DECISIONS)))),
        ("unconstrained", None),
    )
    results: list[dict[str, object]] = []
    for pair in diagnostic_plan():
        user_prompt = json.dumps({
            "base_diagnosis": pair["base"],
            "candidate_response": pair["candidate"],
        }, sort_keys=True)
        for condition, schema in conditions:
            raw = model.complete(
                SYSTEM_PROMPT,
                user_prompt,
                output_schema=schema,
                max_output_tokens=1536 if schema is None else 128,
            )
            parsed: object = None
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                pass
            results.append({
                **pair,
                "condition": condition,
                "schema_enum_order": (
                    list(schema["properties"]["decision"]["enum"])
                    if schema else None
                ),
                "raw_response": raw,
                "parsed_response": parsed,
            })
    return {
        "diagnostic_id": "medgemma-family-mapping-structured-vs-free-v1",
        "synthetic_clinical_labels_only": True,
        "gold_or_patient_data_used": False,
        "request_count": len(results),
        "results": results,
    }
