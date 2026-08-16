from __future__ import annotations

import re
from typing import Any, Iterable


_OPAQUE_REFERENCE_PATTERNS = (
    re.compile(r"\bcandidate:[A-Za-z0-9_.:-]+", re.IGNORECASE),
    re.compile(r"\bsource-chunk:[A-Za-z0-9_.:-]+", re.IGNORECASE),
    re.compile(r"\bSUPPORT-\d+\b", re.IGNORECASE),
    re.compile(r"\b[ASKD]\d+\b"),
    re.compile(r"\b[a-f0-9]{64}\b", re.IGNORECASE),
)
_FORBIDDEN_PRESENTATION_KEYS = {
    "argument_id",
    "candidate_id",
    "case_id",
    "evidence_id",
    "evidence_ids",
    "source_chunk_id",
    "support_option_id",
    "trace_id",
}


def _clean_text(value: object) -> str:
    text = " ".join(str(value or "").split())
    for pattern in _OPAQUE_REFERENCE_PATTERNS:
        text = pattern.sub("internal reference", text)
    return text


def _evidence_basis(values: Iterable[object]) -> list[str]:
    patient = False
    controlled = False
    for value in values:
        text = str(value or "")
        patient = patient or text.startswith("S-")
        controlled = controlled or text.startswith("K")
    output: list[str] = []
    if patient:
        output.append("patient evidence")
    if controlled:
        output.append("controlled-corpus evidence")
    return output


def _error_category(value: object) -> str:
    text = str(value or "").casefold()
    if not text:
        return ""
    if "duplicate diagnosis" in text:
        return "invalid duplicate candidate assessment"
    if "json" in text:
        return "invalid structured response"
    if "citation" in text or "provenance" in text or "evidence" in text:
        return "invalid evidence binding"
    return "technical stage failure"


def assert_presentation_safe(value: object) -> None:
    """Reject opaque execution identifiers in a user-facing artifact."""
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).casefold()
            if normalized in _FORBIDDEN_PRESENTATION_KEYS or normalized.endswith(
                "_sha256"
            ):
                raise ValueError(
                    f"Presentation artifact contains internal key: {key}"
                )
            assert_presentation_safe(child)
        return
    if isinstance(value, list):
        for child in value:
            assert_presentation_safe(child)
        return
    if isinstance(value, str) and any(
        pattern.search(value) for pattern in _OPAQUE_REFERENCE_PATTERNS
    ):
        raise ValueError("Presentation artifact contains an opaque reference.")


def presentation_argument_trace(
    trace: dict[str, Any],
    *,
    case_number: int,
) -> dict[str, Any]:
    """Render a readable stage trace while retaining IDs only in machine audit."""
    proposal = trace.get("proposal") or {}
    argument_diagnosis: dict[str, str] = {}
    generator_candidates: list[dict[str, Any]] = []
    for candidate in proposal.get("candidates") or []:
        diagnosis = _clean_text(candidate.get("diagnosis"))
        arguments: list[dict[str, Any]] = []
        for argument in candidate.get("arguments") or []:
            argument_diagnosis[str(argument.get("argument_id") or "")] = diagnosis
            arguments.append({
                "patient_finding": _clean_text(argument.get("patient_finding")),
                "application": _clean_text(argument.get("application")),
                "rationale": _clean_text(
                    argument.get("model_rationale") or argument.get("premise")
                ),
                "evidence_basis": _evidence_basis(
                    argument.get("evidence_ids") or []
                ),
            })
        generator_candidates.append({
            "diagnosis": diagnosis,
            "arguments": arguments,
        })

    verifier = trace.get("verifier") or {}
    reviews: list[dict[str, Any]] = []
    for review in verifier.get("reviews") or []:
        reviews.append({
            "diagnosis": argument_diagnosis.get(
                str(review.get("argument_id") or ""),
                "",
            ),
            "verdict": _clean_text(review.get("verdict")),
            "explanation": _clean_text(review.get("explanation")),
            "failed_critical_questions": [
                _clean_text(item)
                for item in review.get("failed_critical_questions") or []
            ],
            "evidence_basis": _evidence_basis(review.get("evidence_ids") or []),
        })

    counterarguments: list[dict[str, Any]] = []
    for counterargument in verifier.get("counterarguments") or []:
        counterarguments.append({
            "target_diagnosis": argument_diagnosis.get(
                str(counterargument.get("target_argument_id") or ""),
                "",
            ),
            "premise": _clean_text(counterargument.get("premise")),
            "conclusion": _clean_text(counterargument.get("conclusion")),
            "relation": _clean_text(counterargument.get("relation")),
            "evidence_basis": _evidence_basis(
                counterargument.get("evidence_ids") or []
            ),
        })

    decision = (
        trace.get("resolver_decision")
        or trace.get("llm_adjudication")
        or {}
    )
    selected_arguments = set(decision.get("supporting_argument_ids") or [])
    graph_nodes = (trace.get("argument_graph") or {}).get("nodes") or []
    explanation = " ".join(
        _clean_text(node.get("conclusion") or node.get("premise"))
        for node in graph_nodes
        if str(node.get("argument_id") or "") in selected_arguments
    ).strip()
    output = {
        "case_number": case_number,
        "generator": {
            "candidates": generator_candidates,
            "preferred_diagnosis": _clean_text(
                proposal.get("preferred_diagnosis")
            ),
            "abstained": bool(proposal.get("abstain")),
        },
        "verifier": {
            "reviews": reviews,
            "counterarguments": counterarguments,
            "abstained": bool(verifier.get("abstain")),
        },
        "resolver": {
            "diagnosis": _clean_text(decision.get("selected_diagnosis")),
            "abstained": bool(decision.get("abstained", True)),
            "explanation": explanation,
            "evidence_basis": _evidence_basis(decision.get("evidence_ids") or []),
            "error_category": _error_category(
                decision.get("error") or trace.get("error")
            ),
        },
    }
    assert_presentation_safe(output)
    return output


def presentation_prediction(
    prediction: dict[str, Any],
    *,
    case_number: int,
) -> dict[str, Any]:
    """Render one method response without execution identifiers or gold data."""
    output = {
        "case_number": case_number,
        "method": _clean_text(prediction.get("mode")),
        "diagnosis": _clean_text(prediction.get("predicted_label")),
        "abstained": bool(prediction.get("abstained")),
        "explanation": _clean_text(prediction.get("reasoning")),
        "evidence_basis": _evidence_basis(prediction.get("citations") or []),
        "error_category": _error_category(prediction.get("error")),
    }
    assert_presentation_safe(output)
    return output
