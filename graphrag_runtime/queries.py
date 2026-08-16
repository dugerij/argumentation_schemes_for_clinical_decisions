from __future__ import annotations

from typing import Any, Iterable

from clinical_cds.direct import DirectDataset
from graphrag_runtime.retrieval_text import atomic_retrieval_segments


SECTION_EVIDENCE_IDS = {
    "chief_complaint": "S-CC",
    "history_of_present_illness": "S-HPI",
    "past_medical_history": "S-PMH",
    "family_history": "S-FH",
    "physical_exam": "S-EXAM",
    "pertinent_results": "S-RESULTS",
}
PRIMARY_RETRIEVAL_SECTIONS = (
    "chief_complaint",
    "history_of_present_illness",
    "physical_exam",
)
LOWER_RETRIEVAL_SECTIONS = (
    "pertinent_results",
    "past_medical_history",
    "family_history",
)
FILTER_SECTION_ORDER = (
    "pertinent_results",
    "physical_exam",
    "history_of_present_illness",
    "chief_complaint",
    "past_medical_history",
    "family_history",
)
FORBIDDEN_RETRIEVAL_KEYS = {
    "gold",
    "gold_label",
    "target",
    "target_label",
    "evaluation_gold_label",
    "evaluation_gold_category",
}


def _contains_forbidden_retrieval_key(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            str(key).casefold() in FORBIDDEN_RETRIEVAL_KEYS
            or _contains_forbidden_retrieval_key(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_retrieval_key(child) for child in value)
    return False


def patient_concept_retrieval_query(payload: dict[str, Any]) -> str:
    """Render only patient-supplied clinical evidence for GraphRAG search."""
    if _contains_forbidden_retrieval_key(payload):
        raise ValueError("Evaluation-only fields cannot enter the retrieval query.")
    evidence = payload.get("patient_evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("Retrieval requires non-empty patient evidence.")
    lines: list[str] = []
    seen_ids: set[str] = set()
    for item in evidence:
        if not isinstance(item, dict):
            raise ValueError("Patient evidence record is malformed.")
        evidence_id = str(item.get("evidence_id") or "")
        section = " ".join(str(item.get("section") or "").split())
        text = " ".join(str(item.get("text") or "").split())
        if (
            not evidence_id.startswith("S-")
            or evidence_id in seen_ids
            or not section
            or not text
        ):
            raise ValueError("Patient evidence identifiers and text must be stable.")
        seen_ids.add(evidence_id)
        lines.append(f"{section}: {text}")
    return "Clinical evidence for differential diagnosis:\n" + "\n".join(lines)


def patient_finding_retrieval_queries(payload: dict[str, Any]) -> tuple[str, ...]:
    """Return separate patient findings for dense nearest-neighbour retrieval."""
    records = _validated_patient_evidence(payload)
    return tuple(
        f"{record['section'].replace('_', ' ')}: {segment}"
        for record in records
        for segment in atomic_retrieval_segments(record["text"])
    )


def _validated_patient_evidence(
    payload: dict[str, Any],
) -> tuple[dict[str, str], ...]:
    """Return stable patient evidence without permitting evaluation fields."""
    if _contains_forbidden_retrieval_key(payload):
        raise ValueError("Evaluation-only fields cannot enter retrieval queries.")
    evidence = payload.get("patient_evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("Retrieval requires non-empty patient evidence.")
    records: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_sections: set[str] = set()
    for item in evidence:
        if not isinstance(item, dict):
            raise ValueError("Patient evidence record is malformed.")
        evidence_id = str(item.get("evidence_id") or "")
        section = " ".join(str(item.get("section") or "").split())
        canonical_section = section.casefold().replace(" ", "_")
        text = " ".join(str(item.get("text") or "").split())
        if (
            not evidence_id.startswith("S-")
            or evidence_id in seen_ids
            or canonical_section in seen_sections
            or canonical_section not in SECTION_EVIDENCE_IDS
            or not text
        ):
            raise ValueError("Patient evidence identifiers and sections must be stable.")
        seen_ids.add(evidence_id)
        seen_sections.add(canonical_section)
        records.append({
            "evidence_id": evidence_id,
            "section": canonical_section,
            "text": text,
        })
    return tuple(records)


def initial_retrieval_sections(payload: dict[str, Any]) -> tuple[str, ...]:
    """Choose the initial retrieval layer without consulting labels or outcomes.

    Complaint, HPI, and examination jointly define the preferred clinical
    presentation.  When none of them is supplied, every available lower-layer
    section is promoted into the initial query so sparse records remain usable.
    """
    records = _validated_patient_evidence(payload)
    available = {record["section"] for record in records}
    primary = tuple(
        section for section in PRIMARY_RETRIEVAL_SECTIONS if section in available
    )
    if primary:
        return primary
    fallback = tuple(
        section for section in LOWER_RETRIEVAL_SECTIONS if section in available
    )
    if not fallback:
        raise ValueError("No supported patient section is available for retrieval.")
    return fallback


def initial_context_query(payload: dict[str, Any]) -> str:
    """Build the broad query from the deterministic adaptive initial layer."""
    records = _validated_patient_evidence(payload)
    initial_sections = initial_retrieval_sections(payload)
    selected = [
        record for section in initial_sections
        for record in records if record["section"] == section
    ]
    return "Initial clinical context for broad retrieval:\n" + "\n".join(
        f'{record["section"]}: {record["text"]}' for record in selected
    )


def presentation_context_query(payload: dict[str, Any]) -> str:
    """Use the complete submitted clinical state for broad GraphRAG retrieval.

    The previous primary-section query excluded pertinent results, past history,
    and family history from source discovery. Later section filters could rerank
    an existing source but could not recover a source that initial retrieval had
    missed. The broad query now uses every submitted section while preserving
    the same gold-label isolation boundary.
    """
    records = _validated_patient_evidence(payload)
    return "Clinical evidence for differential diagnosis:\n" + "\n".join(
        f'{record["section"]}: {record["text"]}' for record in records
    )


def context_filter_queries(
    payload: dict[str, Any],
) -> tuple[tuple[str, str, str], ...]:
    """Build ordered queries for section-aware retrieval fusion."""
    records = _validated_patient_evidence(payload)
    by_section = {record["section"]: record for record in records}
    return tuple(
        (
            section,
            by_section[section]["evidence_id"],
            f'{section}: {by_section[section]["text"]}',
        )
        for section in FILTER_SECTION_ORDER
        if section in by_section
    )


def build_validation_queries(
    direct: DirectDataset,
    case_ids: Iterable[str],
) -> tuple[dict[str, Any], ...]:
    expected_ids = tuple(case_ids)
    cases_by_id = {case.case_id: case for case in direct.cases}
    missing = tuple(case_id for case_id in expected_ids if case_id not in cases_by_id)
    if missing:
        raise ValueError(f"Frozen validation cases are missing: {missing}")
    queries: list[dict[str, Any]] = []
    for case_id in expected_ids:
        case = cases_by_id[case_id]
        patient_evidence = [
            {
                "evidence_id": SECTION_EVIDENCE_IDS[section],
                "section": section,
                "text": text,
            }
            for section, text in case.sections.items()
            if section in SECTION_EVIDENCE_IDS and text.strip()
        ]
        if not patient_evidence:
            raise ValueError(f"Validation case has no patient evidence: {case_id}")
        saved_query_payload = {
            "case_id": case_id,
            "task": "diagnosis",
            "patient_evidence": patient_evidence,
            "instruction": (
                "Use only the indexed controlled guideline corpus. Return JSON "
                "with at most eight ranked_candidates; each candidate must have "
                "diagnosis_label and source_chunk_ids copied from the retrieved "
                "GraphRAG context. Return abstain=true only with an empty list."
            ),
        }
        queries.append({
            "case_id": case_id,
            "saved_query_payload": saved_query_payload,
            "evaluation_gold_label": case.gold_label,
            "evaluation_gold_category": case.disease_category,
        })
    return tuple(queries)
