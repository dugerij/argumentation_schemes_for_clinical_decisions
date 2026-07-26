from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from clinical_cds.direct import normalize_label
from clinical_cds.schema import ClinicalCase


def load_patient_case(path: Path) -> ClinicalCase:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    sections = payload.get("sections")
    if not isinstance(sections, dict) or not sections:
        raise ValueError("Patient input must contain a non-empty sections object.")
    normalized_sections = {
        str(name).strip().casefold().replace(" ", "_"): normalize_label(str(text))
        for name, text in sections.items()
        if normalize_label(str(text))
    }
    if not normalized_sections:
        raise ValueError("Patient input does not contain any non-empty clinical sections.")
    external_id = str(payload.get("case_id") or "")
    digest = hashlib.sha256(
        json.dumps(normalized_sections, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return ClinicalCase(
        case_id=f"submitted-{digest}",
        dataset="submitted_patient",
        task="diagnosis",
        sections=normalized_sections,
        gold_label=normalize_label(str(payload.get("gold_label") or "")),
        metadata={
            "external_case_id_present": bool(external_id),
            "submitted_state": True,
        },
    )


def remove_case_section(case: ClinicalCase, section_name: str) -> ClinicalCase:
    if section_name not in case.sections:
        raise ValueError(
            f"Section {section_name!r} is not present in case {case.case_id}."
        )
    sections = dict(case.sections)
    sections.pop(section_name)
    return replace(
        case,
        case_id=f"{case.case_id}-without-{section_name}",
        sections=sections,
        metadata={
            **case.metadata,
            "perturbation": "remove_section",
            "removed_section": section_name,
            "source_case_id": case.case_id,
        },
    )
