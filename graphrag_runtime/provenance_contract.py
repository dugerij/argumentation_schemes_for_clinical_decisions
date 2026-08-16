from __future__ import annotations

import hashlib
import json
import re
from typing import Iterable

from clinical_cds.normalization import (
    UMLSNormalizer,
    normalized_diagnosis_key,
)


CANONICAL_SOURCE_ID_PATTERN = re.compile(r"source-chunk:[0-9a-f]{20}")
CANDIDATE_ID_PATTERN = re.compile(r"candidate:[0-9a-f]{20}")
PROVENANCE_CONTRACT_ID = "canonical-kg-path-candidate-source-choice-v4"


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def require_canonical_source_id(value: object) -> str:
    source_id = str(value or "")
    if not CANONICAL_SOURCE_ID_PATTERN.fullmatch(source_id):
        raise ValueError("Citation is not a canonical controlled source identifier.")
    return source_id


def canonical_candidate_id(
    diagnosis_label: object,
    normalizer: UMLSNormalizer | None = None,
) -> str:
    label = " ".join(str(diagnosis_label or "").split())
    if not label:
        raise ValueError("Canonical candidate label must be non-empty.")
    identity = normalized_diagnosis_key(label, normalizer)
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return f"candidate:{digest}"


def require_canonical_candidate_id(value: object) -> str:
    candidate_id = str(value or "")
    if not CANDIDATE_ID_PATTERN.fullmatch(candidate_id):
        raise ValueError("Candidate is not a canonical controlled candidate ID.")
    return candidate_id


def validate_bundle_citation_allowlist(
    fact_source_ids: Iterable[str],
    allowlist: Iterable[str],
) -> tuple[str, ...]:
    canonical = tuple(require_canonical_source_id(value) for value in allowlist)
    if not canonical or len(canonical) != len(set(canonical)):
        raise ValueError("Citation allowlist must be non-empty and unique.")
    allowed = set(canonical)
    fact_sources = tuple(
        require_canonical_source_id(value) for value in fact_source_ids
    )
    if any(value not in allowed for value in fact_sources):
        raise ValueError("A retrieval fact is outside the case citation allowlist.")
    return canonical
