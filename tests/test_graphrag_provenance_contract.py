from __future__ import annotations

import pytest

from graphrag_runtime.provenance_contract import (
    canonical_candidate_id,
    require_canonical_candidate_id,
    require_canonical_source_id,
    validate_bundle_citation_allowlist,
)


SOURCE_A = "source-chunk:" + "a" * 20
SOURCE_B = "source-chunk:" + "b" * 20


def test_canonical_candidate_id_is_deterministic_and_non_empty():
    assert canonical_candidate_id("Diagnosis A") == canonical_candidate_id(
        "Diagnosis A"
    )
    with pytest.raises(ValueError, match="non-empty"):
        canonical_candidate_id("")


def test_require_canonical_source_id_rejects_non_canonical_values():
    assert require_canonical_source_id(SOURCE_A) == SOURCE_A
    with pytest.raises(ValueError, match="canonical"):
        require_canonical_source_id("not-canonical")


def test_require_canonical_candidate_id_rejects_non_canonical_values():
    candidate_id = canonical_candidate_id("Diagnosis A")
    assert require_canonical_candidate_id(candidate_id) == candidate_id
    with pytest.raises(ValueError, match="canonical"):
        require_canonical_candidate_id("not-canonical")


def test_bundle_allowlist_preserves_order_and_rejects_escape_or_duplicates():
    assert validate_bundle_citation_allowlist(
        (SOURCE_A,),
        (SOURCE_A, SOURCE_B),
    ) == (SOURCE_A, SOURCE_B)
    with pytest.raises(ValueError, match="outside"):
        validate_bundle_citation_allowlist((SOURCE_B,), (SOURCE_A,))
    with pytest.raises(ValueError, match="unique"):
        validate_bundle_citation_allowlist((), (SOURCE_A, SOURCE_A))
