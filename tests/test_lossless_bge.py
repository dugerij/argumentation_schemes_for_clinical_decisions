from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from graphrag_runtime.embedding_boundary import load_exact_bge_tokenizer
from graphrag_runtime.lossless_bge import (
    BGE_MAX_INPUT_TOKENS,
    plan_audit_record,
    plan_lossless_embedding,
    token_weighted_mean_l2,
)


def tokenizer():
    path = Path("/private/tmp/bge-tokenizer-version-i/tokenizer.json")
    if not path.is_file():
        pytest.skip("Exact BGE tokenizer is unavailable.")
    return load_exact_bge_tokenizer(
        "BAAI/bge-small-en-v1.5", tokenizer_json=path
    )


def test_lossless_plan_preserves_every_raw_token_without_overlap():
    value = "clinical evidence node path source chunk " * 800
    exact = tokenizer()
    plan = plan_lossless_embedding(value, exact)
    raw = exact.encode(value, add_special_tokens=False).ids
    recovered = [
        token
        for chunk in plan.chunks
        for token in chunk.token_ids[1:-1]
    ]
    assert recovered == raw
    assert sum(chunk.raw_token_count for chunk in plan.chunks) == len(raw)
    assert all(chunk.model_token_count <= BGE_MAX_INPUT_TOKENS for chunk in plan.chunks)
    assert all(
        left.raw_end == right.raw_start
        for left, right in zip(plan.chunks, plan.chunks[1:], strict=False)
    )


def test_under_limit_input_is_one_unchanged_plan():
    value = "short evidence"
    exact = tokenizer()
    plan = plan_lossless_embedding(value, exact)
    assert plan.unchanged is True
    assert plan.chunk_count == 1
    assert list(plan.chunks[0].token_ids) == exact.encode(
        value, add_special_tokens=True
    ).ids


def test_weighted_mean_preserves_dimension_normalization_and_weighting():
    first = np.zeros(384, dtype=np.float32)
    second = np.zeros(384, dtype=np.float32)
    first[0] = 1
    second[1] = 1
    result = np.asarray(token_weighted_mean_l2([first, second], [3, 1]))
    assert result.shape == (384,)
    assert np.isfinite(result).all()
    assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-6)
    assert result[0] == pytest.approx(3 * result[1], rel=1e-6)


def test_hash_only_audit_has_no_token_ids_or_raw_text():
    plan = plan_lossless_embedding("evidence " * 600, tokenizer())
    audit = plan_audit_record(plan)
    assert "token_ids" not in audit
    assert all("token_ids" not in chunk for chunk in audit["chunks"])
    assert "evidence" not in str(audit)
    assert len(audit["source_sha256"]) == 64
