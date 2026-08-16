from __future__ import annotations

import pytest

from graphrag_runtime.medgemma_chat_boundary import (
    audit_client_structured_output,
    requested_completion_tokens,
)


def _payload(system: str) -> dict:
    return {
        "model": "google/medgemma-27b-text-it",
        "stream": True,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "aggregate-safe-placeholder"},
        ],
    }


@pytest.mark.parametrize("value", [1, 1024, 1536, 2048, 3072])
def test_request_specific_completion_allowance_accepts_positive_values_up_to_cap(
    value: int,
):
    assert requested_completion_tokens({"max_tokens": value}) == value


@pytest.mark.parametrize("value", [None, True, 0, -1, 3073])
def test_request_specific_completion_allowance_fails_closed(value):
    with pytest.raises(ValueError, match="[Cc]ompletion allowance"):
        requested_completion_tokens({"max_tokens": value})


def test_agent_json_schema_is_forwarded_and_hash_audited() -> None:
    payload = _payload("Generator instructions.")
    payload["structured_outputs"] = {
        "json": {
            "type": "object",
            "properties": {"abstain": {"type": "boolean"}},
            "required": ["abstain"],
            "additionalProperties": False,
        }
    }

    audit = audit_client_structured_output(payload)

    assert audit["client_structured_output_forwarded"] is True
    assert len(audit["client_structured_output_schema_sha256"]) == 64
    assert audit["client_unconstrained_contract_id"] is None


@pytest.mark.parametrize(
    "configured",
    [
        {"json": {"type": "object"}},
        {"json": {"type": "array", "additionalProperties": False}},
        {"json": {"type": "object", "additionalProperties": True}},
        {"json": {}, "regex": ".*"},
    ],
)
def test_agent_json_schema_fails_closed_unless_it_is_one_closed_object(
    configured,
) -> None:
    payload = _payload("Generator instructions.")
    payload["structured_outputs"] = configured

    with pytest.raises(ValueError, match="structured output|closed"):
        audit_client_structured_output(payload)
