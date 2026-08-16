from __future__ import annotations

from pathlib import Path

import pytest

from graphrag_runtime.medgemma_prompt_budget import (
    COMPLETION_TOKENS,
    ADJUDICATOR_COMPLETION_TOKENS,
    ARGUMENT_CRITIC_COMPLETION_TOKENS,
    ARGUMENT_GENERATOR_COMPLETION_TOKENS,
    GRAPHRAG_CONTEXT_TOKENS,
    MAX_MODEL_LEN,
    MAX_COMPLETION_TOKENS,
    MAX_RENDERED_INPUT_TOKENS,
    MEDGEMMA_MODEL_ID,
    PromptBudgetExceeded,
    SAFETY_MARGIN_TOKENS,
    audit_prompt_budget,
    frozen_budget_config,
    load_medgemma_tokenizer,
)


class FixedLengthTokenizer:
    def __init__(self, count: int) -> None:
        self.count = count

    def apply_chat_template(self, _messages, **_kwargs):
        return {"input_ids": list(range(self.count))}


def _official_tokenizer_root() -> Path:
    return Path(
        "/Users/oluwatosinoso/.cache/huggingface/hub/"
        "models--google--medgemma-1.5-4b-it/snapshots/"
        "91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b"
    )


def test_exact_medgemma_chat_template_accounting_is_pinned() -> None:
    root = _official_tokenizer_root()
    if not (root / "tokenizer.json").is_file():
        pytest.skip("Pinned official MedGemma tokenizer is not locally cached.")
    tokenizer = load_medgemma_tokenizer(root, local_files_only=True)
    audit = audit_prompt_budget(
        tokenizer,
        [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "Q"},
        ],
    )

    assert audit.rendered_input_tokens == 12
    assert audit.model_id == MEDGEMMA_MODEL_ID
    assert len(audit.rendered_token_ids_sha256) == 64
    assert audit.message_roles == ("system", "user")


def test_exact_boundary_accepts_the_largest_allowed_rendered_prompt() -> None:
    audit = audit_prompt_budget(
        FixedLengthTokenizer(MAX_RENDERED_INPUT_TOKENS),
        [{"role": "user", "content": "hash-only-test"}],
    )
    assert audit.total_reserved_tokens == MAX_MODEL_LEN
    assert audit.remaining_tokens_after_reservations == 0


def test_exact_boundary_rejects_one_token_over_without_truncation() -> None:
    with pytest.raises(PromptBudgetExceeded, match="exceeds"):
        audit_prompt_budget(
            FixedLengthTokenizer(MAX_RENDERED_INPUT_TOKENS + 1),
            [{"role": "user", "content": "hash-only-test"}],
        )


def test_frozen_context_filter_budget_keeps_separate_output_and_safety_capacity() -> None:
    config = frozen_budget_config()
    assert config == {
        "policy": "exact-medgemma-chat-template-v1",
        "model_id": "google/medgemma-1.5-4b-it",
        "model_revision": "91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b",
        "max_model_len": 16_384,
        "graphrag_context_tokens": 10_000,
        "completion_tokens": 1_536,
        "maximum_request_completion_tokens": 3_072,
        "stage_completion_tokens": {
            "standard": 1_536,
            "argument_generator": 3_072,
            "argument_critic": 2_048,
            "adjudicator": 1_024,
            "semantic_judge": 512,
        },
        "safety_margin_tokens": 1_024,
        "maximum_rendered_input_tokens": 13_824,
        "rendered_prompt_truncation": "forbidden",
    }
    assert GRAPHRAG_CONTEXT_TOKENS < MAX_RENDERED_INPUT_TOKENS
    assert COMPLETION_TOKENS + SAFETY_MARGIN_TOKENS == 2_560
    assert MAX_COMPLETION_TOKENS == ARGUMENT_GENERATOR_COMPLETION_TOKENS == 3_072
    assert ARGUMENT_CRITIC_COMPLETION_TOKENS == 2_048
    assert ADJUDICATOR_COMPLETION_TOKENS == 1_024


@pytest.mark.parametrize("completion", [1_536, 2_048, 3_072])
def test_stage_specific_completion_budget_accepts_exact_boundary(completion: int) -> None:
    maximum_input = MAX_MODEL_LEN - completion - SAFETY_MARGIN_TOKENS
    audit = audit_prompt_budget(
        FixedLengthTokenizer(maximum_input),
        [{"role": "user", "content": "hash-only-test"}],
        requested_completion_tokens=completion,
    )
    assert audit.total_reserved_tokens == MAX_MODEL_LEN


@pytest.mark.parametrize("completion", [1_536, 2_048, 3_072])
def test_stage_specific_completion_budget_rejects_one_token_over(completion: int) -> None:
    maximum_input = MAX_MODEL_LEN - completion - SAFETY_MARGIN_TOKENS
    with pytest.raises(PromptBudgetExceeded, match="exceeds"):
        audit_prompt_budget(
            FixedLengthTokenizer(maximum_input + 1),
            [{"role": "user", "content": "hash-only-test"}],
            requested_completion_tokens=completion,
        )


def test_runtime_uses_sequential_chunked_prefill_without_gpu_utilization_change() -> None:
    source = Path("hf_job/run.py").read_text()
    from graphrag_runtime.vllm_config import completion_server_command

    command = completion_server_command(
        "vllm",
        model_id="test-model",
        revision="test-revision",
        port=8003,
        max_model_len=MAX_MODEL_LEN,
    )
    assert command[command.index("--max-model-len") + 1] == str(MAX_MODEL_LEN)
    assert command[command.index("--max-num-seqs") + 1] == "1"
    assert "--enable-chunked-prefill" in command
    assert command[command.index("--gpu-memory-utilization") + 1] == "0.75"
    assert "completion_server_command(" in source
    assert "right_truncate" not in source
    assert "truncation=True" not in source
