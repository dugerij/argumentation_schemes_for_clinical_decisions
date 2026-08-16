from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


MEDGEMMA_MODEL_ID = "google/medgemma-1.5-4b-it"
MEDGEMMA_MODEL_REVISION = "91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b"
MAX_MODEL_LEN = 16_384
GRAPHRAG_CONTEXT_TOKENS = 10_000
COMPLETION_TOKENS = 1_536
MAX_COMPLETION_TOKENS = 3_072
STANDARD_COMPLETION_TOKENS = 1_536
ARGUMENT_GENERATOR_COMPLETION_TOKENS = 3_072
ARGUMENT_CRITIC_COMPLETION_TOKENS = 2_048
ADJUDICATOR_COMPLETION_TOKENS = 1_024
SEMANTIC_JUDGE_COMPLETION_TOKENS = 512
SAFETY_MARGIN_TOKENS = 1_024
MAX_RENDERED_INPUT_TOKENS = MAX_MODEL_LEN - COMPLETION_TOKENS - SAFETY_MARGIN_TOKENS
PROMPT_BUDGET_POLICY = "exact-medgemma-chat-template-v1"


class PromptBudgetExceeded(ValueError):
    """Raised before inference when a fully rendered prompt exceeds the frozen budget."""


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def load_medgemma_tokenizer(
    model_source: str | Path = MEDGEMMA_MODEL_ID,
    *,
    revision: str = MEDGEMMA_MODEL_REVISION,
    local_files_only: bool = False,
):
    """Load the pinned authentic tokenizer used by the completion server."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_source),
        revision=revision if str(model_source) == MEDGEMMA_MODEL_ID else None,
        local_files_only=local_files_only,
        trust_remote_code=False,
    )
    if tokenizer.name_or_path == "":
        raise RuntimeError("MedGemma tokenizer source was not resolved.")
    return tokenizer


def rendered_token_ids(tokenizer: Any, messages: Iterable[Mapping[str, Any]]) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        [dict(message) for message in messages],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )
    token_ids = rendered["input_ids"]
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise RuntimeError("Expected exactly one rendered chat conversation.")
        token_ids = token_ids[0]
    values = [int(value) for value in token_ids]
    if not values:
        raise RuntimeError("MedGemma chat template rendered an empty prompt.")
    return values


@dataclass(frozen=True)
class PromptBudgetAudit:
    policy: str
    model_id: str
    model_revision: str
    rendered_input_tokens: int
    requested_completion_tokens: int
    safety_margin_tokens: int
    max_model_len: int
    maximum_rendered_input_tokens: int
    total_reserved_tokens: int
    remaining_tokens_after_reservations: int
    message_count: int
    message_roles: tuple[str, ...]
    messages_sha256: str
    rendered_token_ids_sha256: str


def audit_prompt_budget(
    tokenizer: Any,
    messages: Iterable[Mapping[str, Any]],
    *,
    requested_completion_tokens: int = COMPLETION_TOKENS,
    max_model_len: int = MAX_MODEL_LEN,
    safety_margin_tokens: int = SAFETY_MARGIN_TOKENS,
    model_id: str = MEDGEMMA_MODEL_ID,
    model_revision: str = MEDGEMMA_MODEL_REVISION,
    policy: str = PROMPT_BUDGET_POLICY,
) -> PromptBudgetAudit:
    materialized = [dict(message) for message in messages]
    if requested_completion_tokens <= 0:
        raise ValueError("A positive fixed completion allowance is required.")
    if safety_margin_tokens <= 0:
        raise ValueError("A positive prompt safety margin is required.")
    ids = rendered_token_ids(tokenizer, materialized)
    rendered_input_tokens = len(ids)
    maximum_input = max_model_len - requested_completion_tokens - safety_margin_tokens
    total = rendered_input_tokens + requested_completion_tokens + safety_margin_tokens
    audit = PromptBudgetAudit(
        policy=policy,
        model_id=model_id,
        model_revision=model_revision,
        rendered_input_tokens=rendered_input_tokens,
        requested_completion_tokens=requested_completion_tokens,
        safety_margin_tokens=safety_margin_tokens,
        max_model_len=max_model_len,
        maximum_rendered_input_tokens=maximum_input,
        total_reserved_tokens=total,
        remaining_tokens_after_reservations=max_model_len - total,
        message_count=len(materialized),
        message_roles=tuple(str(item.get("role")) for item in materialized),
        messages_sha256=canonical_sha256(materialized),
        rendered_token_ids_sha256=hashlib.sha256(
            ",".join(str(value) for value in ids).encode("ascii")
        ).hexdigest(),
    )
    if total > max_model_len:
        raise PromptBudgetExceeded(
            "Rendered model request exceeds the frozen context boundary: "
            f"{rendered_input_tokens}+{requested_completion_tokens}+"
            f"{safety_margin_tokens}>{max_model_len}."
        )
    return audit


def aggregate_prompt_audit(audit: PromptBudgetAudit) -> dict[str, Any]:
    """Return the aggregate/hash-only representation safe for persistent evidence."""
    value = asdict(audit)
    value["message_roles"] = list(audit.message_roles)
    return value


def component_token_counts(
    tokenizer: Any,
    *,
    system_message: str,
    user_message: str,
) -> dict[str, int]:
    """Count prompt components without persisting their licensed/raw contents."""
    markers = {
        "reports": "-----Reports-----",
        "entities": "-----Entities-----",
        "relationships": "-----Relationships-----",
        "covariates": "-----Covariates-----",
        "sources": "-----Sources-----",
    }
    counts: dict[str, int] = {
        "system_message": len(tokenizer.encode(system_message, add_special_tokens=False)),
        "user_message": len(tokenizer.encode(user_message, add_special_tokens=False)),
    }
    positions = sorted(
        (system_message.find(marker), name)
        for name, marker in markers.items()
        if system_message.find(marker) >= 0
    )
    present = {name for _, name in positions}
    for name in markers:
        if name not in present:
            counts[name] = 0
    for index, (start, name) in enumerate(positions):
        end = positions[index + 1][0] if index + 1 < len(positions) else len(system_message)
        counts[name] = len(
            tokenizer.encode(system_message[start:end], add_special_tokens=False)
        )
    return counts


def frozen_budget_config() -> dict[str, Any]:
    return {
        "policy": PROMPT_BUDGET_POLICY,
        "model_id": MEDGEMMA_MODEL_ID,
        "model_revision": MEDGEMMA_MODEL_REVISION,
        "max_model_len": MAX_MODEL_LEN,
        "graphrag_context_tokens": GRAPHRAG_CONTEXT_TOKENS,
        "completion_tokens": COMPLETION_TOKENS,
        "maximum_request_completion_tokens": MAX_COMPLETION_TOKENS,
        "stage_completion_tokens": {
            "standard": STANDARD_COMPLETION_TOKENS,
            "argument_generator": ARGUMENT_GENERATOR_COMPLETION_TOKENS,
            "argument_critic": ARGUMENT_CRITIC_COMPLETION_TOKENS,
            "adjudicator": ADJUDICATOR_COMPLETION_TOKENS,
            "semantic_judge": SEMANTIC_JUDGE_COMPLETION_TOKENS,
        },
        "safety_margin_tokens": SAFETY_MARGIN_TOKENS,
        "maximum_rendered_input_tokens": MAX_RENDERED_INPUT_TOKENS,
        "rendered_prompt_truncation": "forbidden",
    }
