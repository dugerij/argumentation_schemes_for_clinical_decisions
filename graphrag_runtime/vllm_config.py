"""Pinned vLLM command and JSON-schema compatibility configuration.

This module is deliberately GPU-free. The same functions construct and verify
the runtime command before upload, preventing command-line drift from becoming
an A100-only discovery.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Iterable


PINNED_VLLM_VERSION = "0.18.1"
STRUCTURED_OUTPUT_BACKEND = "xgrammar"
STRUCTURED_OUTPUT_CONFIG: dict[str, object] = {
    "backend": STRUCTURED_OUTPUT_BACKEND,
    "disable_any_whitespace": True,
}
STRUCTURED_OUTPUT_CONFIG_JSON = json.dumps(
    STRUCTURED_OUTPUT_CONFIG,
    sort_keys=True,
    separators=(",", ":"),
)

# These constraints remain in the strict post-generation schema/validator but
# are removed from the schema sent to pinned xgrammar because vLLM 0.18.1 does
# not implement them at decoding time.
UNSUPPORTED_XGRAMMAR_SCHEMA_KEYWORDS = frozenset({"uniqueItems"})


def validate_structured_output_config(value: str) -> dict[str, object]:
    """Validate the exact backend configuration accepted by the pinned image."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("vLLM structured-output configuration is not JSON.") from exc
    if parsed != STRUCTURED_OUTPUT_CONFIG:
        raise ValueError(
            "vLLM structured-output configuration drifted from the pinned "
            f"{PINNED_VLLM_VERSION} xgrammar contract."
        )
    return parsed


def decoder_compatible_schema(
    schema: dict[str, object],
    *,
    require_closed_objects: bool = False,
) -> dict[str, object]:
    """Return a copy restricted to the pinned xgrammar decoder subset.

    Generic OpenAI-compatible callers may intentionally supply a minimal JSON
    object schema. The experiment runtime sets ``require_closed_objects=True``;
    that stricter research boundary must not silently change unrelated clients.
    """
    compatible: dict[str, object] = deepcopy(schema)

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for keyword in UNSUPPORTED_XGRAMMAR_SCHEMA_KEYWORDS:
                value.pop(keyword, None)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(compatible)
    from jsonschema import Draft202012Validator

    Draft202012Validator.check_schema(compatible)
    if require_closed_objects:
        validate_decoder_schema(compatible)
    return compatible


def validate_decoder_schema(schema: dict[str, object]) -> None:
    """Reject known remote-decoder failures and open object boundaries."""
    from jsonschema import Draft202012Validator

    Draft202012Validator.check_schema(schema)
    failures: list[str] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for keyword in UNSUPPORTED_XGRAMMAR_SCHEMA_KEYWORDS:
                if keyword in value:
                    failures.append(f"{path}.{keyword}: unsupported by xgrammar")
            if value.get("type") == "object":
                if value.get("additionalProperties") is not False:
                    failures.append(f"{path}: object boundary is not closed")
            for key, child in value.items():
                visit(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(schema, "schema")
    if failures:
        raise ValueError("; ".join(failures))


def completion_server_command(
    executable: str,
    *,
    model_id: str,
    revision: str,
    port: int,
    max_model_len: int,
    gpu_memory_utilization: float = 0.75,
    dtype: str = "bfloat16",
) -> list[str]:
    """Build and self-validate the only supported completion-server command."""
    command = [
        executable,
        "serve",
        model_id,
        "--revision",
        revision,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        dtype,
        "--generation-config",
        "vllm",
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-num-seqs",
        "1",
        "--enable-chunked-prefill",
        "--structured-outputs-config",
        STRUCTURED_OUTPUT_CONFIG_JSON,
    ]
    validate_completion_server_command(
        command,
        model_id=model_id,
        revision=revision,
        port=port,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
    )
    return command


def _single_option(command: list[str], option: str) -> str:
    positions = [index for index, value in enumerate(command) if value == option]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ValueError(f"vLLM command must contain exactly one {option} value.")
    return command[positions[0] + 1]


def validate_completion_server_command(
    command: Iterable[str],
    *,
    model_id: str,
    revision: str,
    port: int,
    max_model_len: int,
    gpu_memory_utilization: float = 0.75,
    dtype: str = "bfloat16",
) -> dict[str, object]:
    """Fail closed on the configuration failures seen in historical runs."""
    values = list(command)
    if len(values) < 3 or values[1:3] != ["serve", model_id]:
        raise ValueError("vLLM completion command model binding changed.")
    expected = {
        "--revision": revision,
        "--host": "127.0.0.1",
        "--port": str(port),
        "--dtype": dtype,
        "--generation-config": "vllm",
        "--max-model-len": str(max_model_len),
        "--gpu-memory-utilization": str(gpu_memory_utilization),
        "--max-num-seqs": "1",
    }
    for option, required in expected.items():
        if _single_option(values, option) != required:
            raise ValueError(f"vLLM completion command {option} changed.")
    if values.count("--enable-chunked-prefill") != 1:
        raise ValueError("vLLM chunked-prefill policy changed.")
    config = validate_structured_output_config(
        _single_option(values, "--structured-outputs-config")
    )
    return {
        "vllm_version": PINNED_VLLM_VERSION,
        "model_id": model_id,
        "revision": revision,
        "port": port,
        "max_model_len": max_model_len,
        "max_num_seqs": 1,
        "chunked_prefill": True,
        "gpu_memory_utilization": gpu_memory_utilization,
        "dtype": dtype,
        "structured_output_config": config,
    }
