from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

import requests

from clinical_cds.ollama import ollama_chat, ollama_llm_endpoint


OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer": {"type": "string", "maxLength": 200},
        "reasoning": {"type": "string", "maxLength": 800},
        "citations": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 4,
        },
        "observations": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "text": {"type": "string", "maxLength": 300},
                    "source_id": {"type": "string", "maxLength": 50},
                },
                "required": ["text", "source_id"],
            },
        },
        "abstain": {"type": "boolean"},
    },
    "required": [
        "answer",
        "reasoning",
        "citations",
        "observations",
        "abstain",
    ],
}


def _vllm_compatible_output_schema(
    output_schema: dict[str, object],
) -> dict[str, object]:
    """Return the frozen schema in the subset accepted by pinned vLLM.

    vLLM 0.18.1 rejects ``uniqueItems`` before generation.  The experiment's
    strict post-generation validators independently enforce identifier and
    binding uniqueness, so removing this decoding-only keyword changes no
    accepted result.  All other constraints, including closed objects and
    case-specific enums, are preserved byte-for-byte in the copied tree.
    """

    compatible: dict[str, object] = deepcopy(output_schema)

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            value.pop("uniqueItems", None)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(compatible)
    return compatible


class DiagnosticModel(Protocol):
    @property
    def model_id(self) -> str:
        ...

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        output_schema: dict[str, object] | None = None,
        context_window: int | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        ...


def is_loopback_endpoint(endpoint: str) -> bool:
    hostname = (urlparse(endpoint).hostname or "").casefold()
    return hostname in {"127.0.0.1", "localhost", "::1"}


def model_name_appears_remote(model_name: str) -> bool:
    normalized = model_name.casefold()
    return (
        "cloud" in normalized
        or normalized.startswith("http://")
        or normalized.startswith("https://")
    )


@dataclass(frozen=True)
class OllamaDiagnosticModel:
    model_name: str
    seed: int = 17
    timeout_seconds: float = 600.0
    context_window: int = 8192
    max_output_tokens: int = 1024
    allow_remote: bool = False

    @property
    def model_id(self) -> str:
        return f"ollama:{self.model_name}"

    @property
    def cache_identity(self) -> str:
        return (
            f"{self.model_id}|seed={self.seed}|context={self.context_window}"
            f"|max_output={self.max_output_tokens}|think=false"
        )

    @property
    def decoding_config(self) -> dict[str, int | float | bool]:
        return {
            "temperature": 0,
            "top_p": 1,
            "top_k": -1,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "seed": self.seed,
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "think": False,
        }

    def assert_data_boundary(self, dataset: str) -> None:
        if dataset not in {"direct", "submitted_patient"} or self.allow_remote:
            return
        endpoint = ollama_llm_endpoint()
        if not is_loopback_endpoint(endpoint) or model_name_appears_remote(self.model_name):
            raise ValueError(
                f"{dataset} data may only be sent to a local model. "
                "Use a loopback Ollama endpoint and a non-cloud model, or pass the explicit "
                "remote-data override only when your data-use agreement permits it."
            )

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        output_schema: dict[str, object] | None = None,
        context_window: int | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        effective_context_window = context_window or self.context_window
        effective_max_output_tokens = (
            max_output_tokens or self.max_output_tokens
        )
        return ollama_chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            timeout=self.timeout_seconds,
            format=output_schema or OUTPUT_SCHEMA,
            options={
                "temperature": 0,
                "top_p": 1,
                "top_k": -1,
                "presence_penalty": 0,
                "frequency_penalty": 0,
                "seed": self.seed,
                "num_ctx": effective_context_window,
                "num_predict": effective_max_output_tokens,
            },
            think=False,
        )


@dataclass(frozen=True)
class OpenAICompatibleDiagnosticModel:
    """Diagnostic model served by a locally controlled vLLM endpoint."""

    model_name: str
    model_revision: str = ""
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = ""
    seed: int = 17
    timeout_seconds: float = 600.0
    context_window: int = 8192
    max_output_tokens: int = 1024
    allow_remote: bool = False

    @property
    def model_id(self) -> str:
        return f"openai-compatible:{self.model_name}"

    @property
    def cache_identity(self) -> str:
        structured_outputs_config = os.environ.get(
            "VLLM_STRUCTURED_OUTPUTS_CONFIG", ""
        )
        return (
            f"{self.model_id}|revision={self.model_revision}|base_url={self.base_url}|seed={self.seed}"
            f"|context={self.context_window}|max_output={self.max_output_tokens}"
            "|temperature=0|top_p=1|top_k=-1|presence_penalty=0"
            "|frequency_penalty=0|n=1|stream=false"
            f"|structured_outputs={structured_outputs_config}"
        )

    @property
    def decoding_config(self) -> dict[str, int | float | bool | str]:
        return {
            "temperature": 0,
            "top_p": 1,
            "top_k": -1,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "n": 1,
            "stream": False,
            "seed": self.seed,
            "model_revision": self.model_revision,
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "base_url": self.base_url,
            "structured_outputs_config": os.environ.get(
                "VLLM_STRUCTURED_OUTPUTS_CONFIG", ""
            ),
        }

    def assert_data_boundary(self, dataset: str) -> None:
        if dataset not in {"direct", "submitted_patient"} or self.allow_remote:
            return
        if not is_loopback_endpoint(self.base_url):
            raise ValueError(
                f"{dataset} data may only be sent to a locally controlled model. "
                "Use a loopback vLLM endpoint, or pass the explicit remote-data "
                "override only when the data-use agreement permits it."
            )

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        output_schema: dict[str, object] | None = None,
        context_window: int | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        del context_window  # The vLLM server owns its configured context limit.
        payload: dict[str, object] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "top_p": 1,
            "top_k": -1,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "n": 1,
            "stream": False,
            "seed": self.seed,
            "max_tokens": max_output_tokens or self.max_output_tokens,
        }
        if output_schema:
            payload["structured_outputs"] = {
                "json": _vllm_compatible_output_schema(output_schema)
            }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        return str(body["choices"][0]["message"]["content"])


def ollama_model_from_env(
    *,
    model_name: str | None = None,
    seed: int = 17,
    allow_remote: bool = False,
    timeout_seconds: float = 600.0,
    context_window: int = 8192,
    max_output_tokens: int = 1024,
) -> OllamaDiagnosticModel:
    configured_name = (
        model_name
        or os.environ.get("DIAGNOSTIC_MODEL")
    )
    if not configured_name:
        raise ValueError(
            "Set DIAGNOSTIC_MODEL or pass --model before running an experiment."
        )
    return OllamaDiagnosticModel(
        model_name=configured_name,
        seed=seed,
        timeout_seconds=timeout_seconds,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        allow_remote=allow_remote,
    )


def openai_compatible_model_from_env(
    *,
    model_name: str | None = None,
    model_revision: str = "",
    base_url: str | None = None,
    seed: int = 17,
    allow_remote: bool = False,
    timeout_seconds: float = 600.0,
    context_window: int = 8192,
    max_output_tokens: int = 1024,
) -> OpenAICompatibleDiagnosticModel:
    configured_name = model_name or os.environ.get("DIAGNOSTIC_MODEL")
    if not configured_name:
        raise ValueError(
            "Set DIAGNOSTIC_MODEL or pass --model before running an experiment."
        )
    return OpenAICompatibleDiagnosticModel(
        model_name=configured_name,
        model_revision=model_revision,
        base_url=(
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or "http://127.0.0.1:8000/v1"
        ),
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        seed=seed,
        timeout_seconds=timeout_seconds,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        allow_remote=allow_remote,
    )
