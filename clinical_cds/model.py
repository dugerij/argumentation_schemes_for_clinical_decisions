from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlparse

from clinical_cds.ollama import ollama_chat, ollama_llm_endpoint


OUTPUT_SCHEMA = {
    "type": "object",
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
    ) -> str:
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
                "seed": self.seed,
                "num_ctx": self.context_window,
                "num_predict": self.max_output_tokens,
            },
            think=False,
        )


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
