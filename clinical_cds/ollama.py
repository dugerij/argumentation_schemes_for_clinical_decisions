import json
import os
import urllib.request
from typing import Any


DEFAULT_OLLAMA_ENDPOINT = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_TIMEOUT = 180.0


def ollama_llm_endpoint() -> str:
    return os.environ.get(
        "OLLAMA_LLM_ENDPOINT",
        os.environ.get("OLLAMA_ENDPOINT", DEFAULT_OLLAMA_ENDPOINT),
    ).rstrip("/")


def ollama_headers() -> dict[str, str]:
    api_key = os.environ.get("OLLAMA_API_KEY", "").strip()
    if not api_key:
        return {}
    header_name = (
        os.environ.get("OLLAMA_AUTH_HEADER", "Authorization").strip()
        or "Authorization"
    )
    auth_scheme = os.environ.get("OLLAMA_AUTH_SCHEME", "Bearer").strip()
    header_value = f"{auth_scheme} {api_key}" if auth_scheme else api_key
    return {header_name: header_value}


def ollama_post(
    path: str,
    payload: dict[str, Any],
    *,
    timeout: float = DEFAULT_OLLAMA_TIMEOUT,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    headers.update(ollama_headers())
    request = urllib.request.Request(
        f"{ollama_llm_endpoint()}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ollama_chat(
    *,
    model: str,
    messages: list[dict[str, str]],
    timeout: float = DEFAULT_OLLAMA_TIMEOUT,
    format: str | dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
    think: bool | None = None,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if format is not None:
        payload["format"] = format
    if options:
        payload["options"] = options
    if think is not None:
        payload["think"] = think
    response = ollama_post("/api/chat", payload, timeout=timeout)
    return response.get("message", {}).get("content", "")
