from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from graphrag_runtime.medgemma_prompt_budget import (
    MAX_COMPLETION_TOKENS,
    MEDGEMMA_MODEL_ID,
    MEDGEMMA_MODEL_REVISION,
    aggregate_prompt_audit,
    audit_prompt_budget,
    load_medgemma_tokenizer,
)
from graphrag_runtime.provenance_contract import canonical_sha256


def requested_completion_tokens(payload: dict[str, Any]) -> int:
    """Validate a request-specific completion cap against the frozen maximum."""
    value = payload.get("max_tokens")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("A positive integer completion allowance is required.")
    if value <= 0:
        raise ValueError("A positive integer completion allowance is required.")
    if value > MAX_COMPLETION_TOKENS:
        raise ValueError("Completion allowance exceeds the frozen budget.")
    return value


def audit_client_structured_output(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Prove that non-GraphRAG agent schemas reach vLLM unchanged."""
    configured = payload.get("structured_outputs")
    if configured is None:
        return {
            "client_structured_output_forwarded": False,
            "client_structured_output_schema_sha256": None,
            "client_unconstrained_contract_id": None,
        }
    if (
        not isinstance(configured, dict)
        or set(configured) != {"json"}
        or not isinstance(configured.get("json"), dict)
    ):
        raise ValueError("Client structured output must contain exactly one JSON schema.")
    schema = configured["json"]
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        raise ValueError(
            "Client JSON schema must be a closed top-level object contract."
        )
    return {
        "client_structured_output_forwarded": True,
        "client_structured_output_schema_sha256": canonical_sha256(schema),
        "client_unconstrained_contract_id": None,
    }


class HashOnlyPromptAuditLog:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, record: dict[str, Any]) -> None:
        with self._lock, self.path.open("a", encoding="utf-8") as target:
            target.write(json.dumps(record, sort_keys=True) + "\n")


def make_chat_boundary_handler(
    *,
    upstream_url: str,
    tokenizer: Any,
    audit_log: HashOnlyPromptAuditLog,
    expected_model_id: str = MEDGEMMA_MODEL_ID,
    model_revision: str = MEDGEMMA_MODEL_REVISION,
    max_model_len: int = 16_384,
    safety_margin_tokens: int = 1_024,
    prompt_budget_policy: str = "exact-medgemma-chat-template-v1",
) -> type[BaseHTTPRequestHandler]:
    counter = iter(range(1, 2**63))

    def upstream_request_url(path: str) -> str:
        base = upstream_url.rstrip("/")
        if base.endswith("/v1") and path.startswith("/v1/"):
            return f"{base}{path[3:]}"
        return f"{base}{path}"

    class MedGemmaBoundaryHandler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *args: object) -> None:
            return

        def _proxy(self, body: bytes | None = None) -> None:
            headers = {"Content-Type": self.headers.get("Content-Type", "application/json")}
            authorization = self.headers.get("Authorization")
            if authorization:
                headers["Authorization"] = authorization
            request = Request(
                upstream_request_url(self.path),
                data=body,
                headers=headers,
                method=self.command,
            )
            try:
                with urlopen(request, timeout=900) as response:
                    response_body = response.read()
                    self.send_response(response.status)
                    self.send_header(
                        "Content-Type",
                        response.headers.get("Content-Type", "application/json"),
                    )
                    self.send_header("Content-Length", str(len(response_body)))
                    self.end_headers()
                    self.wfile.write(response_body)
            except HTTPError as exc:
                response_body = exc.read()
                self.send_response(exc.code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)
            except OSError as exc:
                # Connection resets/timeouts to the upstream vLLM server
                # (URLError/TimeoutError are both OSError subclasses) must
                # still produce a diagnosable HTTP response instead of an
                # opaque dropped connection -- a transient upstream restart
                # should surface as a retryable 502, not a silent hang, on a
                # job with no automatic retry.
                self.send_error(502, f"Upstream request failed: {exc}")

        def do_GET(self) -> None:  # noqa: N802
            self._proxy()

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") != "/v1/chat/completions":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            payload = json.loads(body)
            if payload.get("model") != expected_model_id:
                self.send_error(400, "Completion model differs from the frozen model.")
                return
            try:
                completion_tokens = requested_completion_tokens(payload)
            except ValueError as exc:
                self.send_error(400, str(exc))
                return
            messages = payload.get("messages")
            if not isinstance(messages, list) or not messages:
                self.send_error(400, "Chat messages are required.")
                return
            try:
                client_schema_audit = audit_client_structured_output(
                    payload
                )
            except ValueError as exc:
                self.send_error(400, str(exc))
                return
            try:
                audit = audit_prompt_budget(
                    tokenizer,
                    messages,
                    requested_completion_tokens=completion_tokens,
                    max_model_len=max_model_len,
                    safety_margin_tokens=safety_margin_tokens,
                    model_id=expected_model_id,
                    model_revision=model_revision,
                    policy=prompt_budget_policy,
                )
            except (TypeError, ValueError, RuntimeError) as exc:
                self.send_error(400, str(exc))
                return
            # The GraphRAG local-search candidate-choice structured-output
            # contract (the only thing that ever set these fields to
            # non-default values) was retired as dead code -- it was never
            # reachable from the executed pipeline. These keys are kept
            # constant so the audit-log schema consumed by
            # job.run.audit_medgemma_prompt_boundary /
            # audit_all_completion_prompts is unchanged.
            structured_output_audit = {
                "structured_output_injected": False,
                "structured_output_contract_id": None,
                "response_schema_sha256": None,
                "post_generation_schema_sha256": None,
            }
            audit_log.append(
                {
                    "request_id": next(counter),
                    **aggregate_prompt_audit(audit),
                    **client_schema_audit,
                    **structured_output_audit,
                    "raw_prompt_persisted": False,
                }
            )
            self._proxy(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
            )

    return MedGemmaBoundaryHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-url", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--audit-log", type=Path, required=True)
    parser.add_argument("--model-source", default=MEDGEMMA_MODEL_ID)
    parser.add_argument("--revision", default=MEDGEMMA_MODEL_REVISION)
    parser.add_argument("--expected-model-id", default=MEDGEMMA_MODEL_ID)
    parser.add_argument("--max-model-len", type=int, default=16_384)
    parser.add_argument("--safety-margin-tokens", type=int, default=1_024)
    parser.add_argument(
        "--prompt-budget-policy",
        default="exact-medgemma-chat-template-v1",
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tokenizer = load_medgemma_tokenizer(
        args.model_source,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port),
        make_chat_boundary_handler(
            upstream_url=args.upstream_url,
            tokenizer=tokenizer,
            audit_log=HashOnlyPromptAuditLog(args.audit_log),
            expected_model_id=args.expected_model_id,
            model_revision=args.revision,
            max_model_len=args.max_model_len,
            safety_margin_tokens=args.safety_margin_tokens,
            prompt_budget_policy=args.prompt_budget_policy,
        ),
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
