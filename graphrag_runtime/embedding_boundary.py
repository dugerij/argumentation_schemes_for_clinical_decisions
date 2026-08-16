from __future__ import annotations

import argparse
import hashlib
import json
import threading
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from tokenizers import Tokenizer

from graphrag_runtime.lossless_bge import (
    LOSSLESS_POLICY,
    plan_audit_record,
    plan_lossless_embedding,
    token_weighted_mean_l2,
)


BGE_MAX_INPUT_TOKENS = 512
TRUNCATION_POLICY = "exact_bge_tokenizer_audited_right_truncation_v1"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_token_ids(ids: list[int]) -> str:
    encoded = ",".join(str(value) for value in ids).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class EmbeddingBoundaryRecord:
    request_id: int
    input_index: int
    input_kind: str
    source_sha256: str
    forwarded_sha256: str
    source_bge_tokens: int
    forwarded_bge_tokens: int
    dropped_bge_tokens: int
    truncated: bool
    policy: str = TRUNCATION_POLICY


def load_exact_bge_tokenizer(
    model_id: str,
    *,
    tokenizer_json: Path | None = None,
) -> Tokenizer:
    """Load the tokenizer used by the BGE embedding server."""
    if tokenizer_json is not None:
        tokenizer = Tokenizer.from_file(str(tokenizer_json))
    else:
        tokenizer = Tokenizer.from_pretrained(model_id)
    if len(tokenizer.encode("", add_special_tokens=True).ids) < 2:
        raise RuntimeError("BGE tokenizer is missing expected boundary tokens.")
    return tokenizer


def constrain_embedding_text(
    text: str,
    tokenizer: Tokenizer,
    *,
    max_tokens: int = BGE_MAX_INPUT_TOKENS,
) -> tuple[str, dict[str, int | bool | str]]:
    """Right-truncate with the exact tokenizer and retain an auditable mapping."""
    source_ids = tokenizer.encode(text, add_special_tokens=True).ids
    if len(source_ids) <= max_tokens:
        return text, {
            "source_sha256": _sha256_text(text),
            "forwarded_sha256": _sha256_text(text),
            "source_token_ids_sha256": _sha256_token_ids(source_ids),
            "forwarded_token_ids_sha256": _sha256_token_ids(source_ids),
            "source_bge_tokens": len(source_ids),
            "forwarded_bge_tokens": len(source_ids),
            "dropped_bge_tokens": 0,
            "truncated": False,
            "policy": TRUNCATION_POLICY,
        }

    raw_ids = tokenizer.encode(text, add_special_tokens=False).ids
    special_overhead = len(tokenizer.encode("", add_special_tokens=True).ids)
    retained_count = min(len(raw_ids), max_tokens - special_overhead)
    forwarded = ""
    forwarded_ids: list[int] = []
    while retained_count >= 0:
        # Retain literal special-token strings in clinical/report text. The
        # tokenizer adds its own model boundary tokens on the next encode.
        forwarded = tokenizer.decode(
            raw_ids[:retained_count],
            skip_special_tokens=False,
        )
        forwarded_ids = tokenizer.encode(
            forwarded,
            add_special_tokens=True,
        ).ids
        if len(forwarded_ids) <= max_tokens:
            break
        retained_count -= 1
    if retained_count < 0 or len(forwarded_ids) > max_tokens:
        raise RuntimeError("Could not enforce the BGE embedding boundary.")
    return forwarded, {
        "source_sha256": _sha256_text(text),
        "forwarded_sha256": _sha256_text(forwarded),
        "source_token_ids_sha256": _sha256_token_ids(source_ids),
        "forwarded_token_ids_sha256": _sha256_token_ids(forwarded_ids),
        "source_bge_tokens": len(source_ids),
        "forwarded_bge_tokens": len(forwarded_ids),
        "dropped_bge_tokens": len(source_ids) - len(forwarded_ids),
        "truncated": True,
        "policy": TRUNCATION_POLICY,
    }


def constrain_embedding_token_ids(
    ids: list[int],
    *,
    max_tokens: int = BGE_MAX_INPUT_TOKENS,
) -> tuple[list[int], dict[str, int | bool | str]]:
    forwarded = ids[:max_tokens]
    return forwarded, {
        "source_sha256": _sha256_token_ids(ids),
        "forwarded_sha256": _sha256_token_ids(forwarded),
        "source_bge_tokens": len(ids),
        "forwarded_bge_tokens": len(forwarded),
        "dropped_bge_tokens": len(ids) - len(forwarded),
        "truncated": len(ids) > max_tokens,
        "policy": TRUNCATION_POLICY,
    }


def constrain_embedding_payload(
    payload: dict[str, Any],
    tokenizer: Tokenizer,
    *,
    request_id: int,
    max_tokens: int = BGE_MAX_INPUT_TOKENS,
) -> tuple[dict[str, Any], list[EmbeddingBoundaryRecord]]:
    """Rewrite only the OpenAI embedding input field, preserving order."""
    source_input = payload.get("input")
    single = isinstance(source_input, str) or (
        isinstance(source_input, list)
        and all(isinstance(value, int) for value in source_input)
    )
    inputs = [source_input] if single else source_input
    if not isinstance(inputs, list) or not inputs:
        raise TypeError("Embedding payload input must be non-empty.")

    forwarded_inputs: list[str | list[int]] = []
    records: list[EmbeddingBoundaryRecord] = []
    for index, value in enumerate(inputs):
        if isinstance(value, str):
            forwarded, audit = constrain_embedding_text(
                value,
                tokenizer,
                max_tokens=max_tokens,
            )
            kind = "text"
        elif isinstance(value, list) and all(
            isinstance(token, int) for token in value
        ):
            forwarded, audit = constrain_embedding_token_ids(
                value,
                max_tokens=max_tokens,
            )
            kind = "token_ids"
        else:
            raise TypeError("Embedding inputs must be text or integer token IDs.")
        if int(audit["forwarded_bge_tokens"]) > max_tokens:
            raise RuntimeError("Outbound BGE input exceeds the hard boundary.")
        forwarded_inputs.append(forwarded)
        records.append(
            EmbeddingBoundaryRecord(
                request_id=request_id,
                input_index=index,
                input_kind=kind,
                source_sha256=str(audit["source_sha256"]),
                forwarded_sha256=str(audit["forwarded_sha256"]),
                source_bge_tokens=int(audit["source_bge_tokens"]),
                forwarded_bge_tokens=int(audit["forwarded_bge_tokens"]),
                dropped_bge_tokens=int(audit["dropped_bge_tokens"]),
                truncated=bool(audit["truncated"]),
            )
        )
    rewritten = dict(payload)
    rewritten["input"] = forwarded_inputs[0] if single else forwarded_inputs
    return rewritten, records


class BoundaryAuditLog:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, record: EmbeddingBoundaryRecord | dict[str, Any]) -> None:
        value = record if isinstance(record, dict) else asdict(record)
        line = json.dumps(value, sort_keys=True) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as target:
            target.write(line)


def audit_embedding_boundary(path: Path) -> dict[str, int | bool | str]:
    records = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise RuntimeError("Embedding boundary audit is empty.")
    lossless = all(record.get("policy") == LOSSLESS_POLICY for record in records)
    forwarded_counts = (
        [
            int(chunk["model_token_count"])
            for record in records
            for chunk in record["chunks"]
        ]
        if lossless
        else [int(record["forwarded_bge_tokens"]) for record in records]
    )
    over_limit = [value for value in forwarded_counts if value > BGE_MAX_INPUT_TOKENS]
    if over_limit:
        raise RuntimeError("Embedding boundary emitted an over-limit input.")
    return {
        "all_outbound_inputs_within_bge_limit": True,
        "input_count": len(records),
        "maximum_source_bge_tokens": max(
            int(record.get("source_model_token_count", record.get("source_bge_tokens", 0)))
            for record in records
        ),
        "maximum_forwarded_bge_tokens": max(forwarded_counts),
        "truncated_input_count": 0 if lossless else sum(
            bool(record["truncated"]) for record in records
        ),
        "split_input_count": sum(int(record.get("chunk_count", 1)) > 1 for record in records),
        "raw_tokens_lost": 0 if lossless else sum(
            int(record.get("dropped_bge_tokens", 0)) for record in records
        ),
        "policy": LOSSLESS_POLICY if lossless else TRUNCATION_POLICY,
        "audit_log_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    }


def expand_lossless_embedding_payload(
    payload: dict[str, Any],
    tokenizer: Tokenizer,
    *,
    request_id: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = payload.get("input")
    single = isinstance(source, str)
    inputs = [source] if single else source
    if not isinstance(inputs, list) or not inputs or not all(
        isinstance(value, str) for value in inputs
    ):
        raise TypeError("Lossless BGE boundary accepts non-empty text inputs only.")
    expanded: list[list[int]] = []
    mappings = []
    for input_index, text in enumerate(inputs):
        plan = plan_lossless_embedding(text, tokenizer)
        start = len(expanded)
        for chunk in plan.chunks:
            chunk_token_ids = list(chunk.token_ids)
            if len(chunk_token_ids) > 512:
                raise RuntimeError("Lossless BGE boundary produced an over-limit request.")
            expanded.append(chunk_token_ids)
        mappings.append({
            "request_id": request_id,
            "input_index": input_index,
            "expanded_start": start,
            "expanded_end": len(expanded),
            "raw_token_weights": [chunk.raw_token_count for chunk in plan.chunks],
            **plan_audit_record(plan),
        })
    rewritten = dict(payload)
    rewritten["input"] = expanded
    return rewritten, mappings


def aggregate_lossless_embedding_response(
    response_payload: dict[str, Any],
    mappings: list[dict[str, Any]],
) -> dict[str, Any]:
    data = sorted(response_payload.get("data") or [], key=lambda item: int(item["index"]))
    expected = max(int(item["expanded_end"]) for item in mappings)
    if len(data) != expected:
        raise RuntimeError("Upstream BGE response cardinality mismatch.")
    output = []
    for original_index, mapping in enumerate(mappings):
        start = int(mapping["expanded_start"])
        end = int(mapping["expanded_end"])
        vectors = [data[index]["embedding"] for index in range(start, end)]
        if len(vectors) == 1:
            vector = vectors[0]
            array = np.asarray(vector, dtype=np.float64)
            if array.shape != (384,) or not np.isfinite(array).all():
                raise RuntimeError("Unchanged BGE vector failed validation.")
        else:
            vector = token_weighted_mean_l2(vectors, mapping["raw_token_weights"])
        output.append({"object": "embedding", "index": original_index, "embedding": vector})
    rewritten = dict(response_payload)
    rewritten["data"] = output
    return rewritten


def make_embedding_boundary_handler(
    *,
    upstream_url: str,
    tokenizer: Tokenizer,
    audit_log: BoundaryAuditLog,
) -> type[BaseHTTPRequestHandler]:
    request_counter = iter(range(1, 2**63))

    def upstream_request_url(path: str) -> str:
        base = upstream_url.rstrip("/")
        if base.endswith("/v1") and path.startswith("/v1/"):
            return f"{base}{path[3:]}"
        return f"{base}{path}"

    class EmbeddingBoundaryHandler(BaseHTTPRequestHandler):
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
                with urlopen(request, timeout=600) as response:
                    response_body = response.read()
                    self.send_response(response.status)
                    self.send_header(
                        "Content-Type",
                        response.headers.get("Content-Type", "application/json"),
                    )
                    self.end_headers()
                    self.wfile.write(response_body)
            except HTTPError as exc:
                response_body = exc.read()
                self.send_response(exc.code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(response_body)
            except OSError as exc:
                # Connection resets/timeouts to the upstream server
                # (URLError/TimeoutError are both OSError subclasses) must
                # still produce a diagnosable HTTP response instead of an
                # opaque dropped connection.
                self.send_error(502, f"Upstream request failed: {exc}")

        def do_GET(self) -> None:  # noqa: N802
            self._proxy()

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") != "/v1/embeddings":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            request_id = next(request_counter)
            rewritten, records = expand_lossless_embedding_payload(
                payload,
                tokenizer,
                request_id=request_id,
            )
            headers = {"Content-Type": "application/json"}
            authorization = self.headers.get("Authorization")
            if authorization:
                headers["Authorization"] = authorization
            request = Request(
                upstream_request_url(self.path),
                data=json.dumps(rewritten).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urlopen(request, timeout=600) as response:
                    upstream = json.loads(response.read())
            except HTTPError as exc:
                response_body = exc.read()
                self.send_response(exc.code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(response_body)
                return
            except OSError as exc:
                self.send_error(502, f"Upstream request failed: {exc}")
                return
            aggregated = aggregate_lossless_embedding_response(upstream, records)
            for record in records:
                audit_log.append(record)
            response_body = json.dumps(aggregated).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

    return EmbeddingBoundaryHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-url", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--audit-log", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--tokenizer-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tokenizer = load_exact_bge_tokenizer(
        args.model_id,
        tokenizer_json=args.tokenizer_json,
    )
    audit_log = BoundaryAuditLog(args.audit_log)
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port),
        make_embedding_boundary_handler(
            upstream_url=args.upstream_url.rstrip("/"),
            tokenizer=tokenizer,
            audit_log=audit_log,
        ),
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
