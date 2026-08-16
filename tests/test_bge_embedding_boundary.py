from __future__ import annotations

import json
import os
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from graphrag_runtime.embedding_boundary import (
    BGE_MAX_INPUT_TOKENS,
    BoundaryAuditLog,
    audit_embedding_boundary,
    constrain_embedding_payload,
    constrain_embedding_text,
    load_exact_bge_tokenizer,
    make_embedding_boundary_handler,
)


def _exact_tokenizer_json() -> Path:
    configured = os.environ.get("BGE_TOKENIZER_JSON")
    path = Path(
        configured
        or "/private/tmp/bge-tokenizer-version-i/tokenizer.json"
    )
    if not path.is_file():
        pytest.skip(
            "Exact BAAI/bge-small-en-v1.5 tokenizer is not available locally."
        )
    return path


def _report_shaped_texts() -> list[str]:
    rng = random.Random(20260803)
    findings = [
        "epigastric pain",
        "microcytic anaemia",
        "mucosal inflammation",
        "negative troponin",
        "progressive dysphagia",
        "[CLS] literal [SEP] marker",
        "β-hCG and naïve T-cell notation",
    ]
    texts = []
    for report_index in range(64):
        clauses = []
        for premise_index in range(rng.randint(80, 180)):
            finding = rng.choice(findings)
            weight = rng.random()
            clauses.append(
                f"NODE-{report_index:03d}-{premise_index:03d} supports "
                f"{finding}; weight={weight:.8f}; source=CHUNK-"
                f"{rng.randrange(1, 500):04d}."
            )
        texts.append(" Community report: " + " ".join(clauses))
    return texts


def test_exact_bge_boundary_constrains_stochastic_graph_report_texts():
    tokenizer = load_exact_bge_tokenizer(
        "BAAI/bge-small-en-v1.5",
        tokenizer_json=_exact_tokenizer_json(),
    )
    texts = _report_shaped_texts()
    assert any(
        len(tokenizer.encode(text, add_special_tokens=True).ids)
        > BGE_MAX_INPUT_TOKENS
        for text in texts
    )

    for text in texts:
        forwarded, audit = constrain_embedding_text(text, tokenizer)
        repeated, repeated_audit = constrain_embedding_text(text, tokenizer)
        exact_count = len(
            tokenizer.encode(forwarded, add_special_tokens=True).ids
        )
        assert exact_count <= BGE_MAX_INPUT_TOKENS
        assert exact_count == audit["forwarded_bge_tokens"]
        assert repeated == forwarded
        assert repeated_audit == audit
        assert len(str(audit["source_sha256"])) == 64
        assert len(str(audit["forwarded_sha256"])) == 64


def test_embedding_payload_constrains_text_and_token_id_inputs():
    tokenizer = load_exact_bge_tokenizer(
        "BAAI/bge-small-en-v1.5",
        tokenizer_json=_exact_tokenizer_json(),
    )
    long_text = _report_shaped_texts()[0]
    rewritten, records = constrain_embedding_payload(
        {"model": "BAAI/bge-small-en-v1.5", "input": [long_text, "short"]},
        tokenizer,
        request_id=7,
    )
    assert len(records) == 2
    assert all(record.request_id == 7 for record in records)
    assert all(
        len(tokenizer.encode(value, add_special_tokens=True).ids)
        <= BGE_MAX_INPUT_TOKENS
        for value in rewritten["input"]
    )

    token_payload, token_records = constrain_embedding_payload(
        {"input": list(range(900))},
        tokenizer,
        request_id=8,
    )
    assert len(token_payload["input"]) == BGE_MAX_INPUT_TOKENS
    assert token_records[0].source_bge_tokens == 900
    assert token_records[0].forwarded_bge_tokens == BGE_MAX_INPUT_TOKENS


def test_proxy_logs_identity_and_never_forwards_over_limit(
    tmp_path: Path,
):
    tokenizer = load_exact_bge_tokenizer(
        "BAAI/bge-small-en-v1.5",
        tokenizer_json=_exact_tokenizer_json(),
    )
    captured: list[dict[str, object]] = []

    class UpstreamHandler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *args: object) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            body = b'{"data": []}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length))
            captured.append(payload)
            inputs = payload["input"]
            if isinstance(inputs, str):
                inputs = [inputs]
            body = json.dumps(
                {
                    "object": "list",
                    "data": [
                        {"object": "embedding", "index": index, "embedding": [0.0]}
                        | {"embedding": [1.0] + [0.0] * 383}
                        for index, _ in enumerate(inputs)
                    ],
                    "model": payload.get("model"),
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    audit_path = tmp_path / "embedding-boundary.jsonl"
    proxy = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_embedding_boundary_handler(
            upstream_url=f"http://127.0.0.1:{upstream.server_port}",
            tokenizer=tokenizer,
            audit_log=BoundaryAuditLog(audit_path),
        ),
    )
    proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()
    source_text = _report_shaped_texts()[1]
    try:
        request = Request(
            f"http://127.0.0.1:{proxy.server_port}/v1/embeddings",
            data=json.dumps(
                {
                    "model": "BAAI/bge-small-en-v1.5",
                    "input": [source_text, "short evidence"],
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=10) as response:
            assert response.status == 200
            response_payload = json.loads(response.read())
    finally:
        proxy.shutdown()
        upstream.shutdown()
        proxy.server_close()
        upstream.server_close()

    assert len(captured) == 1
    assert len(captured[0]["input"]) > 2
    assert len(response_payload["data"]) == 2
    assert all(
        isinstance(value, list) and len(value) <= BGE_MAX_INPUT_TOKENS
        for value in captured[0]["input"]
    )
    boundary_audit = audit_embedding_boundary(audit_path)
    assert boundary_audit["all_outbound_inputs_within_bge_limit"] is True
    assert boundary_audit["maximum_forwarded_bge_tokens"] <= 512
    assert boundary_audit["truncated_input_count"] == 0
    assert boundary_audit["split_input_count"] == 1
    assert boundary_audit["raw_tokens_lost"] == 0
    audit_text = audit_path.read_text(encoding="utf-8")
    assert source_text not in audit_text
    assert "source_sha256" in audit_text
    assert "token_ids_sha256" in audit_text
