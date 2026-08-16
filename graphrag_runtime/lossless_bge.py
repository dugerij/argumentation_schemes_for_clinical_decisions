from __future__ import annotations

import hashlib
import json
import math
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer


BGE_MAX_INPUT_TOKENS = 512
LOSSLESS_POLICY = "exact_bge_no_overlap_weighted_mean_l2_v1"


def sha256_token_ids(ids: list[int]) -> str:
    payload = ",".join(str(value) for value in ids).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExactTokenChunk:
    chunk_index: int
    raw_start: int
    raw_end: int
    raw_token_count: int
    model_token_count: int
    token_ids_sha256: str
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class LosslessEmbeddingPlan:
    source_sha256: str
    source_token_ids_sha256: str
    source_raw_token_count: int
    source_model_token_count: int
    chunk_count: int
    unchanged: bool
    policy: str
    chunks: tuple[ExactTokenChunk, ...]


def plan_lossless_embedding(
    text: str,
    tokenizer: Tokenizer,
    *,
    max_tokens: int = BGE_MAX_INPUT_TOKENS,
) -> LosslessEmbeddingPlan:
    raw_ids = tokenizer.encode(text, add_special_tokens=False).ids
    model_ids = tokenizer.encode(text, add_special_tokens=True).ids
    empty_model_ids = tokenizer.encode("", add_special_tokens=True).ids
    if len(empty_model_ids) != 2:
        raise RuntimeError("Expected exactly two BGE boundary special tokens.")
    prefix, suffix = empty_model_ids
    if model_ids != [prefix, *raw_ids, suffix]:
        raise RuntimeError("BGE special-token layout is not the verified CLS/text/SEP form.")
    raw_capacity = max_tokens - 2
    if raw_capacity <= 0:
        raise ValueError("Embedding boundary leaves no room for evidence tokens.")

    raw_slices = (
        [raw_ids]
        if len(model_ids) <= max_tokens
        else [raw_ids[start : start + raw_capacity] for start in range(0, len(raw_ids), raw_capacity)]
    )
    if not raw_slices:
        raw_slices = [[]]
    chunks: list[ExactTokenChunk] = []
    cursor = 0
    recovered: list[int] = []
    for index, raw_slice in enumerate(raw_slices):
        chunk_ids = [prefix, *raw_slice, suffix]
        if len(chunk_ids) > max_tokens:
            raise RuntimeError("Lossless BGE chunk exceeds the hard model boundary.")
        recovered.extend(raw_slice)
        chunks.append(
            ExactTokenChunk(
                chunk_index=index,
                raw_start=cursor,
                raw_end=cursor + len(raw_slice),
                raw_token_count=len(raw_slice),
                model_token_count=len(chunk_ids),
                token_ids_sha256=sha256_token_ids(chunk_ids),
                token_ids=tuple(chunk_ids),
            )
        )
        cursor += len(raw_slice)
    if recovered != raw_ids or cursor != len(raw_ids):
        raise RuntimeError("Lossless BGE chunking lost, duplicated, or reordered tokens.")
    return LosslessEmbeddingPlan(
        source_sha256=sha256_text(text),
        source_token_ids_sha256=sha256_token_ids(model_ids),
        source_raw_token_count=len(raw_ids),
        source_model_token_count=len(model_ids),
        chunk_count=len(chunks),
        unchanged=len(chunks) == 1 and len(model_ids) <= max_tokens,
        policy=LOSSLESS_POLICY,
        chunks=tuple(chunks),
    )


def token_weighted_mean_l2(
    vectors: list[list[float] | np.ndarray],
    raw_token_counts: list[int],
    *,
    expected_dimension: int = 384,
) -> list[float]:
    if not vectors or len(vectors) != len(raw_token_counts):
        raise ValueError("Vectors and token weights must be non-empty and cardinality-aligned.")
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != expected_dimension:
        raise RuntimeError("BGE embedding dimension mismatch.")
    if not np.isfinite(matrix).all():
        raise RuntimeError("BGE embedding contains NaN or infinity.")
    weights = np.asarray([max(1, value) for value in raw_token_counts], dtype=np.float64)
    pooled = np.average(matrix, axis=0, weights=weights)
    norm = float(np.linalg.norm(pooled))
    if not math.isfinite(norm) or norm <= 0:
        raise RuntimeError("Cannot normalize a zero or non-finite pooled embedding.")
    result = pooled / norm
    if result.shape != (expected_dimension,) or not np.isfinite(result).all():
        raise RuntimeError("Pooled BGE embedding failed validation.")
    return result.astype(np.float32).tolist()


class OnnxBGEEncoder:
    """Official BGE ONNX CLS-pooling encoder with normalized output."""

    def __init__(self, model_path: Path, *, dimension: int = 384) -> None:
        import onnxruntime as ort

        self.dimension = dimension
        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        inputs = {value.name for value in self._session.get_inputs()}
        if inputs != {"input_ids", "attention_mask", "token_type_ids"}:
            raise RuntimeError(f"Unexpected BGE ONNX inputs: {sorted(inputs)}")
        outputs = self._session.get_outputs()
        if not outputs or outputs[0].name != "last_hidden_state":
            raise RuntimeError("Unexpected BGE ONNX output contract.")

    def encode_token_ids(
        self,
        token_id_rows: list[list[int] | tuple[int, ...]],
        *,
        batch_size: int = 32,
    ) -> list[list[float]]:
        if not token_id_rows:
            return []
        result: list[list[float]] = []
        for start in range(0, len(token_id_rows), batch_size):
            rows = token_id_rows[start : start + batch_size]
            if any(len(row) > BGE_MAX_INPUT_TOKENS for row in rows):
                raise RuntimeError("Attempted to send an over-limit row to BGE ONNX.")
            width = max(len(row) for row in rows)
            input_ids = np.zeros((len(rows), width), dtype=np.int64)
            attention = np.zeros((len(rows), width), dtype=np.int64)
            for index, row in enumerate(rows):
                input_ids[index, : len(row)] = row
                attention[index, : len(row)] = 1
            token_types = np.zeros_like(input_ids)
            hidden = self._session.run(
                ["last_hidden_state"],
                {
                    "input_ids": input_ids,
                    "attention_mask": attention,
                    "token_type_ids": token_types,
                },
            )[0]
            vectors = hidden[:, 0, :].astype(np.float64)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            if (
                vectors.shape[1] != self.dimension
                or not np.isfinite(vectors).all()
                or not np.isfinite(norms).all()
                or np.any(norms <= 0)
            ):
                raise RuntimeError("Official BGE ONNX output failed validation.")
            vectors /= norms
            result.extend(vectors.astype(np.float32).tolist())
        return result


class HashOnlyAuditLog:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, sort_keys=True) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as target:
            target.write(line)


def plan_audit_record(plan: LosslessEmbeddingPlan) -> dict[str, Any]:
    return {
        "source_sha256": plan.source_sha256,
        "source_token_ids_sha256": plan.source_token_ids_sha256,
        "source_raw_token_count": plan.source_raw_token_count,
        "source_model_token_count": plan.source_model_token_count,
        "chunk_count": plan.chunk_count,
        "unchanged": plan.unchanged,
        "policy": plan.policy,
        "chunks": [
            {
                key: value
                for key, value in asdict(chunk).items()
                if key != "token_ids"
            }
            for chunk in plan.chunks
        ],
    }
