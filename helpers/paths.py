from __future__ import annotations

from pathlib import Path


OUTPUT_ROOT = Path("output")
LOG_ROOT = OUTPUT_ROOT / "logs"
FRAMEWORK_LOG_ROOT = LOG_ROOT / "framework"
CACHE_ROOT = OUTPUT_ROOT / "cache"

EVENT_LOG_PATH = FRAMEWORK_LOG_ROOT / "events.jsonl"
EVAL_RECORD_LOG_PATH = FRAMEWORK_LOG_ROOT / "eval_records.jsonl"
INDEX_BUILD_LOG_PATH = FRAMEWORK_LOG_ROOT / "index_build.jsonl"
EMBEDDING_BENCHMARK_LOG_PATH = FRAMEWORK_LOG_ROOT / "embedding_benchmark.jsonl"
MODEL_BENCHMARK_LOG_PATH = FRAMEWORK_LOG_ROOT / "model_benchmark.jsonl"
