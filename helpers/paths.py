from __future__ import annotations

import os
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

DEFAULT_MIMIC_DISCHARGE_CANDIDATES = (
    "data/mimic_iv_note/discharge.csv",
    "data/mimic_iv_note/discharge.csv.gz",
    "data/mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note/discharge.csv",
    "data/mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note/discharge.csv.gz",
)


def resolve_mimic_discharge_csv(explicit_path: str | None = None) -> Path:
    """Resolve the discharge-note source file from an explicit path, env var, or common local layouts."""

    configured = explicit_path or os.environ.get("MIMIC_DISCHARGE_CSV")
    if configured:
        configured_path = Path(configured)
        if configured_path.exists():
            return configured_path

    for candidate in DEFAULT_MIMIC_DISCHARGE_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return path

    return Path(configured) if configured else Path(DEFAULT_MIMIC_DISCHARGE_CANDIDATES[0])
