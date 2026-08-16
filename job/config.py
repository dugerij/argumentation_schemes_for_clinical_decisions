from __future__ import annotations

import os
import re
from pathlib import Path, PurePosixPath


RUN_SCOPE_ENV = "RUN_SCOPE"
RUN_ID_ENV = "RUN_ID"
RUN_PHASE_ENV = "RUN_PHASE"
RUN_PHASES = ("comparison", "retrieval", "evaluation", "mapping_diagnostic")

SCOPES: dict[str, dict[str, object]] = {
    "validation": {
        "queries_relative": Path(
            "output/graphrag/microsoft-graphrag-controlled-corpus-v1/"
            "validation/bge-boundary-dry-run-20260803T023000Z/queries/"
            "validation_queries.jsonl"
        ),
        "queries_sha256": (
            "5637e26929a415a9da82f6d852f1d2f81ce927c2a7b5b8cf057f629e75d73b01"
        ),
        "case_count": 5,
        "timeout": "2h",
        "run_name": "comparison-validation",
    },
    "development": {
        "queries_relative": Path(
            "output/graphrag/microsoft-graphrag-controlled-corpus-v1/"
            "development/sealed-development-queries-v1/development_queries.jsonl"
        ),
        "queries_sha256": (
            "23b9d846e8278a0adb99f4dd6f10c842b5e3d314703fbd55a1a175ebefa0a23e"
        ),
        "case_count": 88,
        "timeout": "4h",
        "run_name": "comparison-development",
    },
}


def selected_scope() -> str:
    scope = os.environ.get(RUN_SCOPE_ENV, "validation")
    if scope not in SCOPES:
        raise ValueError(f"RUN_SCOPE must be one of {tuple(SCOPES)}; got {scope!r}")
    return scope


def selected_scope_config() -> dict[str, object]:
    return dict(SCOPES[selected_scope()])


def selected_run_id() -> str:
    run_id = os.environ.get(RUN_ID_ENV, "").strip()
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,79}", run_id):
        raise ValueError(
            "Set RUN_ID to a unique 3-80 character lowercase identifier using "
            "letters, numbers, dots, underscores, or hyphens."
        )
    return run_id


def selected_phase() -> str:
    phase = os.environ.get(RUN_PHASE_ENV, "comparison")
    if phase not in RUN_PHASES:
        raise ValueError(f"RUN_PHASE must be one of {RUN_PHASES}; got {phase!r}")
    return phase


RUN_SCOPE = selected_scope()
RUN_CONFIG = selected_scope_config()
RUN_ID = selected_run_id()
RUN_PHASE = selected_phase()
OUTPUT_TARGET = f"runs/{RUN_ID}"
RUNTIME_SCRATCH_ROOT = f"/tmp/argumentation-schemes/{RUN_ID}"


def validate_output_target(value: object) -> tuple[str, str]:
    if not isinstance(value, str) or value != OUTPUT_TARGET:
        raise ValueError(f"OUTPUT_TARGET must equal {OUTPUT_TARGET!r}.")
    parts = PurePosixPath(value).parts
    if parts != ("runs", RUN_ID):
        raise ValueError("OUTPUT_TARGET must be the canonical runs/<RUN_ID> path.")
    return parts
