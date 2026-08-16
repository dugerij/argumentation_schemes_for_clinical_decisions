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
    "test": {
        "queries_relative": Path(
            "output/graphrag/microsoft-graphrag-controlled-corpus-v1/"
            "test/sealed-test-queries-v1/test_queries.jsonl"
        ),
        "queries_sha256": (
            "bac9ea39ce67b3657a1bcf72e729e5417501c607f1bcc97008f7de98ed5d6763"
        ),
        "case_count": 423,
        "timeout": "8h",
        "run_name": "comparison-test",
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


# RUN_SCOPE, RUN_CONFIG, RUN_ID, RUN_PHASE, OUTPUT_TARGET and RUNTIME_SCRATCH_ROOT all
# depend on environment variables that a caller (job.prepare, job.run) sets before
# importing this module. They are resolved lazily, on first attribute access, rather
# than at import time: something that only needs SCOPES -- the notebook does, to list
# the available scopes before it has chosen RUN_ID -- can import this module before
# RUN_ID exists. A caller that does need one of them still sees the identical
# ValueError, at the identical `from job.config import RUN_ID` line, because attribute
# access is what triggers resolution.
_RESOLVED: dict[str, object] = {}
_LAZY_NAMES = frozenset(
    {"RUN_SCOPE", "RUN_CONFIG", "RUN_ID", "RUN_PHASE", "OUTPUT_TARGET", "RUNTIME_SCRATCH_ROOT"}
)


def _resolved() -> dict[str, object]:
    if not _RESOLVED:
        run_id = selected_run_id()
        _RESOLVED["RUN_SCOPE"] = selected_scope()
        _RESOLVED["RUN_CONFIG"] = selected_scope_config()
        _RESOLVED["RUN_ID"] = run_id
        _RESOLVED["RUN_PHASE"] = selected_phase()
        _RESOLVED["OUTPUT_TARGET"] = f"runs/{run_id}"
        _RESOLVED["RUNTIME_SCRATCH_ROOT"] = f"/tmp/argumentation-schemes/{run_id}"
    return _RESOLVED


def __getattr__(name: str) -> object:
    if name in _LAZY_NAMES:
        return _resolved()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def validate_output_target(value: object) -> tuple[str, str]:
    resolved = _resolved()
    output_target = resolved["OUTPUT_TARGET"]
    if not isinstance(value, str) or value != output_target:
        raise ValueError(f"OUTPUT_TARGET must equal {output_target!r}.")
    parts = PurePosixPath(value).parts
    if parts != ("runs", resolved["RUN_ID"]):
        raise ValueError("OUTPUT_TARGET must be the canonical runs/<RUN_ID> path.")
    return parts
