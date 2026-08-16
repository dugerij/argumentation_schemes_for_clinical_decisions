"""Stage the one current runtime for direct Hugging Face CLI submission."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
from pathlib import Path

from clinical_cds.experiment import protocol_sha256
from clinical_cds.normalization import GRAPH_LABEL_UMLS_QUERIES
from clinical_cds.terminology.local_umls import build_local_umls_subset
from clinical_cds.terminology.candidates import extract_candidate_terms
from hf_job.config import OUTPUT_TARGET, RUN_CONFIG, RUN_ID, RUN_PHASE, RUN_SCOPE


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGING_ROOT = PROJECT_ROOT / ".hf-runs" / RUN_ID
RECOVERED_INDEX = PROJECT_ROOT / (
    "output/graphrag/microsoft-graphrag-controlled-corpus-v1/corpus-index/"
    "lossless-bge-recovery-20260803T040000Z"
)
CONTROLLED_TABLES = PROJECT_ROOT / (
    "output/graphrag/evidence-grounded-v1-validation-20260802t230000z-v3/"
    "job-6a6feeb36b79c09949c1ff48/workspace"
)
TABLES = (
    "communities.parquet",
    "community_reports.parquet",
    "documents.parquet",
    "entities.parquet",
    "relationships.parquet",
    "text_units.parquet",
)
FULL_UMLS_DB = PROJECT_ROOT / os.environ.get(
    "UMLS_DB", "output/cache/umls_local.sqlite3"
)
RUN_CASE_LIMIT_ENV = "RUN_CASE_LIMIT"
RUN_SAMPLE_SEED_ENV = "RUN_SAMPLE_SEED"
DEFAULT_SAMPLE_SEED = "bounded-development-v1"
RESUME_COMPARISON_ROOT_ENV = "RESUME_COMPARISON_ROOT"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def select_bounded_records(
    records: list[dict[str, object]],
    *,
    limit: int | None,
    seed: str,
) -> list[dict[str, object]]:
    """Select cases by opaque case identity only, never by labels or content."""
    if limit is None:
        return records
    if not 1 <= limit < len(records):
        raise ValueError(
            f"RUN_CASE_LIMIT must be between 1 and {len(records) - 1}; got {limit}."
        )
    return sorted(
        records,
        key=lambda record: hashlib.sha256(
            f"{seed}:{record['case_id']}".encode()
        ).hexdigest(),
    )[:limit]


def selected_query_payload() -> tuple[Path, list[dict[str, object]], dict[str, object] | None]:
    query_path = PROJECT_ROOT / Path(RUN_CONFIG["queries_relative"])
    records = [
        json.loads(line)
        for line in query_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    raw_limit = os.environ.get(RUN_CASE_LIMIT_ENV, "").strip()
    if not raw_limit:
        return query_path, records, None
    if RUN_SCOPE != "development":
        raise ValueError("RUN_CASE_LIMIT is permitted only for development runs.")
    try:
        limit = int(raw_limit)
    except ValueError as exc:
        raise ValueError("RUN_CASE_LIMIT must be an integer.") from exc
    seed = os.environ.get(RUN_SAMPLE_SEED_ENV, DEFAULT_SAMPLE_SEED).strip()
    if not seed or len(seed) > 80:
        raise ValueError("RUN_SAMPLE_SEED must contain 1-80 characters.")
    selected = select_bounded_records(records, limit=limit, seed=seed)
    return query_path, selected, {
        "method": "sha256-seeded-case-id-v1",
        "seed": seed,
        "limit": limit,
        "source_case_count": len(records),
        "source_query_sha256": file_sha256(query_path),
        "selection_uses": "case_id_only",
    }


def encoded_query_records(records: list[dict[str, object]]) -> bytes:
    return b"".join(
        (json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n").encode()
        for record in records
    )


def directory_manifest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def module_file(module: str) -> Path | None:
    candidate = PROJECT_ROOT / (module.replace(".", "/") + ".py")
    if candidate.is_file():
        return candidate
    package = PROJECT_ROOT / module.replace(".", "/") / "__init__.py"
    return package if package.is_file() else None


def local_imports(path: Path) -> set[Path]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    relative = path.relative_to(PROJECT_ROOT)
    package = list(relative.with_suffix("").parts[:-1])
    if path.name == "__init__.py":
        package = list(relative.parts[:-1])
    found: set[Path] = set()
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = package[:]
            if node.level:
                base = base[: len(base) - node.level + 1]
            if node.module:
                base.extend(node.module.split("."))
            modules.append(".".join(base) if node.level else (node.module or ""))
            prefix = ".".join(base)
            modules.extend(
                ".".join(part for part in (prefix, alias.name) if part)
                for alias in node.names
                if alias.name != "*"
            )
        for module in modules:
            if module.startswith(("clinical_cds", "graphrag_runtime", "hf_job")):
                source = module_file(module)
                if source:
                    found.add(source)
    return found


def controlled_diagnosis_labels(documents_path: Path) -> set[str]:
    import pandas as pd

    documents = pd.read_parquet(documents_path, columns=["raw_data"])
    return {
        str(dict(raw)["diagnosis_label"]).strip()
        for raw in documents["raw_data"].tolist()
        if str(dict(raw).get("diagnosis_label") or "").strip()
    }


def controlled_clinical_terms(documents_path: Path) -> set[str]:
    import pandas as pd

    documents = pd.read_parquet(documents_path, columns=["raw_data"])
    terms: set[str] = set()
    for raw in documents["raw_data"].tolist():
        item = dict(raw)
        terms.update(extract_candidate_terms(str(item.get("text") or ""), limit=24))
    return terms


def runtime_files() -> tuple[Path, ...]:
    required = {
        PROJECT_ROOT / "hf_job/run.py",
        PROJECT_ROOT / "graphrag_runtime/medgemma_chat_boundary.py",
    }
    pending = list(required)
    while pending:
        path = pending.pop()
        parent = path.parent
        while parent != PROJECT_ROOT:
            init = parent / "__init__.py"
            if init.is_file() and init not in required:
                required.add(init)
                pending.append(init)
            parent = parent.parent
        for dependency in local_imports(path):
            if dependency not in required:
                required.add(dependency)
                pending.append(dependency)
    required.add(PROJECT_ROOT / "graphrag_runtime/requirements.graphrag.txt")
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Current runtime is incomplete: {missing}")
    return tuple(sorted(required))


def stage_source(destination: Path) -> dict[str, object]:
    query_path, records, sample = selected_query_payload()
    effective_query_sha256 = (
        hashlib.sha256(encoded_query_records(records)).hexdigest()
        if sample else file_sha256(query_path)
    )
    files = runtime_files()
    for source in files:
        target = destination / source.relative_to(PROJECT_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    hashes = {
        source.relative_to(PROJECT_ROOT).as_posix(): file_sha256(source)
        for source in files
    }
    manifest: dict[str, object] = {
        "query_only": True,
        "scope": RUN_SCOPE,
        "phase": RUN_PHASE,
        "case_count": len(records),
        "query_sha256": effective_query_sha256,
        "sample": sample,
        "output_target": OUTPUT_TARGET,
        "run_id": RUN_ID,
        "runtime_sha256": canonical_sha256(hashes),
        "protocol_sha256": protocol_sha256(),
        "files": hashes,
    }
    (destination / "query_runtime_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def stage_input(destination: Path) -> dict[str, object]:
    query_path, records, sample = selected_query_payload()
    required = [
        query_path,
        RECOVERED_INDEX / "corpus_index_manifest.json",
        RECOVERED_INDEX / "integrity_report.json",
        RECOVERED_INDEX / "model_manifest.json",
        CONTROLLED_TABLES / "settings.json",
        *(CONTROLLED_TABLES / "output" / name for name in TABLES),
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required local run inputs are missing: {missing}")

    output = destination / "workspace/output"
    output.mkdir(parents=True)
    for name in TABLES:
        shutil.copy2(CONTROLLED_TABLES / "output" / name, output / name)
    shutil.copytree(RECOVERED_INDEX / "vector_store", output / "lancedb")
    shutil.copy2(CONTROLLED_TABLES / "settings.json", destination / "workspace/settings.json")
    (destination / "queries").mkdir()
    staged_queries = destination / "queries/validation_queries.jsonl"
    if sample:
        staged_queries.write_bytes(encoded_query_records(records))
    else:
        shutil.copy2(query_path, staged_queries)
    for name in ("corpus_index_manifest.json", "integrity_report.json", "model_manifest.json"):
        shutil.copy2(RECOVERED_INDEX / name, destination / name)

    terminology_terms = controlled_diagnosis_labels(
        CONTROLLED_TABLES / "output/documents.parquet"
    )
    terminology_terms.update(controlled_clinical_terms(
        CONTROLLED_TABLES / "output/documents.parquet"
    ))
    terminology_terms.update(GRAPH_LABEL_UMLS_QUERIES.values())
    for record in records:
        for item in record["saved_query_payload"]["patient_evidence"]:
            terminology_terms.update(extract_candidate_terms(
                str(item.get("text") or ""), limit=48
            ))
    terminology_db = destination / "terminology/umls_study_subset.sqlite3"
    build_local_umls_subset(FULL_UMLS_DB, terminology_db, terminology_terms)

    resume: dict[str, object] | None = None
    if RUN_PHASE == "evaluation":
        raw_resume = os.environ.get(RESUME_COMPARISON_ROOT_ENV, "").strip()
        if not raw_resume:
            raise EnvironmentError(
                "RESUME_COMPARISON_ROOT is required for evaluation-only preparation."
            )
        resume_root = Path(raw_resume).resolve()
        required_resume = (
            "predictions.jsonl", "argument_traces.jsonl", "manifest.json",
            "adjudications.jsonl", "presentation_responses.jsonl",
            "presentation_traces.jsonl",
        )
        missing_resume = [name for name in required_resume
                          if not (resume_root / name).is_file()]
        if missing_resume:
            raise FileNotFoundError(
                f"Frozen comparison resume artifacts are missing: {missing_resume}"
            )
        resume_destination = destination / "resume/comparison"
        resume_destination.mkdir(parents=True)
        for name in required_resume:
            shutil.copy2(resume_root / name, resume_destination / name)
        prediction_rows = [json.loads(line) for line in
            (resume_root / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
        trace_rows = [json.loads(line) for line in
            (resume_root / "argument_traces.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
        expected_ids = [str(record["case_id"]) for record in records]
        if len(prediction_rows) != len(expected_ids) * 4:
            raise RuntimeError("Evaluation resume must contain four predictions per case.")
        if sorted({str(row["case_id"]) for row in prediction_rows}) != sorted(expected_ids):
            raise RuntimeError("Evaluation resume prediction cases changed.")
        if len(trace_rows) != len(expected_ids) or sorted(
            str(row["case_id"]) for row in trace_rows
        ) != sorted(expected_ids):
            raise RuntimeError("Evaluation resume argument-trace cases changed.")
        resume_files = directory_manifest(resume_destination)
        resume = {
            "source_root": str(resume_root),
            "files": resume_files,
            "package_sha256": canonical_sha256(resume_files),
            "prediction_count": len(prediction_rows),
            "argument_trace_count": len(trace_rows),
            "case_ids": expected_ids,
        }

    files = directory_manifest(destination)
    index_manifest = json.loads(
        (RECOVERED_INDEX / "corpus_index_manifest.json").read_text(encoding="utf-8")
    )
    manifest: dict[str, object] = {
        "query_only": True,
        "scope": RUN_SCOPE,
        "phase": RUN_PHASE,
        "output_target": OUTPUT_TARGET,
        "run_id": RUN_ID,
        "files": files,
        "package_sha256": canonical_sha256(files),
        "vector_tree_sha256": index_manifest["vector_store"]["tree_sha256"],
        "corpus_index_manifest_sha256": file_sha256(
            RECOVERED_INDEX / "corpus_index_manifest.json"
        ),
        "query_sha256": file_sha256(staged_queries),
        "sample": sample,
        "case_ids": [str(record["case_id"]) for record in records],
        "case_count": len(records),
        "patient_documents_indexed": False,
        "umls": {
            "release": "2026AA",
            "scope": "study-diagnosis-cuis-and-aliases",
            "database_relative": "terminology/umls_study_subset.sqlite3",
            "database_sha256": file_sha256(terminology_db),
            "raw_rrf_included": False,
        },
        "evaluation_resume": resume,
    }
    (destination / "query_package_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    if STAGING_ROOT.exists():
        raise FileExistsError(
            f"{STAGING_ROOT} already exists. Choose a new RUN_ID; do not reuse a run."
        )
    source = STAGING_ROOT / "source"
    inputs = STAGING_ROOT / "input"
    output = STAGING_ROOT / "output"
    source.mkdir(parents=True)
    inputs.mkdir()
    output.mkdir()
    (output / ".keep").write_text("hf-job-output\n", encoding="utf-8")
    source_manifest = stage_source(source)
    input_manifest = stage_input(inputs)
    print(json.dumps({
        "run_id": RUN_ID,
        "scope": RUN_SCOPE,
        "phase": RUN_PHASE,
        "staging_root": str(STAGING_ROOT),
        "source_files": len(source_manifest["files"]),
        "case_count": input_manifest["case_count"],
        "timeout": RUN_CONFIG["timeout"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
