from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from graphrag_runtime.medgemma_prompt_budget import (
    ARGUMENT_CRITIC_COMPLETION_TOKENS,
    ARGUMENT_GENERATOR_COMPLETION_TOKENS,
    COMPLETION_TOKENS,
    GRAPHRAG_CONTEXT_TOKENS,
    MAX_COMPLETION_TOKENS,
    MAX_MODEL_LEN,
    MEDGEMMA_MODEL_ID,
    MEDGEMMA_MODEL_REVISION,
    SAFETY_MARGIN_TOKENS,
    frozen_budget_config,
)
from graphrag_runtime.vllm_config import (
    PINNED_VLLM_VERSION,
    STRUCTURED_OUTPUT_CONFIG_JSON,
    completion_server_command,
)
from graphrag_runtime.queries import (
    patient_concept_retrieval_query,
    patient_finding_retrieval_queries,
)
from graphrag_runtime.provenance_contract import (
    PROVENANCE_CONTRACT_ID,
    canonical_sha256 as provenance_sha256,
)
from hf_job.config import (
    OUTPUT_TARGET,
    RUN_CONFIG,
    RUN_ID,
    RUN_PHASE,
    RUNTIME_SCRATCH_ROOT,
    RUN_SCOPE,
    validate_output_target,
)


CHAT_MODEL_ID = MEDGEMMA_MODEL_ID
CHAT_MODEL_REVISION = MEDGEMMA_MODEL_REVISION
JUDGE_MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
JUDGE_MODEL_REVISION = "5a5a776300a41aaa681dd7ff0106608ef2bc90db"
EMBEDDING_MODEL_ID = "BAAI/bge-small-en-v1.5"
EMBEDDING_MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
CHAT_URL = "http://127.0.0.1:8000/v1"
CHAT_UPSTREAM_URL = "http://127.0.0.1:8003/v1"
JUDGE_URL = "http://127.0.0.1:8005/v1"
JUDGE_UPSTREAM_URL = "http://127.0.0.1:8004/v1"
EMBEDDING_URL = "http://127.0.0.1:8001/v1"
EMBEDDING_UPSTREAM_URL = "http://127.0.0.1:8002/v1"
VLLM_VERSION = PINNED_VLLM_VERSION
STRUCTURED_OUTPUTS_CONFIG = STRUCTURED_OUTPUT_CONFIG_JSON
VLLM_RPC_BASE_PATH_ENV = "VLLM_RPC_BASE_PATH"
VLLM_IPC_ROOT_ENV = "VLLM_IPC_ROOT"


def validate_argument_generation_contract() -> dict[str, Any]:
    """Fail before GPU startup if client retries cannot pass the server boundary."""
    from clinical_cds.argumentation_v3 import (
        attack_validation_schema,
        differential_attack_schema,
        direct_differential_schema,
        evidence_aware_activation_schema,
    )
    from clinical_cds.model import _vllm_compatible_output_schema
    from clinical_cds.runner import MAX_RETRY_OUTPUT_TOKENS

    caps = {
        "runner_retry": MAX_RETRY_OUTPUT_TOKENS,
        "argument_generator": ARGUMENT_GENERATOR_COMPLETION_TOKENS,
        "argument_critic": ARGUMENT_CRITIC_COMPLETION_TOKENS,
    }
    if any(cap > MAX_COMPLETION_TOKENS for cap in caps.values()):
        raise RuntimeError(
            "Argument completion allowance exceeds the frozen MedGemma boundary."
        )
    schemas = {
        "attack_validation": attack_validation_schema(),
        "activation": evidence_aware_activation_schema(
            tuple(f"C{index}" for index in range(1, 9)),
            {f"P{index}": f"C{index}" for index in range(1, 9)},
        ),
        "direct": direct_differential_schema(
            ("C1", "C2"),
            tuple(f"P{index}" for index in range(1, 9)),
            tuple(f"S{index}" for index in range(1, 9)),
            ("Child",),
        ),
        "verifier": differential_attack_schema(
            ("C1", "C2"), tuple(f"P{index}" for index in range(1, 9))
        ),
    }

    def unbounded_arrays(value: Any, path: str = "$") -> list[str]:
        failures: list[str] = []
        if isinstance(value, dict):
            if value.get("type") == "array" and "maxItems" not in value:
                failures.append(path)
            for key, child in value.items():
                failures.extend(unbounded_arrays(child, f"{path}.{key}"))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                failures.extend(unbounded_arrays(child, f"{path}[{index}]"))
        return failures

    failures = {
        name: unbounded_arrays(_vllm_compatible_output_schema(schema))
        for name, schema in schemas.items()
    }
    failures = {name: paths for name, paths in failures.items() if paths}
    if failures:
        raise RuntimeError(
            f"Version III structured output contains unbounded arrays: {failures}"
        )
    return {
        "status": "passed",
        "maximum_completion_tokens": MAX_COMPLETION_TOKENS,
        "stage_completion_tokens": caps,
        "vllm_compatible_arrays_bounded": True,
    }
VLLM_IPC_ROOT_DEFAULT = "/tmp/vli"
LINUX_UNIX_SOCKET_PATH_LIMIT_BYTES = 107
VLLM_IPC_SAFETY_MARGIN_BYTES = 20
VLLM_IPC_MAX_PATH_BYTES = (
    LINUX_UNIX_SOCKET_PATH_LIMIT_BYTES - VLLM_IPC_SAFETY_MARGIN_BYTES
)
VLLM_UUID_WORST_CASE = "ffffffff-ffff-ffff-ffff-ffffffffffff"
PINNED_VLLM_SOURCE_SHA256 = {
    "vllm/envs.py": "34adae037709253e5a123873dbb3336aa7aa45dd96bab2b98bd4a74f11b7c45f",
    "vllm/utils/network_utils.py": (
        "da7453a23fd15bc6ffa5ef45419a08b5effdc2d8c535d34876d776e57d37e0e6"
    ),
}
EXPECTED_VECTOR_TREE_SHA256 = (
    "86813e21aef0d960e1592a372e8cd01e50461f817ef8497d200664934a3de462"
)
EXPECTED_CORPUS_INDEX_MANIFEST_SHA256 = (
    "b3293c7231dc78f3211c4e9302690d676126735482f3279fe36a0f936efe3011"
)
REQUIRED_TABLES = (
    "communities.parquet",
    "community_reports.parquet",
    "documents.parquet",
    "entities.parquet",
    "relationships.parquet",
    "text_units.parquet",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inside-graphrag-env", action="store_true")
    parser.add_argument("--synthetic-preflight", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    aggregate = hashlib.sha256()
    for path in sorted(item for item in Path(root).rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(file_sha256(path).encode("ascii"))
        aggregate.update(b"\n")
    return aggregate.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _clear_existing_output_roots(*roots: Path) -> None:
    for root in roots:
        if root.exists():
            shutil.rmtree(root)


def _directory_manifest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def require_package(root: Path) -> dict[str, Any]:
    manifest_path = root / "query_package_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Query package manifest is missing.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("files") or {}
    actual = _directory_manifest(root)
    actual.pop("query_package_manifest.json", None)
    if actual != expected:
        raise RuntimeError("Query package files differ from the immutable manifest.")
    if manifest.get("query_only") is not True:
        raise RuntimeError("Package is not frozen as query-only.")
    if manifest.get("scope") != RUN_SCOPE:
        raise RuntimeError("Query package scope differs from the selected scope.")
    if manifest.get("phase") != RUN_PHASE:
        raise RuntimeError("Query package phase differs from the selected phase.")
    case_ids = tuple(str(value) for value in manifest.get("case_ids") or ())
    sample = manifest.get("sample")
    expected_case_count = (
        int(sample["limit"])
        if isinstance(sample, dict)
        else int(RUN_CONFIG["case_count"])
    )
    if sample and (
        RUN_SCOPE != "development"
        or sample.get("method") != "sha256-seeded-case-id-v1"
        or sample.get("selection_uses") != "case_id_only"
        or int(sample.get("source_case_count") or 0) != int(RUN_CONFIG["case_count"])
        or sample.get("source_query_sha256") != RUN_CONFIG["queries_sha256"]
    ):
        raise RuntimeError("Bounded development sample binding changed.")
    if (
        len(case_ids) != expected_case_count
        or len(set(case_ids)) != len(case_ids)
        or int(manifest.get("case_count") or 0) != len(case_ids)
    ):
        raise RuntimeError("Query package case identity binding changed.")
    queries_path = root / "queries/validation_queries.jsonl"
    if (
        not queries_path.is_file()
        or file_sha256(queries_path) != manifest.get("query_sha256")
        or (not sample and manifest.get("query_sha256") != RUN_CONFIG["queries_sha256"])
    ):
        raise RuntimeError("Query package artifact hash binding changed.")
    if manifest.get("vector_tree_sha256") != EXPECTED_VECTOR_TREE_SHA256:
        raise RuntimeError("Recovered vector tree binding changed.")
    if (
        manifest.get("corpus_index_manifest_sha256")
        != EXPECTED_CORPUS_INDEX_MANIFEST_SHA256
    ):
        raise RuntimeError("Recovered corpus-index manifest binding changed.")
    return manifest


def require_source_package(root: Path) -> dict[str, Any]:
    path = root / "query_runtime_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = manifest.get("files") or {}
    actual = _directory_manifest(root)
    actual.pop("query_runtime_manifest.json", None)
    if actual != expected:
        raise RuntimeError("Query runtime files differ from the immutable manifest.")
    if manifest.get("query_only") is not True:
        raise RuntimeError("Runtime is not frozen as query-only.")
    sample = manifest.get("sample")
    if (
        manifest.get("scope") != RUN_SCOPE
        or manifest.get("phase") != RUN_PHASE
        or (sample is None and int(manifest.get("case_count") or 0) != int(RUN_CONFIG["case_count"]))
        or (sample is None and manifest.get("query_sha256") != RUN_CONFIG["queries_sha256"])
    ):
        raise RuntimeError("Query runtime scope binding changed.")
    return manifest


def require_output_target_binding(
    source_manifest: dict[str, Any],
    package_manifest: dict[str, Any],
) -> tuple[str, str]:
    """Fail closed unless every representation is the selected frozen target."""
    representations = {
        "runtime_manifest": source_manifest.get("output_target"),
        "query_package": package_manifest.get("output_target"),
    }
    validated = {
        name: validate_output_target(value)
        for name, value in representations.items()
    }
    if len(set(representations.values())) != 1:
        raise RuntimeError("GraphRAG output target representations conflict.")
    run_ids = {
        "runtime_manifest": source_manifest.get("run_id"),
        "query_package": package_manifest.get("run_id"),
    }
    if any(value != RUN_ID for value in run_ids.values()):
        raise RuntimeError("GraphRAG artifact run ID differs from RUN_ID.")
    if len(set(run_ids.values())) != 1:
        raise RuntimeError("GraphRAG artifact run ID representations conflict.")
    for field in ("scope", "phase", "case_count", "query_sha256", "sample"):
        if source_manifest.get(field) != package_manifest.get(field):
            raise RuntimeError(f"Runtime and query package {field} bindings conflict.")
    prefix, run_id = next(iter(validated.values()))
    if f"{prefix}/{run_id}" != OUTPUT_TARGET:
        raise RuntimeError("GraphRAG output target is not byte-identical to its run ID.")
    return prefix, run_id


def _server_ready(base_url: str) -> bool:
    try:
        with urlopen(f"{base_url}/models", timeout=5) as response:
            return response.status == 200
    except OSError:
        return False


def _wait_for_server(process: subprocess.Popen[str], url: str, log: Path) -> None:
    deadline = time.time() + 900
    while time.time() < deadline:
        if _server_ready(url):
            return
        if process.poll() is not None:
            tail = log.read_text(errors="replace")[-12_000:]
            raise RuntimeError(f"Model server exited with {process.returncode}\n{tail}")
        time.sleep(5)
    raise TimeoutError(f"Model server did not become ready; inspect {log}")


def _stop_server(process: subprocess.Popen[str]) -> None:
    """Stop one local model/boundary process without masking prior failures."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def _ensure_environment(project_root: Path, entrypoint: Path | None = None) -> None:
    installed_vllm = _installed_runtime_version("vllm")
    if installed_vllm != VLLM_VERSION:
        raise RuntimeError(
            "The job image must provide the pinned GPU runtime "
            f"vllm=={VLLM_VERSION}; found {installed_vllm!r}. "
            "Use vllm/vllm-openai:v0.18.1. The job deliberately does not "
            "download or replace vLLM at startup."
        )
    venv_root = Path("/tmp/graphrag-query-3.1.0-venv")
    subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(venv_root)],
        check=True,
    )
    subprocess.run(
        [
            str(venv_root / "bin/pip"),
            "install",
            "-r",
            str(project_root / "graphrag_runtime/requirements.graphrag.txt"),
        ],
        check=True,
    )
    os.execv(
        str(venv_root / "bin/python"),
        [
            str(venv_root / "bin/python"),
            str((entrypoint or Path(__file__)).resolve()),
            "--inside-graphrag-env",
        ],
    )


def _require_preinstalled_runtime() -> dict[str, str]:
    expected = {"graphrag": "3.1.0"}
    actual = {package: _installed_runtime_version(package) for package in expected}
    if actual != expected:
        raise RuntimeError(f"Preinstalled runtime does not match its lock: {actual}")
    return actual


def _installed_runtime_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        # Some prebuilt containers expose an importable vLLM module and CLI but
        # omit dist-info metadata. Accept the exact module version and keep the
        # lock fail-closed for any mismatch or missing version.
        try:
            module = importlib.import_module(package)
        except Exception:
            return None
        resolved = getattr(module, "__version__", None)
        return str(resolved) if resolved is not None else None


def require_a100_80gb() -> dict[str, Any]:
    line = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    rows = line.splitlines()
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one GPU, found {rows}")
    name, memory = [value.strip() for value in rows[0].rsplit(",", 1)]
    if "A100" not in name or int(memory) < 79_000:
        raise RuntimeError(f"Expected A100 80GB, found {line}")
    return {"name": name, "memory_mib": int(memory)}


def load_controlled_corpus(documents_path: Path):
    import pandas as pd
    from graphrag_runtime.corpus import ControlledPremise

    documents = pd.read_parquet(documents_path)
    records = []
    for raw in documents["raw_data"].tolist():
        item = dict(raw)
        records.append(ControlledPremise(
            id=str(item["id"]),
            title=str(item["title"]),
            text=str(item["text"]),
            graph_id=str(item["graph_id"]),
            category=str(item["category"]),
            node_id=str(item["node_id"]),
            diagnosis_label=str(item["diagnosis_label"]),
            premise_type=str(item["premise_type"]),
            diagnostic_path=tuple(str(v) for v in item["diagnostic_path"]),
            source_chunk_id=str(item["source_chunk_id"]),
            knowledge_source_ids=tuple(
                str(v) for v in item["knowledge_source_ids"]
            ),
            source_origin=str(item["source_origin"]),
        ))
    return tuple(records)


def _sanitized_case_output(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key != "saved_query_payload"
    } | {
        "saved_query_payload_sha256": canonical_sha256(
            record.get("saved_query_payload") or {}
        )
    }


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path.name != "terminal_manifest.json"
    }


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _require_not_writable(path: Path, label: str) -> None:
    """Fail closed unless the mounted immutable input rejects a write probe."""
    probe = path / ".query-runtime-write-probe"
    try:
        descriptor = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError:
        return
    else:
        os.close(descriptor)
        probe.unlink()
        raise PermissionError(f"{label} is writable; a read-only mount is required.")


def _require_writable_directory(path: Path, label: str) -> None:
    path.mkdir(parents=True, exist_ok=False)
    probe = path / ".write-probe"
    try:
        probe.write_text("writable\n", encoding="utf-8")
        if probe.read_text(encoding="utf-8") != "writable\n":
            raise RuntimeError(f"{label} write probe was not readable.")
    finally:
        probe.unlink(missing_ok=True)


def _require_secure_owned_directory(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise PermissionError(f"{label} must be a real directory, not a symlink.")
    metadata = path.stat()
    if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
        raise PermissionError(f"{label} is not owned by the current process user.")
    mode = stat.S_IMODE(metadata.st_mode)
    if mode & 0o077:
        raise PermissionError(
            f"{label} permissions are too broad: expected 0700-compatible, got {mode:04o}."
        )
    probe = path / ".ipc-write-probe"
    try:
        descriptor = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
    finally:
        probe.unlink(missing_ok=True)


def _ipc_path_bytes(base_path: Path, suffix: str = VLLM_UUID_WORST_CASE) -> int:
    return len(os.fsencode(str(Path(base_path).resolve() / suffix)))


def _validate_vllm_ipc_base(
    base_path: Path,
    *,
    zmq_limit_bytes: int = LINUX_UNIX_SOCKET_PATH_LIMIT_BYTES,
) -> dict[str, Any]:
    effective_limit = min(LINUX_UNIX_SOCKET_PATH_LIMIT_BYTES, int(zmq_limit_bytes))
    maximum_allowed = effective_limit - VLLM_IPC_SAFETY_MARGIN_BYTES
    worst_case_bytes = _ipc_path_bytes(base_path)
    if worst_case_bytes > maximum_allowed:
        raise ValueError(
            "vLLM IPC path is unsafe: "
            f"{worst_case_bytes} bytes exceeds {maximum_allowed} bytes "
            f"({effective_limit}-byte limit minus "
            f"{VLLM_IPC_SAFETY_MARGIN_BYTES}-byte safety margin)."
        )
    return {
        "configured_base_path": str(base_path),
        "resolved_base_path": str(Path(base_path).resolve()),
        "generated_suffix_bytes": len(VLLM_UUID_WORST_CASE.encode("ascii")),
        "worst_case_socket_path_bytes": worst_case_bytes,
        "effective_zmq_limit_bytes": effective_limit,
        "required_safety_margin_bytes": VLLM_IPC_SAFETY_MARGIN_BYTES,
        "actual_margin_bytes": effective_limit - worst_case_bytes,
    }


def prepare_vllm_ipc_namespaces(
    ipc_root: Path,
    *,
    run_id: str,
    approved_run_id: str | None = None,
    process_id: int | None = None,
    zmq_limit_bytes: int = LINUX_UNIX_SOCKET_PATH_LIMIT_BYTES,
    include_final_reasoner: bool = False,
) -> dict[str, Any]:
    """Create short private namespaces for vLLM's exact UUID IPC generator."""
    ipc_root = Path(ipc_root)
    if not ipc_root.is_absolute():
        raise ValueError("VLLM IPC root must be absolute.")
    resolved_tmp = Path("/tmp").resolve()
    resolved_root = ipc_root.resolve()
    if resolved_root.parent != resolved_tmp:
        raise ValueError("VLLM IPC root must be a direct child of /tmp.")
    if ipc_root.exists():
        _require_secure_owned_directory(ipc_root, "vLLM IPC root")
    else:
        ipc_root.mkdir(mode=0o700)
        _require_secure_owned_directory(ipc_root, "vLLM IPC root")

    pid = int(process_id if process_id is not None else os.getpid())
    if pid <= 0:
        raise ValueError("Process ID must be positive.")
    expected_run_id = approved_run_id or RUN_ID
    if run_id != expected_run_id:
        raise ValueError("vLLM IPC namespace run ID differs from the frozen run ID.")
    run_token = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:8]
    session = ipc_root / f"{run_token}-p{pid}"
    session.mkdir(mode=0o700, exist_ok=False)
    _require_secure_owned_directory(session, "vLLM IPC session")
    bases = {
        "embedding": session / "bge",
        "chat": session / "med",
    }
    if include_final_reasoner:
        bases["final_reasoner"] = session / "qwn"
    audits: dict[str, dict[str, Any]] = {}
    try:
        for server, base in bases.items():
            base.mkdir(mode=0o700, exist_ok=False)
            _require_secure_owned_directory(base, f"vLLM {server} IPC namespace")
            audits[server] = _validate_vllm_ipc_base(
                base,
                zmq_limit_bytes=zmq_limit_bytes,
            )
        if len({str(path.resolve()) for path in bases.values()}) != len(bases):
            raise RuntimeError("vLLM server IPC namespaces are not unique.")
    except Exception:
        shutil.rmtree(session, ignore_errors=True)
        try:
            ipc_root.rmdir()
        except OSError:
            pass
        raise
    return {
        "root": ipc_root,
        "session": session,
        "bases": bases,
        "audit": audits,
        "run_token": run_token,
        "process_id": pid,
    }


def cleanup_vllm_ipc_namespaces(paths: dict[str, Any]) -> dict[str, Any]:
    session = Path(paths["session"])
    root = Path(paths["root"])
    shutil.rmtree(session)
    if session.exists():
        raise RuntimeError("vLLM IPC session was not removed.")
    root_removed = False
    try:
        root.rmdir()
        root_removed = True
    except OSError:
        # Another collision-safe session may legitimately share the private root.
        pass
    return {
        "session_removed": True,
        "shared_root_removed": root_removed,
    }


def vllm_server_environment(base_path: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment[VLLM_RPC_BASE_PATH_ENV] = str(Path(base_path).resolve())
    return environment


def verify_pinned_vllm_ipc_control(
    vllm_executable: Path,
    bases: dict[str, Path],
) -> dict[str, Any]:
    """Invoke vLLM's installed 0.18.1 path function under each child env."""
    vllm_executable = Path(vllm_executable).resolve()
    if not vllm_executable.is_file():
        raise FileNotFoundError("vLLM executable was not found.")
    # The official vLLM image exposes python3 rather than a sibling executable
    # literally named `python`.  The active GraphRAG venv was created with
    # --system-site-packages, so its interpreter is the authoritative runtime
    # to audit: it must itself import and report the pinned vLLM version.
    python_executable = Path(sys.executable).resolve()
    if not python_executable.is_file():
        raise FileNotFoundError("Active Python interpreter was not found.")
    script = (
        "import json, zmq; "
        "from importlib.metadata import version; "
        "from vllm import envs; "
        "from vllm.utils.network_utils import get_open_zmq_ipc_path; "
        "print(json.dumps({'version': version('vllm'), "
        "'base': envs.VLLM_RPC_BASE_PATH, "
        "'generated': get_open_zmq_ipc_path(), "
        "'zmq_limit': getattr(zmq, 'IPC_PATH_MAX_LEN', 107)}))"
    )
    result: dict[str, Any] = {}
    for server, base in bases.items():
        completed = subprocess.run(
            [str(python_executable), "-c", script],
            env=vllm_server_environment(base),
            check=True,
            capture_output=True,
            text=True,
        )
        record = json.loads(completed.stdout.strip().splitlines()[-1])
        if record["version"] != VLLM_VERSION:
            raise RuntimeError("Installed vLLM version differs from the pinned version.")
        if Path(record["base"]).resolve() != base.resolve():
            raise RuntimeError("vLLM ignored the explicit VLLM_RPC_BASE_PATH.")
        prefix = f"ipc://{base.resolve()}/"
        if not str(record["generated"]).startswith(prefix):
            raise RuntimeError("vLLM generated an IPC path outside the approved namespace.")
        suffix = str(record["generated"])[len(prefix):]
        if len(suffix.encode("ascii")) != len(VLLM_UUID_WORST_CASE):
            raise RuntimeError("vLLM generated an unexpected IPC suffix shape.")
        path_audit = _validate_vllm_ipc_base(
            base,
            zmq_limit_bytes=int(record["zmq_limit"]),
        )
        result[server] = {
            **path_audit,
            "generated_path_sha256": hashlib.sha256(
                str(record["generated"]).encode("utf-8")
            ).hexdigest(),
            "vllm_version": record["version"],
            "control": VLLM_RPC_BASE_PATH_ENV,
        }
    return result


def prepare_runtime_workspace(
    *,
    query_root: Path,
    scratch_root: Path,
    chat_url: str = CHAT_URL,
    embedding_url: str = EMBEDDING_URL,
) -> dict[str, Path]:
    """Create a query-only GraphRAG workspace with all mutable paths in scratch."""
    query_root = query_root.resolve()
    scratch_root = scratch_root.resolve()
    if _is_relative_to(scratch_root, query_root) or _is_relative_to(query_root, scratch_root):
        raise RuntimeError("Runtime scratch and immutable query package must be disjoint.")
    _require_not_writable(query_root, "Query package")
    _require_not_writable(query_root / "workspace/output/lancedb", "Vector store")
    _require_writable_directory(scratch_root, "Runtime scratch")

    paths = {
        "root": scratch_root,
        "workspace": scratch_root / "graphrag-workspace",
        "logs": scratch_root / "logs",
        "reporting": scratch_root / "graphrag-reporting",
        "cache": scratch_root / "graphrag-cache",
        "temp": scratch_root / "temp",
        "input": scratch_root / "graphrag-input",
        "output": scratch_root / "graphrag-output",
        "update_output": scratch_root / "graphrag-update-output",
        "model_cache": scratch_root / "model-cache",
    }
    for name, path in paths.items():
        if name != "root":
            path.mkdir(parents=True, exist_ok=False)

    source_settings = json.loads(
        (query_root / "workspace/settings.json").read_text(encoding="utf-8")
    )
    source_settings["cache"]["storage"]["base_dir"] = str(paths["cache"])
    source_settings["reporting"]["base_dir"] = str(paths["reporting"])
    source_settings["input"]["storage"]["base_dir"] = str(paths["input"])
    source_settings["output"]["base_dir"] = str(paths["output"])
    source_settings["input_storage"] = {
        "type": "file",
        "base_dir": str(paths["input"]),
    }
    source_settings["output_storage"] = {
        "type": "file",
        "base_dir": str(paths["output"]),
    }
    source_settings["update_output_storage"] = {
        "type": "file",
        "base_dir": str(paths["update_output"]),
    }
    source_settings["vector_store"]["db_uri"] = str(
        query_root / "workspace/output/lancedb"
    )
    source_settings["completion_models"]["local_chat"]["api_base"] = chat_url
    source_settings["completion_models"]["local_chat"]["model"] = CHAT_MODEL_ID
    source_settings["completion_models"]["local_chat"]["call_args"] = {
        "max_tokens": COMPLETION_TOKENS,
    }
    source_settings["completion_models"]["local_chat"]["retry"] = None
    source_settings["embedding_models"]["local_embedding"]["api_base"] = embedding_url
    source_settings.setdefault("local_search", {})["max_context_tokens"] = (
        GRAPHRAG_CONTEXT_TOKENS
    )
    (paths["workspace"] / "settings.json").write_text(
        json.dumps(source_settings, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.environ.update({
        "TMPDIR": str(paths["temp"]),
        "TMP": str(paths["temp"]),
        "TEMP": str(paths["temp"]),
    })
    return paths


def cleanup_clinical_runtime_scratch(
    paths: dict[str, Path],
    *,
    run_root: Path,
) -> dict[str, Any]:
    """Remove generated GraphRAG/model scratch that may contain licensed text."""
    removable = (
        "workspace",
        "reporting",
        "cache",
        "temp",
        "input",
        "output",
        "update_output",
        "model_cache",
    )
    removed: list[str] = []
    for name in removable:
        path = Path(paths[name])
        if not _is_relative_to(path, run_root):
            raise RuntimeError(f"Refusing to clean scratch outside run root: {name}")
        if path.exists():
            shutil.rmtree(path)
        if path.exists():
            raise RuntimeError(f"Licensed runtime scratch was not removed: {name}")
        removed.append(name)
    return {
        "status": "passed",
        "removed_components": removed,
        "raw_clinical_runtime_scratch_published": False,
    }


def exception_record(
    exc: BaseException,
    *,
    stage: str,
    case_id: str | None = None,
) -> dict[str, Any]:
    """Capture the first exception without traceback locals or clinical payloads."""
    rendered = "".join(
        traceback.TracebackException.from_exception(
            exc,
            capture_locals=False,
        ).format()
    )
    return {
        "stage": stage,
        "case_id": case_id,
        "error_type": type(exc).__name__,
        "message": str(exc),
        "traceback": rendered,
        "traceback_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
    }


def audit_live_query_boundary(path: Path, audit_function) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        return {
            "status": "not_invoked",
            "proxy_invoked": False,
            "cause": "No live-query embedding request reached the boundary.",
        }
    return {"status": "validated", "proxy_invoked": True, **audit_function(path)}


def audit_medgemma_prompt_boundary(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        return {
            "status": "not_invoked",
            "proxy_invoked": False,
            "cause": "No completion request reached the MedGemma prompt boundary.",
        }
    records = _load_jsonl(path)
    required = {
        "rendered_input_tokens",
        "requested_completion_tokens",
        "safety_margin_tokens",
        "max_model_len",
        "messages_sha256",
        "rendered_token_ids_sha256",
        "structured_output_contract_id",
        "structured_output_injected",
        "response_schema_sha256",
        "candidate_choice_count",
        "candidate_source_pair_count",
    }
    if not records or any(required - set(record) for record in records):
        raise RuntimeError("MedGemma prompt audit is incomplete.")
    if any(
        int(record["rendered_input_tokens"])
        + int(record["requested_completion_tokens"])
        + int(record["safety_margin_tokens"])
        > int(record["max_model_len"])
        for record in records
    ):
        raise RuntimeError("An over-budget prompt passed the MedGemma boundary.")
    from graphrag_runtime.audit import GRAPHRAG_CANDIDATE_CHOICE_CONTRACT_ID

    if any(
        record["structured_output_injected"] is not True
        or record["structured_output_contract_id"]
        != GRAPHRAG_CANDIDATE_CHOICE_CONTRACT_ID
        or not record["response_schema_sha256"]
        or int(record["candidate_choice_count"]) < 1
        or int(record["candidate_source_pair_count"])
        < int(record["candidate_choice_count"])
        for record in records
    ):
        raise RuntimeError(
            "A GraphRAG request bypassed the structured-output boundary."
        )
    return {
        "status": "validated",
        "proxy_invoked": True,
        "request_count": len(records),
        "maximum_rendered_input_tokens": max(
            int(record["rendered_input_tokens"]) for record in records
        ),
        "audit_log_sha256": file_sha256(path),
        "raw_prompt_persisted": False,
        "structured_output_request_count": len(records),
        "structured_output_contract_id": GRAPHRAG_CANDIDATE_CHOICE_CONTRACT_ID,
        "unique_response_schema_count": len({
            str(record["response_schema_sha256"]) for record in records
        }),
        "response_schema_sequence_sha256": canonical_sha256([
            str(record["response_schema_sha256"]) for record in records
        ]),
        "minimum_candidate_choice_count": min(
            int(record["candidate_choice_count"]) for record in records
        ),
        "maximum_candidate_choice_count": max(
            int(record["candidate_choice_count"]) for record in records
        ),
    }


def audit_all_completion_prompts(path: Path, *, boundary_name: str,
                                 allow_not_invoked: bool = False) -> dict[str, Any]:
    """Validate every generation request, including post-retrieval agent calls."""
    if not path.is_file() or path.stat().st_size == 0:
        if allow_not_invoked:
            return {"status": "not_invoked", "proxy_invoked": False,
                    "cause": "Every judgment was resolved deterministically."}
        raise RuntimeError(f"No request reached the {boundary_name} prompt boundary.")
    records = _load_jsonl(path)
    required = {
        "rendered_input_tokens", "requested_completion_tokens",
        "safety_margin_tokens", "max_model_len", "messages_sha256",
        "rendered_token_ids_sha256", "client_structured_output_forwarded",
        "structured_output_injected", "client_unconstrained_contract_id",
    }
    if not records or any(required - set(record) for record in records):
        raise RuntimeError(f"{boundary_name} prompt audit is incomplete.")
    if any(
        int(record["rendered_input_tokens"])
        + int(record["requested_completion_tokens"])
        + int(record["safety_margin_tokens"])
        > int(record["max_model_len"])
        for record in records
    ):
        raise RuntimeError(f"An over-budget prompt passed the {boundary_name} boundary.")
    if any(
        not (
            record["client_structured_output_forwarded"] is True
            or record["structured_output_injected"] is True
        )
        for record in records
    ):
        raise RuntimeError(f"An unconstrained request passed the {boundary_name} boundary.")
    return {
        "status": "validated", "proxy_invoked": True,
        "request_count": len(records),
        "maximum_rendered_input_tokens": max(
            int(record["rendered_input_tokens"]) for record in records
        ),
        "maximum_reserved_tokens": max(
            int(record["rendered_input_tokens"])
            + int(record["requested_completion_tokens"])
            + int(record["safety_margin_tokens"])
            for record in records
        ),
        "audit_log_sha256": file_sha256(path),
        "raw_prompt_persisted": False,
        "allowlisted_unconstrained_request_count": 0,
    }


def _run_evaluation_resume(
    *, query_root: Path, output_root: Path, output_prefix: str, run_id: str,
    package_manifest: dict[str, Any],
) -> int:
    """Run only the blinded Qwen judge over a sealed prior comparison."""
    from clinical_cds.clinical_acceptability_judge import (
        JUDGE_ID, summarize_clinical_judgments, write_clinical_judgments,
    )
    from clinical_cds.model import OpenAICompatibleDiagnosticModel
    from clinical_cds.schema import ClinicalCase, DiagnosticGraph, PredictionRecord

    resume = package_manifest.get("evaluation_resume")
    resume_root = query_root / "resume/comparison"
    if not isinstance(resume, dict):
        raise RuntimeError("Evaluation phase requires a frozen comparison resume.")
    resume_files = _directory_manifest(resume_root)
    if resume_files != resume.get("files") or canonical_sha256(
        resume_files
    ) != resume.get("package_sha256"):
        raise RuntimeError("Frozen comparison resume artifacts changed.")
    queries = _load_jsonl(query_root / "queries/validation_queries.jsonl")
    expected_ids = tuple(str(value) for value in package_manifest["case_ids"])
    cases = tuple(ClinicalCase(
        case_id=str(row["case_id"]), dataset="direct", task="diagnosis",
        sections={str(item["section"]): str(item["text"])
                  for item in row["saved_query_payload"]["patient_evidence"]},
        gold_label=str(row["evaluation_gold_label"]),
        disease_category=str(row.get("evaluation_gold_category") or ""),
        metadata={"patient_role": "query_only", "knowledge_corpus_membership": False},
    ) for row in queries)
    records = tuple(PredictionRecord.from_dict(row) for row in
                    _load_jsonl(resume_root / "predictions.jsonl"))
    expected_modes = {
        "direct", "flat_rag", "graph_rag", "evidence_grounded_argumentation",
    }
    if tuple(case.case_id for case in cases) != expected_ids or any(
        {record.mode.value for record in records if record.case_id == case_id}
        != expected_modes for case_id in expected_ids
    ) or len(records) != len(expected_ids) * 4:
        raise RuntimeError("Frozen predictions do not bind four methods to every case.")
    corpus = load_controlled_corpus(query_root / "workspace/output/documents.parquet")
    paths: dict[str, list[tuple[str, ...]]] = {}
    categories: dict[str, str] = {}
    for premise in corpus:
        categories[premise.graph_id] = premise.category
        values = paths.setdefault(premise.graph_id, [])
        if premise.diagnostic_path not in values:
            values.append(premise.diagnostic_path)
    graphs = tuple(DiagnosticGraph(
        graph_id=graph_id, category=categories[graph_id], nodes=(), edges=(),
        diagnostic_paths={f"path:{index}": value
                          for index, value in enumerate(paths[graph_id])},
    ) for graph_id in sorted(paths))

    if os.environ.get("RUNTIME_SCRATCH_ROOT") != RUNTIME_SCRATCH_ROOT:
        raise ValueError("Evaluation scratch root differs from the frozen target.")
    run_root = Path(RUNTIME_SCRATCH_ROOT)
    publish_root = output_root / output_prefix / run_id
    _clear_existing_output_roots(run_root, publish_root)
    runtime_paths = prepare_runtime_workspace(query_root=query_root, scratch_root=run_root)
    logs = runtime_paths["logs"]
    audit_path = logs / "qwen-prompt-boundary-audit.jsonl"
    original_input = _directory_manifest(query_root)
    started = datetime.now(timezone.utc)
    gpu = require_a100_80gb()
    vllm = shutil.which("vllm")
    if not vllm:
        raise FileNotFoundError("vLLM executable is unavailable.")
    servers: list[subprocess.Popen[str]] = []
    ipc_paths: dict[str, Any] | None = None
    failure: BaseException | None = None
    try:
        os.environ["VLLM_STRUCTURED_OUTPUTS_CONFIG"] = STRUCTURED_OUTPUTS_CONFIG
        ipc_root = os.environ.get(VLLM_IPC_ROOT_ENV)
        if not ipc_root:
            raise EnvironmentError(f"{VLLM_IPC_ROOT_ENV} must be set explicitly.")
        ipc_paths = prepare_vllm_ipc_namespaces(
            Path(ipc_root), run_id=run_id, include_final_reasoner=True,
        )
        with ((logs / "reasoner-vllm.log").open("w") as model_log,
              (logs / "reasoner-boundary.log").open("w") as boundary_log):
            server = subprocess.Popen(completion_server_command(
                vllm, model_id=JUDGE_MODEL_ID, revision=JUDGE_MODEL_REVISION,
                port=8004, max_model_len=MAX_MODEL_LEN,
                gpu_memory_utilization=0.80, dtype="auto",
            ), stdout=model_log, stderr=subprocess.STDOUT, text=True,
                env=vllm_server_environment(ipc_paths["bases"]["final_reasoner"]))
            servers.append(server)
            _wait_for_server(server, JUDGE_UPSTREAM_URL, logs / "reasoner-vllm.log")
            snapshot = Path(__import__("huggingface_hub").snapshot_download(
                repo_id=JUDGE_MODEL_ID, revision=JUDGE_MODEL_REVISION,
                allow_patterns=["added_tokens.json", "chat_template.jinja",
                                "special_tokens_map.json", "tokenizer.json",
                                "tokenizer_config.json"],
            ))
            boundary = subprocess.Popen([
                sys.executable, "-m", "graphrag_runtime.medgemma_chat_boundary",
                "--upstream-url", JUDGE_UPSTREAM_URL, "--port", "8005",
                "--audit-log", str(audit_path), "--model-source", str(snapshot),
                "--revision", JUDGE_MODEL_REVISION, "--expected-model-id", JUDGE_MODEL_ID,
                "--prompt-budget-policy", "exact-qwen-chat-template-v1", "--local-files-only",
            ], stdout=boundary_log, stderr=subprocess.STDOUT, text=True)
            servers.append(boundary)
            _wait_for_server(boundary, JUDGE_URL, logs / "reasoner-boundary.log")
            model = OpenAICompatibleDiagnosticModel(
                model_name=JUDGE_MODEL_ID, model_revision=JUDGE_MODEL_REVISION,
                base_url=JUDGE_URL, timeout_seconds=600,
                context_window=MAX_MODEL_LEN,
                max_output_tokens=ARGUMENT_GENERATOR_COMPLETION_TOKENS,
            )
            results = write_clinical_judgments(
                records=records, cases=cases, graphs=graphs, model=model,
                output_path=run_root / "clinical_family_judgments.jsonl",
            )
            _write_json_new(run_root / "clinical_family_summary.json", {
                "judge_id": JUDGE_ID, "model_id": JUDGE_MODEL_ID,
                "evaluation_rule": "relationship-derived-family-pass-fail",
                "abstention_policy": (
                    "exclude_verified_insufficient_evidence;fail_execution_and_other"
                ),
                "modes": list(summarize_clinical_judgments(records, results)),
            })
            qwen_audit = audit_all_completion_prompts(
                audit_path, boundary_name="Qwen", allow_not_invoked=True,
            )
            _write_json_new(run_root / "evaluation_resume_audit.json", {
                "run_id": run_id, "phase": "evaluation",
                "source_resume_sha256": resume["package_sha256"],
                "prediction_count": len(records), "case_count": len(cases),
                "predictions_sha256": file_sha256(resume_root / "predictions.jsonl"),
                "argument_traces_sha256": file_sha256(resume_root / "argument_traces.jsonl"),
                "qwen_prompt_boundary": qwen_audit, "gpu": gpu,
                "medgemma_invocation_count": 0,
            })
    except Exception as exc:
        failure = exc
        _write_json_new(run_root / "terminal_failure.json", {
            "status": "error", "run_id": run_id,
            "first_exception": exception_record(exc, stage="evaluation_resume"),
        })
        raise
    finally:
        for server in reversed(servers):
            _stop_server(server)
        if ipc_paths is not None:
            cleanup_vllm_ipc_namespaces(ipc_paths)
        unchanged = _directory_manifest(query_root) == original_input
        _write_json_new(run_root / "terminal_manifest.json", {
            "run_id": run_id,
            "status": "error" if failure or not unchanged else "completed",
            "started_at": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "phase": "evaluation", "input_package_unchanged": unchanged,
            "medgemma_invocation_count": 0,
            "artifact_sha256": _artifact_hashes(run_root),
        })
        _clear_existing_output_roots(publish_root)
        publish_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(run_root, publish_root)
    return 0


def main() -> int:
    args = parse_args()
    project_root = Path(os.environ.get("PROJECT_ROOT", "/workspace/project"))
    query_root = Path(os.environ.get("QUERY_PACKAGE_ROOT", "/workspace/query"))
    output_root = Path(os.environ.get("OUTPUT_ROOT", "/outputs"))
    source_manifest = require_source_package(project_root)
    package_manifest = require_package(query_root)
    output_prefix, run_id = require_output_target_binding(
        source_manifest,
        package_manifest,
    )
    if not args.inside_graphrag_env:
        _ensure_environment(project_root)
        return 0
    if os.environ.get("HF_RUNTIME_PREINSTALLED") == "yes":
        _require_preinstalled_runtime()
    elif version("graphrag") != "3.1.0":
        raise RuntimeError("Isolated GraphRAG version is not 3.1.0.")

    sys.path.insert(0, str(project_root))
    from clinical_cds.method_contract import CURRENT_ARGUMENT_METHOD
    from clinical_cds.model import OpenAICompatibleDiagnosticModel
    from clinical_cds.normalization import UMLSNormalizer
    from clinical_cds.runner import ExperimentRunner
    from clinical_cds.schema import ClinicalCase, DiagnosticGraph
    from clinical_cds.experiment import run_experiment, protocol_sha256
    from clinical_cds.clinical_acceptability_judge import (
        JUDGE_ID,
        summarize_clinical_judgments,
        write_clinical_judgments,
    )
    from clinical_cds.validation import audit_validation
    from graphrag_runtime.audit import (
        audit_five_case_retrieval,
        audit_retrieval_checkpoint,
        bind_candidate_choices_strict,
    )
    from graphrag_runtime.embedding_boundary import audit_embedding_boundary
    from graphrag_runtime.retrieval import (
        DensePremiseIndex,
        FixedGraphRagKnowledgeRetriever,
        flat_seeded_graph_candidate_choices,
        IndependentFlatPremiseRetriever,
        select_diverse_candidate_routes,
    )

    if protocol_sha256() != source_manifest["protocol_sha256"]:
        raise RuntimeError("Imported protocol differs from the staged source.")
    if RUN_PHASE == "evaluation":
        return _run_evaluation_resume(
            query_root=query_root, output_root=output_root,
            output_prefix=output_prefix, run_id=run_id,
            package_manifest=package_manifest,
        )

    argument_generation_preflight = validate_argument_generation_contract()

    immutable_workspace = query_root / "workspace"
    queries_path = query_root / "queries/validation_queries.jsonl"
    output_tables = immutable_workspace / "output"
    vector_root = output_tables / "lancedb"
    if tree_sha256(vector_root) != EXPECTED_VECTOR_TREE_SHA256:
        raise RuntimeError("Read-only recovered vector store changed.")
    if tuple(sorted(path.name for path in output_tables.glob("*.parquet"))) != REQUIRED_TABLES:
        raise RuntimeError("Query-only table allowlist changed.")
    initial_package_files = _directory_manifest(query_root)
    umls_manifest = package_manifest.get("umls") or {}
    umls_relative = str(umls_manifest.get("database_relative") or "")
    umls_db = query_root / umls_relative
    if (
        umls_relative != "terminology/umls_study_subset.sqlite3"
        or not umls_db.is_file()
        or file_sha256(umls_db) != umls_manifest.get("database_sha256")
    ):
        raise RuntimeError("The frozen UMLS 2026AA study subset is missing or changed.")
    corpus = load_controlled_corpus(output_tables / "documents.parquet")
    queries = _load_jsonl(queries_path)
    expected_case_ids = tuple(str(value) for value in package_manifest.get("case_ids") or ())
    if (
        tuple(str(row.get("case_id")) for row in queries) != expected_case_ids
        or len(expected_case_ids) != int(package_manifest.get("case_count") or 0)
        or (
            package_manifest.get("sample") is None
            and len(expected_case_ids) != int(RUN_CONFIG["case_count"])
        )
    ):
        raise RuntimeError("Frozen query scope or case order changed.")

    configured_scratch = os.environ.get("RUNTIME_SCRATCH_ROOT")
    if not configured_scratch:
        raise EnvironmentError("RUNTIME_SCRATCH_ROOT must be set explicitly.")
    if configured_scratch != RUNTIME_SCRATCH_ROOT:
        raise ValueError(
            "Runtime output directory differs from the frozen GraphRAG target."
        )
    run_root = Path(configured_scratch)
    publish_root = output_root / output_prefix / run_id
    _clear_existing_output_roots(run_root, publish_root)
    runtime_paths = prepare_runtime_workspace(
        query_root=query_root,
        scratch_root=run_root,
    )
    _write_json_new(
        run_root / "argument_generation_preflight.json",
        argument_generation_preflight,
    )
    umls_normalizer = UMLSNormalizer.from_path(
        umls_db,
        lookup_cache_db_path=(
            runtime_paths["cache"] / "umls_lookup_cache.sqlite3"
        ),
    )
    runtime_workspace = runtime_paths["workspace"]
    logs = runtime_paths["logs"]
    boundary_audit = logs / "query-embedding-boundary-audit.jsonl"
    prompt_audit = logs / "medgemma-prompt-boundary-audit.jsonl"
    judge_prompt_audit = logs / "qwen-prompt-boundary-audit.jsonl"
    started = datetime.now(timezone.utc)
    gpu = require_a100_80gb()
    vllm = shutil.which("vllm")
    if not vllm:
        raise FileNotFoundError("vLLM executable is unavailable.")

    servers: list[subprocess.Popen[str]] = []
    ipc_paths: dict[str, Any] | None = None
    primary_exception: BaseException | None = None
    dense_premise_index_build_invocations = 0
    try:
        os.environ["VLLM_STRUCTURED_OUTPUTS_CONFIG"] = STRUCTURED_OUTPUTS_CONFIG
        configured_ipc_root = os.environ.get(VLLM_IPC_ROOT_ENV)
        if not configured_ipc_root:
            raise EnvironmentError(f"{VLLM_IPC_ROOT_ENV} must be set explicitly.")
        ipc_paths = prepare_vllm_ipc_namespaces(
            Path(configured_ipc_root),
            run_id=run_id,
            include_final_reasoner=True,
        )
        ipc_control = verify_pinned_vllm_ipc_control(
            Path(vllm),
            ipc_paths["bases"],
        )
        _write_json_new(run_root / "vllm_ipc_preflight.json", {
            "status": "passed",
            "vllm_version": VLLM_VERSION,
            "control": VLLM_RPC_BASE_PATH_ENV,
            "source_sha256": PINNED_VLLM_SOURCE_SHA256,
            "session_path": str(ipc_paths["session"]),
            "server_namespaces": ipc_control,
        })
        with (
            (logs / "chat-vllm.log").open("w") as chat_log,
            (logs / "chat-boundary.log").open("w") as chat_boundary_log,
            (logs / "reasoner-vllm.log").open("w") as reasoner_log,
            (logs / "reasoner-boundary.log").open("w") as reasoner_boundary_log,
            (logs / "embedding-vllm.log").open("w") as embedding_log,
            (logs / "embedding-boundary.log").open("w") as boundary_log,
        ):
            embedding = subprocess.Popen(
                [
                    vllm, "serve", EMBEDDING_MODEL_ID,
                    "--revision", EMBEDDING_MODEL_REVISION,
                    "--host", "127.0.0.1", "--port", "8002",
                    "--runner", "pooling", "--convert", "embed",
                    "--gpu-memory-utilization", "0.10",
                ],
                stdout=embedding_log,
                stderr=subprocess.STDOUT,
                text=True,
                env=vllm_server_environment(ipc_paths["bases"]["embedding"]),
            )
            servers.append(embedding)
            _wait_for_server(embedding, EMBEDDING_UPSTREAM_URL, logs / "embedding-vllm.log")
            boundary = subprocess.Popen(
                [
                    sys.executable, "-m", "graphrag_runtime.embedding_boundary",
                    "--upstream-url", EMBEDDING_UPSTREAM_URL,
                    "--port", "8001", "--audit-log", str(boundary_audit),
                    "--model-id", EMBEDDING_MODEL_ID,
                    "--tokenizer-json",
                    str(
                        Path(
                            __import__("huggingface_hub").snapshot_download(
                                repo_id=EMBEDDING_MODEL_ID,
                                revision=EMBEDDING_MODEL_REVISION,
                                allow_patterns=["tokenizer.json"],
                                local_files_only=True,
                            )
                        )
                        / "tokenizer.json"
                    ),
                ],
                stdout=boundary_log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            servers.append(boundary)
            _wait_for_server(boundary, EMBEDDING_URL, logs / "embedding-boundary.log")
            dense_premise_index = DensePremiseIndex.from_openai_endpoint(
                corpus,
                api_base=EMBEDDING_URL,
                model=EMBEDDING_MODEL_ID,
            )
            dense_premise_index_build_invocations += 1
            reasoner_command = completion_server_command(
                vllm, model_id=JUDGE_MODEL_ID,
                revision=JUDGE_MODEL_REVISION, port=8004,
                max_model_len=MAX_MODEL_LEN,
                gpu_memory_utilization=0.55, dtype="auto",
            )
            reasoner = subprocess.Popen(
                reasoner_command, stdout=reasoner_log,
                stderr=subprocess.STDOUT, text=True,
                env=vllm_server_environment(ipc_paths["bases"]["final_reasoner"]),
            )
            servers.append(reasoner)
            _wait_for_server(reasoner, JUDGE_UPSTREAM_URL, logs / "reasoner-vllm.log")
            reasoner_snapshot = Path(
                __import__("huggingface_hub").snapshot_download(
                    repo_id=JUDGE_MODEL_ID,
                    revision=JUDGE_MODEL_REVISION,
                    allow_patterns=[
                        "added_tokens.json", "chat_template.jinja",
                        "special_tokens_map.json", "tokenizer.json",
                        "tokenizer_config.json",
                    ],
                )
            )
            reasoner_boundary = subprocess.Popen(
                [
                    sys.executable, "-m", "graphrag_runtime.medgemma_chat_boundary",
                    "--upstream-url", JUDGE_UPSTREAM_URL,
                    "--port", "8005", "--audit-log", str(judge_prompt_audit),
                    "--model-source", str(reasoner_snapshot),
                    "--revision", JUDGE_MODEL_REVISION,
                    "--expected-model-id", JUDGE_MODEL_ID,
                    "--prompt-budget-policy", "exact-qwen-chat-template-v1",
                    "--local-files-only",
                ],
                stdout=reasoner_boundary_log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            servers.append(reasoner_boundary)
            _wait_for_server(reasoner_boundary, JUDGE_URL, logs / "reasoner-boundary.log")
            medgemma_snapshot = Path(
                __import__("huggingface_hub").snapshot_download(
                    repo_id=CHAT_MODEL_ID,
                    revision=CHAT_MODEL_REVISION,
                    allow_patterns=[
                        "added_tokens.json",
                        "chat_template.jinja",
                        "special_tokens_map.json",
                        "tokenizer.json",
                        "tokenizer_config.json",
                    ],
                )
            )
            chat_command = completion_server_command(
                vllm,
                model_id=CHAT_MODEL_ID,
                revision=CHAT_MODEL_REVISION,
                port=8003,
                max_model_len=MAX_MODEL_LEN,
                gpu_memory_utilization=0.20,
            )
            chat = subprocess.Popen(
                chat_command,
                stdout=chat_log,
                stderr=subprocess.STDOUT,
                text=True,
                env=vllm_server_environment(ipc_paths["bases"]["chat"]),
            )
            servers.append(chat)
            _wait_for_server(chat, CHAT_UPSTREAM_URL, logs / "chat-vllm.log")
            chat_boundary = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "graphrag_runtime.medgemma_chat_boundary",
                    "--upstream-url",
                    CHAT_UPSTREAM_URL,
                    "--port",
                    "8000",
                    "--audit-log",
                    str(prompt_audit),
                    "--model-source",
                    str(medgemma_snapshot),
                    "--revision",
                    CHAT_MODEL_REVISION,
                    "--local-files-only",
                ],
                stdout=chat_boundary_log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            servers.append(chat_boundary)
            _wait_for_server(chat_boundary, CHAT_URL, logs / "chat-boundary.log")

            if RUN_PHASE == "mapping_diagnostic":
                from hf_job.mapping_diagnostic import run_mapping_diagnostic

                diagnostic_model = OpenAICompatibleDiagnosticModel(
                    model_name=CHAT_MODEL_ID,
                    model_revision=CHAT_MODEL_REVISION,
                    base_url=CHAT_URL,
                    timeout_seconds=600,
                    context_window=MAX_MODEL_LEN,
                    max_output_tokens=128,
                )
                _write_json_new(
                    run_root / "mapping_diagnostic.json",
                    run_mapping_diagnostic(diagnostic_model),
                )
                return 0

            case_outputs: list[dict[str, Any]] = []
            case_failures: list[dict[str, Any]] = []
            corpus_by_source = {
                record.source_chunk_id: record for record in corpus
            }
            for query_record in queries:
                payload = query_record["saved_query_payload"]
                citation_allowlist: tuple[str, ...] = ()
                candidate_choices: tuple[
                    tuple[str, str, tuple[str, ...]], ...
                ] = ()
                retrieved_candidate_labels: list[str] = []
                retrieved_candidate_graph_ids: list[str] = []
                try:
                    structured_query_text = patient_concept_retrieval_query(payload)
                    query_text = structured_query_text
                    umls_expansions = umls_normalizer.expand_text(query_text)
                    if umls_expansions:
                        query_text = " ".join((query_text, *umls_expansions))
                    dense_query_vectors = dense_premise_index.embed_queries(
                        patient_finding_retrieval_queries(payload),
                        api_base=EMBEDDING_URL,
                        model=EMBEDDING_MODEL_ID,
                    )
                    dense_neighbors = dense_premise_index.nearest(
                        dense_query_vectors,
                    )
                    candidate_choices = flat_seeded_graph_candidate_choices(
                        query_text,
                        corpus,
                        dense_neighbors=dense_neighbors,
                        normalizer=umls_normalizer,
                    )
                    citation_allowlist = tuple(dict.fromkeys(
                        source_id
                        for _, _, source_ids in candidate_choices
                        for source_id in source_ids
                    ))
                    retrieved_candidate_labels = [
                        label for _, label, _ in candidate_choices
                    ]
                    retrieved_candidate_graph_ids = [
                        corpus_by_source[source_ids[0]].graph_id
                        for _, _, source_ids in candidate_choices
                    ]
                    patient_ids = tuple(
                        str(item["evidence_id"])
                        for item in payload["patient_evidence"]
                    )
                    response_sha256 = provenance_sha256([
                        [candidate_id, label, list(source_ids)]
                        for candidate_id, label, source_ids in candidate_choices
                    ])
                    selected_routes = select_diverse_candidate_routes(
                        candidate_choices,
                        corpus,
                        query_text=structured_query_text,
                        normalizer=umls_normalizer,
                    )
                    selected_graph_ids = {
                        str(route["family_key"]).removeprefix("graph:")
                        for route in selected_routes
                    }
                    citation_allowlist = tuple(dict.fromkeys((
                        *citation_allowlist,
                        *(
                            record.source_chunk_id
                            for record in corpus
                            if record.graph_id in selected_graph_ids
                        ),
                    )))
                    deterministic_selection = {
                        "ranked_candidates": [
                            {
                                "rank": route["rank"],
                                "candidate_id": route["candidate_id"],
                                "source_chunk_ids": route["source_chunk_ids"],
                            }
                            for route in selected_routes
                        ],
                        "abstain": not bool(selected_routes),
                    }
                    bindings = bind_candidate_choices_strict(
                        deterministic_selection,
                        corpus,
                        patient_ids,
                        candidate_choices=candidate_choices,
                    )
                    abstain = not bool(bindings)
                    output = {
                        **query_record,
                        "ranked_candidates": list(bindings),
                        "retrieved_candidate_labels": retrieved_candidate_labels,
                        "retrieved_candidate_graph_ids": retrieved_candidate_graph_ids,
                        "family_selection_trace": [
                            {
                                "selected_rank": route["rank"],
                                "original_candidate_rank": route[
                                    "original_candidate_rank"
                                ],
                                "candidate_id": route["candidate_id"],
                                "diagnosis_label": route["diagnosis_label"],
                                "family_key": route["family_key"],
                                "assignment_method": route[
                                    "family_assignment_method"
                                ],
                            }
                            for route in selected_routes
                        ],
                        "family_routes": [
                            {
                                "family_rank": route["rank"],
                                "graph_id": str(route["family_key"]).removeprefix(
                                    "graph:"
                                ),
                                "family_key": route["family_key"],
                                "representative_diagnosis": route[
                                    "diagnosis_label"
                                ],
                                "alternatives": route["family_alternatives"],
                            }
                            for route in selected_routes
                        ],
                        "citation_allowlist": list(citation_allowlist),
                        "citation_allowlist_sha256": provenance_sha256(
                            list(citation_allowlist)
                        ),
                        "candidate_choice_allowlist_sha256": provenance_sha256([
                            [candidate_id, label, list(source_ids)]
                            for candidate_id, label, source_ids in candidate_choices
                        ]),
                        "provenance_contract_id": PROVENANCE_CONTRACT_ID,
                        "section_context_filter_contract_id": (
                            "not-applicable-hybrid-premise-neighbor-retrieval"
                        ),
                        "abstain": abstain,
                        "error": None,
                        "response_sha256": response_sha256,
                        "route_selection_policy": (
                            "family-first-graph-membership-v7"
                        ),
                        "generated_single_route_response_used": False,
                    }
                except Exception as exc:
                    case_id = str(query_record.get("case_id"))
                    case_failures.append(
                        exception_record(exc, stage="graph_candidate_retrieval", case_id=case_id)
                    )
                    output = {
                        **query_record,
                        "ranked_candidates": [],
                        "retrieved_candidate_labels": retrieved_candidate_labels,
                        "retrieved_candidate_graph_ids": retrieved_candidate_graph_ids,
                        "citation_allowlist": list(citation_allowlist),
                        "abstain": True,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                case_outputs.append(output)
                if RUN_SCOPE == "development" and len(case_outputs) == 5:
                    # checkpoint["all_gates_pass"] is informational, not an
                    # automated gate: this job keeps running the remaining
                    # cases and the paid generation phase regardless of its
                    # value. A researcher must read
                    # five_case_structural_checkpoint.json and decide whether
                    # to proceed before authorizing a larger run -- this
                    # matches "never submits or retries a paid job
                    # automatically" (experimentsReadme.md); it does not
                    # extend to auto-halting on a structural failure either.
                    checkpoint = audit_retrieval_checkpoint(
                        case_outputs,
                        expected_case_ids=expected_case_ids[:5],
                    )
                    checkpoint["run_id"] = run_id
                    _write_json_new(
                        run_root / "five_case_structural_checkpoint.json",
                        checkpoint,
                    )

            boundary_result = audit_live_query_boundary(
                boundary_audit,
                audit_embedding_boundary,
            )
            prompt_boundary_result = audit_medgemma_prompt_boundary(prompt_audit)
            _write_json_new(
                run_root / "query_execution_audit.json",
                {
                    "run_id": run_id,
                    "case_failure_count": len(case_failures),
                    "case_failures": case_failures,
                    "live_query_embedding_boundary": boundary_result,
                    "medgemma_prompt_boundary": prompt_boundary_result,
                },
            )
            _write_json_new(
                run_root / "case_outputs.json",
                [_sanitized_case_output(record) for record in case_outputs],
            )
            # retrieval["gates"]["development_authorized"] is likewise
            # informational -- it is written to retrieval_audit.json for
            # manual review, not checked here to stop the run.
            retrieval = audit_five_case_retrieval(
                case_outputs,
                corpus,
                expected_case_ids=expected_case_ids,
            )
            retrieval.update({
                "run_id": run_id,
                "gpu": gpu,
                "graphrag_version": version("graphrag"),
                "vllm_version": VLLM_VERSION,
                "chat_model_id": CHAT_MODEL_ID,
                "embedding_model_id": EMBEDDING_MODEL_ID,
                "embedding_model_revision": EMBEDDING_MODEL_REVISION,
                "medgemma_prompt_budget": frozen_budget_config(),
                "query_package_sha256": package_manifest["package_sha256"],
                "corpus_index_manifest_sha256": EXPECTED_CORPUS_INDEX_MANIFEST_SHA256,
                "vector_tree_sha256": EXPECTED_VECTOR_TREE_SHA256,
                "indexing_invocations": 0,
                "chat_extraction_invocations": 0,
                "community_generation_invocations": 0,
                "embedding_rebuild_invocations": 0,
                "dense_premise_index_build_invocations": (
                    dense_premise_index_build_invocations
                ),
                "route_retrieval": {
                    "policy": "family-first-graph-membership-v7",
                    "dense_neighbor_limit": 64,
                    "final_route_limit": 8,
                },
                "umls": {
                    "release": str(umls_manifest.get("release") or ""),
                    "normalizer_id": umls_normalizer.normalizer_id,
                    "database_sha256": str(
                        umls_manifest.get("database_sha256") or ""
                    ),
                    "raw_rrf_included": False,
                },
                "live_query_embedding_boundary": boundary_result,
                "medgemma_prompt_boundary": prompt_boundary_result,
            })
            _write_json_new(run_root / "retrieval_audit.json", retrieval)
            if RUN_PHASE == "retrieval":
                _write_json_new(run_root / "retrieval_only_summary.json", {
                    "run_id": run_id,
                    "scope": RUN_SCOPE,
                    "phase": RUN_PHASE,
                    "case_count": len(case_outputs),
                    "route_selection_policy": (
                        "family-first-graph-membership-v7"
                    ),
                    "maximum_routes": 8,
                    "maximum_facts": 12,
                    "retrieval_audit_sha256": file_sha256(
                        run_root / "retrieval_audit.json"
                    ),
                })
                return 0
            cases = tuple(
                ClinicalCase(
                    case_id=str(record["case_id"]),
                    dataset="direct",
                    task="diagnosis",
                    sections={
                        str(item["section"]): str(item["text"])
                        for item in record["saved_query_payload"]["patient_evidence"]
                    },
                    gold_label=str(record["evaluation_gold_label"]),
                    disease_category=str(record.get("evaluation_gold_category") or ""),
                    metadata={
                        "patient_role": "query_only",
                        "knowledge_corpus_membership": False,
                    },
                )
                for record in case_outputs
            )
            model = OpenAICompatibleDiagnosticModel(
                model_name=CHAT_MODEL_ID,
                model_revision=CHAT_MODEL_REVISION,
                base_url=CHAT_URL,
                timeout_seconds=600,
                context_window=MAX_MODEL_LEN,
                max_output_tokens=COMPLETION_TOKENS,
            )
            judge_model = OpenAICompatibleDiagnosticModel(
                model_name=JUDGE_MODEL_ID,
                model_revision=JUDGE_MODEL_REVISION,
                base_url=JUDGE_URL,
                timeout_seconds=600,
                context_window=MAX_MODEL_LEN,
                max_output_tokens=ARGUMENT_GENERATOR_COMPLETION_TOKENS,
            )
            runner = ExperimentRunner(
                model=model,
                reasoner_model=model,
                verifier_model=model,
                retriever=FixedGraphRagKnowledgeRetriever(
                    corpus,
                    case_outputs,
                    normalizer=umls_normalizer,
                ),
                flat_retriever=IndependentFlatPremiseRetriever(corpus),
                cache_dir=runtime_paths["model_cache"],
                top_k=12,
                standard_output_token_cap=COMPLETION_TOKENS,
                reasoner_output_token_cap=ARGUMENT_GENERATOR_COMPLETION_TOKENS,
                verifier_output_token_cap=ARGUMENT_CRITIC_COMPLETION_TOKENS,
                argument_method=CURRENT_ARGUMENT_METHOD,
            )
            comparison = run_experiment(
                cases=cases,
                runner=runner,
                output_dir=run_root,
                run_name=str(RUN_CONFIG["run_name"]),
                fail_fast=False,
                before_adjudication=None,
            )
            paths_by_graph: dict[str, list[tuple[str, ...]]] = {}
            category_by_graph: dict[str, str] = {}
            for premise in corpus:
                category_by_graph[premise.graph_id] = premise.category
                paths = paths_by_graph.setdefault(premise.graph_id, [])
                if premise.diagnostic_path not in paths:
                    paths.append(premise.diagnostic_path)
            judge_graphs = tuple(
                DiagnosticGraph(
                    graph_id=graph_id,
                    category=category_by_graph[graph_id],
                    nodes=(),
                    edges=(),
                    diagnostic_paths={
                        f"path:{index}": path
                        for index, path in enumerate(paths_by_graph[graph_id])
                    },
                )
                for graph_id in sorted(paths_by_graph)
            )
            judge_results = write_clinical_judgments(
                records=comparison.records,
                cases=cases,
                graphs=judge_graphs,
                model=judge_model,
                output_path=run_root / "clinical_family_judgments.jsonl",
            )
            _write_json_new(
                run_root / "clinical_family_summary.json",
                {
                    "judge_id": JUDGE_ID,
                    "model_id": JUDGE_MODEL_ID,
                    "evaluation_rule": (
                        "pass_if_same_diagnostic_family_else_fail"
                    ),
                    "abstention_policy": (
                        "exclude_verified_insufficient_evidence;"
                        "fail_execution_and_other"
                    ),
                    "modes": list(summarize_clinical_judgments(
                        comparison.records, judge_results
                    )),
                },
            )
            final_medgemma_prompt_audit = audit_all_completion_prompts(
                prompt_audit, boundary_name="MedGemma"
            )
            final_qwen_prompt_audit = audit_all_completion_prompts(
                judge_prompt_audit, boundary_name="Qwen", allow_not_invoked=True
            )
            structural = audit_validation(
                comparison.output_dir,
                expected_case_ids=expected_case_ids,
                retrieval_audit=retrieval,
            )
            structural.update({
                "run_id": run_id,
                "symbolic_invocation_count": 0,
                "comparison_manifest_sha256": file_sha256(comparison.manifest_path),
                "comparison_predictions_sha256": file_sha256(comparison.predictions_path),
                "comparison_argument_traces_sha256": file_sha256(comparison.argument_traces_path),
                "comparison_adjudications_sha256": file_sha256(comparison.adjudications_path),
                "comparison_presentation_responses_sha256": file_sha256(
                    comparison.presentation_responses_path
                ),
                "comparison_presentation_traces_sha256": file_sha256(
                    comparison.presentation_traces_path
                ),
                "clinical_family_judgments_sha256": file_sha256(
                    run_root / "clinical_family_judgments.jsonl"
                ),
                "clinical_family_summary_sha256": file_sha256(
                    run_root / "clinical_family_summary.json"
                ),
                "final_medgemma_prompt_boundary": final_medgemma_prompt_audit,
                "final_qwen_prompt_boundary": final_qwen_prompt_audit,
            })
            _write_json_new(run_root / "structural_audit.json", structural)
    except Exception as exc:
        primary_exception = exc
        _write_json_new(run_root / "terminal_failure.json", {
            "status": "error",
            "run_id": run_id,
            "first_exception": exception_record(exc, stage="query_validation"),
        })
        raise
    finally:
        for server in reversed(servers):
            _stop_server(server)
        if ipc_paths is not None:
            try:
                cleanup = cleanup_vllm_ipc_namespaces(ipc_paths)
                _write_json_new(run_root / "vllm_ipc_cleanup.json", cleanup)
            except Exception as cleanup_exc:
                _write_json_new(run_root / "vllm_ipc_cleanup_failure.json", {
                    "status": "error",
                    "cleanup_exception": exception_record(
                        cleanup_exc,
                        stage="vllm_ipc_cleanup",
                    ),
                    "primary_exception_preserved": primary_exception is not None,
                })
                if primary_exception is None:
                    raise
        input_unchanged = _directory_manifest(query_root) == initial_package_files
        if not input_unchanged:
            _write_json_new(run_root / "input_mutation_failure.json", {
                "status": "error",
                "message": "Read-only query package changed during execution.",
            })
        try:
            scratch_cleanup = cleanup_clinical_runtime_scratch(
                runtime_paths,
                run_root=run_root,
            )
            _write_json_new(run_root / "clinical_runtime_scratch_cleanup.json", scratch_cleanup)
        except Exception as cleanup_exc:
            _write_json_new(run_root / "clinical_runtime_scratch_cleanup_failure.json", {
                "status": "error",
                "cleanup_exception": exception_record(
                    cleanup_exc,
                    stage="clinical_runtime_scratch_cleanup",
                ),
                "primary_exception_preserved": primary_exception is not None,
            })
            if primary_exception is None:
                raise
        terminal = {
            "run_id": run_id,
            "status": "error" if (
                (run_root / "terminal_failure.json").exists() or not input_unchanged
            ) else "completed",
            "started_at": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "query_only": True,
            "phase": RUN_PHASE,
            "input_package_unchanged": input_unchanged,
            "indexing_invocations": 0,
            "chat_extraction_invocations": 0,
            "community_generation_invocations": 0,
            "embedding_rebuild_invocations": 0,
            "dense_premise_index_build_invocations": (
                dense_premise_index_build_invocations
            ),
            "medgemma_prompt_budget": frozen_budget_config(),
            "model_roles": {
                "medgemma_standard_generator_verifier_and_final_reasoner": {
                    "model_id": CHAT_MODEL_ID,
                    "model_revision": CHAT_MODEL_REVISION,
                },
                "posthoc_family_judge": {
                    "model_id": JUDGE_MODEL_ID,
                    "model_revision": JUDGE_MODEL_REVISION,
                },
                "deterministic_discussion_controller": {
                    "resolver_id": CURRENT_ARGUMENT_METHOD.resolver_id,
                    "model_authority": "termination_and_trace_validation_only",
                },
            },
            "case_error_policy": "record_and_continue",
            "sequential_request_processing": True,
            "reproducibility": {
                "policy_id": "pinned-greedy-sequential-v1",
                "request_processing": "sequential",
                "models": {
                    "medgemma": model.decoding_config,
                    "qwen_family_judge": judge_model.decoding_config,
                },
            },
            "chunked_prefill": True,
            "gpu_memory_utilization": {
                "embedding": 0.10, "qwen_judge": 0.55,
                "medgemma_verifier": 0.20,
            },
            "artifact_sha256": _artifact_hashes(run_root),
        }
        terminal["reproducibility"]["fingerprint_sha256"] = canonical_sha256(
            terminal["reproducibility"]
        )
        _write_json_new(run_root / "terminal_manifest.json", terminal)
        _clear_existing_output_roots(publish_root)
        publish_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(run_root, publish_root)
        if not input_unchanged:
            raise RuntimeError("Read-only query package changed during execution.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
