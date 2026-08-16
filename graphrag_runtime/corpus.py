from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from clinical_cds.direct import DirectDataset, label_key
from clinical_cds.graph_extensions import GraphExtension


FALLBACK_ID = "microsoft-graphrag-controlled-corpus-v1"
GRAPHRAG_VERSION = "3.1.0"
VLLM_VERSION = "0.18.1"
CHAT_MODEL_ID = "google/medgemma-1.5-4b-it"
EMBEDDING_MODEL_ID = "BAAI/bge-small-en-v1.5"
EMBEDDING_BATCH_MAX_TOKENS = 448
DEFAULT_TIMEOUT_SECONDS = 3600
DEFAULT_MAX_COST_USD = 2.50
FORBIDDEN_DOCUMENT_KEYS = {
    "case_id",
    "patient_id",
    "gold_label",
    "target_label",
    "directory_label",
    "annotation_nodes",
    "sections",
}


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ControlledPremise:
    id: str
    title: str
    text: str
    graph_id: str
    category: str
    node_id: str
    diagnosis_label: str
    premise_type: str
    diagnostic_path: tuple[str, ...]
    source_chunk_id: str
    knowledge_source_ids: tuple[str, ...]
    source_origin: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _source_chunk_id(graph_id: str, node_id: str, text: str) -> str:
    identity = canonical_sha256({
        "graph_id": graph_id,
        "node_id": node_id,
        "text": text,
    })[:20]
    return f"source-chunk:{identity}"


def build_controlled_corpus(
    direct: DirectDataset,
    extensions: Iterable[GraphExtension],
) -> tuple[ControlledPremise, ...]:
    """Map supplied and extension KG premises to one comparator-neutral corpus."""
    extensions_by_category = {
        label_key(extension.graph.category): extension
        for extension in extensions
    }
    records: list[ControlledPremise] = []
    for graph in sorted(direct.graphs, key=lambda item: item.graph_id):
        extension = extensions_by_category.get(label_key(graph.category))
        for node in sorted(graph.nodes, key=lambda item: item.node_id):
            if node.kind != "premise":
                continue
            text = " ".join(str(node.text or "").split())
            diagnosis = " ".join(str(node.diagnosis_label or "").split())
            premise_type = " ".join(str(node.premise_type or "").split())
            if not text or not diagnosis or not premise_type:
                raise ValueError(f"Incomplete premise node: {node.node_id}")
            path = tuple(graph.diagnostic_paths.get(label_key(diagnosis)) or ())
            if not path:
                raise ValueError(
                    f"Missing diagnostic path for premise node: {node.node_id}"
                )
            source_chunk_id = _source_chunk_id(
                graph.graph_id,
                node.node_id,
                text,
            )
            if extension is None:
                source_ids = (f"direct-supplied-graph:{graph.graph_id}",)
                source_origin = "supplied_direct_diagnostic_guideline_graph"
            else:
                source_ids = tuple(node.knowledge_source_ids)
                source_origin = "provenance_backed_project_extension"
                if not source_ids:
                    raise ValueError(
                        f"Extension premise lacks source provenance: {node.node_id}"
                    )
            records.append(ControlledPremise(
                id=source_chunk_id,
                title=f"{graph.category} | {diagnosis} | {premise_type}",
                text=text,
                graph_id=graph.graph_id,
                category=graph.category,
                node_id=node.node_id,
                diagnosis_label=diagnosis,
                premise_type=premise_type,
                diagnostic_path=path,
                source_chunk_id=source_chunk_id,
                knowledge_source_ids=source_ids,
                source_origin=source_origin,
            ))
    ids = [record.id for record in records]
    if not records or len(ids) != len(set(ids)):
        raise ValueError("Controlled premise records must be non-empty and unique.")
    return tuple(records)


def audit_controlled_corpus(
    records: Iterable[ControlledPremise],
    *,
    expected_graph_count: int,
    expected_extension_count: int,
) -> dict[str, Any]:
    record_list = tuple(records)
    payloads = tuple(record.to_dict() for record in record_list)
    graph_ids = {record.graph_id for record in record_list}
    extension_records = tuple(
        record
        for record in record_list
        if record.source_origin == "provenance_backed_project_extension"
    )
    forbidden_keys = sorted({
        key
        for payload in payloads
        for key in payload
        if key.casefold() in FORBIDDEN_DOCUMENT_KEYS
    })
    missing_source_provenance = sum(
        not record.source_chunk_id
        or not record.knowledge_source_ids
        or not record.node_id
        or not record.diagnostic_path
        for record in record_list
    )
    gates = {
        "knowledge_graph_count": len(graph_ids) == expected_graph_count,
        "project_extension_count": len({
            record.graph_id for record in extension_records
        }) == expected_extension_count,
        "patient_documents_indexed_zero": not forbidden_keys,
        "no_gold_or_label_leakage_fields": not forbidden_keys,
        "complete_kg_path_source_provenance": missing_source_provenance == 0,
        "unique_source_chunk_ids": len(record_list)
        == len({record.source_chunk_id for record in record_list}),
        "nonempty_premise_text": all(record.text for record in record_list),
    }
    return {
        "record_count": len(record_list),
        "graph_count": len(graph_ids),
        "extension_graph_count": len({
            record.graph_id for record in extension_records
        }),
        "patient_document_count": 0,
        "forbidden_document_keys": forbidden_keys,
        "missing_source_provenance_count": missing_source_provenance,
        "corpus_payload_sha256": canonical_sha256(payloads),
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def _settings_template() -> dict[str, Any]:
    return {
        "completion_models": {
            "local_chat": {
                "type": "litellm",
                "model_provider": "openai",
                "model": CHAT_MODEL_ID,
                "api_base": "http://127.0.0.1:8000/v1",
                "api_key": "local-private-job",
                "retry": {"type": "immediate", "max_retries": 2},
            }
        },
        "embedding_models": {
            "local_embedding": {
                "type": "litellm",
                "model_provider": "openai",
                "model": EMBEDDING_MODEL_ID,
                "api_base": "http://127.0.0.1:8001/v1",
                "api_key": "local-private-job",
                "retry": {"type": "immediate", "max_retries": 2},
            }
        },
        "input": {
            "type": "jsonl",
            # GraphRAG applies string.Template substitution to the full file;
            # ``$$`` survives that pass as the intended regex end anchor.
            "file_pattern": ".*\\.jsonl$$",
            "id_column": "id",
            "title_column": "title",
            "text_column": "text",
            "storage": {"type": "file", "base_dir": "input"},
        },
        "chunking": {
            "type": "tokens",
            "size": 300,
            "overlap": 0,
            "prepend_metadata": [
                "graph_id",
                "category",
                "node_id",
                "diagnosis_label",
                "premise_type",
                "diagnostic_path",
                "source_chunk_id",
                "knowledge_source_ids",
            ],
        },
        "output": {"type": "file", "base_dir": "output"},
        "cache": {
            "type": "json",
            "storage": {"type": "file", "base_dir": "cache"},
        },
        "reporting": {"type": "file", "base_dir": "logs"},
        "vector_store": {
            "type": "lancedb",
            "db_uri": "output/lancedb",
            "vector_size": 384,
        },
        "extract_graph": {
            "completion_model_id": "local_chat",
            "entity_types": ["condition", "finding", "test", "treatment"],
            "max_gleanings": 0,
        },
        "summarize_descriptions": {"completion_model_id": "local_chat"},
        "community_reports": {"completion_model_id": "local_chat"},
        "embed_text": {
            "embedding_model_id": "local_embedding",
            # This remains a conservative GraphRAG-side batching hint. The
            # hard 512-token guarantee is enforced and audited with the exact
            # BGE tokenizer at the outbound embedding boundary.
            "batch_max_tokens": EMBEDDING_BATCH_MAX_TOKENS,
        },
        "local_search": {
            "completion_model_id": "local_chat",
            "embedding_model_id": "local_embedding",
            "top_k_entities": 10,
            "top_k_relationships": 10,
        },
    }


def validate_installed_graphrag_schema(settings: dict[str, Any]) -> dict[str, Any]:
    """Validate against the pinned package when it exists in the active env."""
    try:
        from importlib.metadata import version
        from graphrag.config.models.graph_rag_config import GraphRagConfig
    except ImportError:
        return {
            "status": "not_installed_in_active_environment",
            "expected_version": GRAPHRAG_VERSION,
        }
    installed_version = version("graphrag")
    if installed_version != GRAPHRAG_VERSION:
        return {
            "status": "wrong_version_not_validated",
            "expected_version": GRAPHRAG_VERSION,
            "installed_version": installed_version,
        }
    parsed = GraphRagConfig.model_validate(settings)
    if set(parsed.completion_models) != {"local_chat"}:
        raise ValueError("GraphRAG completion model was not bound by the schema.")
    if set(parsed.embedding_models) != {"local_embedding"}:
        raise ValueError("GraphRAG embedding model was not bound by the schema.")
    input_type = str(parsed.input.type)
    if input_type != "jsonl":
        raise ValueError("GraphRAG input was not bound as JSONL.")
    return {
        "status": "passed",
        "installed_version": installed_version,
        "completion_model_ids": sorted(parsed.completion_models),
        "embedding_model_ids": sorted(parsed.embedding_models),
        "input_type": input_type,
    }


def _write_new(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def prepare_fallback_workspace(
    direct: DirectDataset,
    extensions: Iterable[GraphExtension],
    output_dir: Path,
    *,
    expected_graph_count: int = 25,
    expected_extension_count: int = 1,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_cost_usd: float = DEFAULT_MAX_COST_USD,
    validation_case_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite fallback workspace: {output_dir}"
        )
    records = build_controlled_corpus(direct, extensions)
    audit = audit_controlled_corpus(
        records,
        expected_graph_count=expected_graph_count,
        expected_extension_count=expected_extension_count,
    )
    if not audit["all_gates_pass"]:
        raise ValueError(f"Controlled corpus gates failed: {audit['gates']}")

    corpus_path = output_dir / "input" / "controlled_premises.jsonl"
    corpus_text = "".join(
        json.dumps(
            record.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n"
        for record in records
    )
    _write_new(corpus_path, corpus_text)
    settings = _settings_template()
    schema_validation = validate_installed_graphrag_schema(settings)
    settings_path = output_dir / "settings.template.json"
    _write_new(
        settings_path,
        json.dumps(settings, indent=2, sort_keys=True) + "\n",
    )
    requirements_path = output_dir / "requirements.graphrag.txt"
    _write_new(requirements_path, f"graphrag=={GRAPHRAG_VERSION}\n")

    query_path = None
    if validation_case_ids is not None:
        from .queries import build_validation_queries

        queries = build_validation_queries(direct, validation_case_ids)
        query_path = output_dir / "queries" / "validation_queries.jsonl"
        _write_new(
            query_path,
            "".join(
                json.dumps(
                    query,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ) + "\n"
                for query in queries
            ),
        )

    corpus_sha256 = file_sha256(corpus_path)
    settings_sha256 = file_sha256(settings_path)
    requirements_sha256 = file_sha256(requirements_path)
    query_sha256 = file_sha256(query_path) if query_path is not None else None
    extension_provenance = {
        extension.extension_id: extension.provenance_sha256
        for extension in extensions
    }
    manifest = {
        "fallback_id": FALLBACK_ID,
        "status": "dry_run_prepared_not_submitted",
        "private_job_required": True,
        "paid_job_submitted": False,
        "graphrag_version": GRAPHRAG_VERSION,
        "vllm_version": VLLM_VERSION,
        "dependency_isolation": "job-local virtual environment",
        "query_method": "local_search",
        "chat_model_id": CHAT_MODEL_ID,
        "embedding_model_id": EMBEDDING_MODEL_ID,
        "chat_endpoint": "http://127.0.0.1:8000/v1",
        "embedding_endpoint": "http://127.0.0.1:8001/v1",
        "graphrag_schema_validation": schema_validation,
        "controlled_knowledge_corpus": {
            **audit,
            "supplied_direct_graph_count": expected_graph_count
            - expected_extension_count,
            "project_extension_count": expected_extension_count,
            "patient_cases_are_queries_only": True,
            "pooled_mimic_notes_indexed": False,
            "umls_data_included": False,
            "prior_local_outputs_included": False,
            "extension_provenance_sha256": extension_provenance,
        },
        "controlled_comparator": {
            "flat_rag_role": "independent_comparator_only",
            "flat_rag_corpus_path": "input/controlled_premises.jsonl",
            "graph_rag_corpus_path": "input/controlled_premises.jsonl",
            "flat_rag_corpus_sha256": corpus_sha256,
            "graph_rag_corpus_sha256": corpus_sha256,
            "identical_premise_text_and_records": True,
        },
        "artifacts": {
            "corpus_sha256": corpus_sha256,
            "settings_template_sha256": settings_sha256,
            "requirements_sha256": requirements_sha256,
            "validation_queries_sha256": query_sha256,
        },
        "validation_proposal": {
            "job_count": 1,
            "accelerator": "A100",
            "timeout_seconds": timeout_seconds,
            "maximum_cost_usd": max_cost_usd,
            "automatic_retries": 0,
            "output_destination": "selected by RUN_ID at manual submission time",
            "execution_control": "manual_hf_cli_submission",
        },
        "reproducible_preparation_command": (
            "python -m scripts.prepare_graphrag_runtime "
            "--direct-root data/mimic_iv_ext_direct/unpacked "
            "--output-dir <new-versioned-output-dir> "
            "--timeout-seconds 3600 --max-cost-usd 2.50"
        ),
        "prohibited_partitions_accessed": [],
    }
    manifest_path = output_dir / "dry_run_manifest.json"
    _write_new(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    manifest["artifacts"]["dry_run_manifest_sha256"] = file_sha256(manifest_path)
    return manifest
