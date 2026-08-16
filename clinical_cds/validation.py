from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from clinical_cds.io import load_jsonl
from clinical_cds.experiment import MAIN_MODES, PROTOCOL_ID


def _contains_key(value: object, forbidden: set[str]) -> bool:
    if isinstance(value, dict):
        return any(
            str(key).casefold() in forbidden
            or _contains_key(child, forbidden)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_key(child, forbidden) for child in value)
    return False


def audit_validation(
    run_dir: Path,
    *,
    expected_case_ids: Iterable[str],
    retrieval_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    expected_ids = tuple(expected_case_ids)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    predictions = load_jsonl(run_dir / "predictions.jsonl")
    traces = load_jsonl(run_dir / "argument_traces.jsonl")
    decisions = load_jsonl(run_dir / "adjudications.jsonl")
    mode_values = tuple(mode.value for mode in MAIN_MODES)
    counts = Counter(
        (str(row.get("case_id") or ""), str(row.get("mode") or ""))
        for row in predictions
    )
    trace_by_case = {str(row.get("case_id") or ""): row for row in traces}
    decision_by_case = {
        str(row.get("case_id") or ""): row for row in decisions
    }
    identity_ok = True
    provenance_ok = True
    decision_ok = True
    no_symbolic = True
    no_leakage = True
    flat_independent = True
    for case_id in expected_ids:
        trace = trace_by_case.get(case_id) or {}
        decision = decision_by_case.get(case_id) or {}
        graph_record = next(
            (
                row for row in predictions
                if row.get("case_id") == case_id
                and row.get("mode") == "graph_rag"
            ),
            {},
        )
        flat_record = next(
            (
                row for row in predictions
                if row.get("case_id") == case_id
                and row.get("mode") == "flat_rag"
            ),
            {},
        )
        expected_hash = str(
            (trace.get("argument_input") or {}).get(
                "ordered_retrieval_bundle_sha256"
            )
            or ""
        )
        consumers = trace.get("retrieval_bundle_consumers") or {}
        identity_ok = identity_ok and bool(expected_hash) and all(
            value == expected_hash for value in consumers.values()
        ) and graph_record.get("metadata", {}).get(
            "ordered_retrieval_bundle_sha256"
        ) == expected_hash and decision.get(
            "retrieval_bundle_sha256"
        ) == expected_hash
        provenance_ok = provenance_ok and bool(
            trace.get("knowledge_source_provenance")
        ) and all(
            item.get("node_id")
            and item.get("diagnostic_path")
            and item.get("source_chunk_id")
            and item.get("knowledge_source_ids")
            for item in trace.get("knowledge_source_provenance") or []
        )
        candidate_ids = {
            str(item.get("candidate_id") or "")
            for item in (trace.get("proposal") or {}).get("candidates") or []
        }
        selected = str(decision.get("selected_candidate_id") or "")
        abstained = bool(decision.get("abstained"))
        resolution = trace.get("dialectical_resolution") or {}
        preserved_graph_fallback = (
            not abstained
            and not selected
            and resolution.get("action") in {"keep", "graphrag_fallback"}
            and str(resolution.get("selected_diagnosis") or "")
                == str(graph_record.get("predicted_label") or "")
        )
        decision_ok = decision_ok and (
            (abstained and not selected)
            or (not abstained and selected in candidate_ids)
            or preserved_graph_fallback
        ) and not decision.get("error")
        no_symbolic = no_symbolic and (
            trace.get("symbolic_resolution") is None
            and trace.get("symbolic_resolver_executed") is False
            and trace.get("dialectical_resolver_executed") is True
            and bool(trace.get("dialectical_resolution"))
        )
        no_leakage = no_leakage and not _contains_key(
            trace.get("dialectical_trace") or {},
            {"gold_label", "target_label", "directory_label"},
        )
        flat_contract = trace.get("independent_flat_retrieval") or {}
        retrievers = manifest.get("retrievers") or {}
        flat_independent = flat_independent and (
            flat_contract.get("shared_with_graph_or_argumentation") is False
            and bool(flat_contract.get("ordered_retrieval_bundle_sha256"))
            and flat_record.get("metadata", {}).get(
                "ordered_retrieval_bundle_sha256"
            ) == flat_contract.get("ordered_retrieval_bundle_sha256")
            and flat_record.get("metadata", {}).get("retriever_id")
            == retrievers.get("flat_rag")
            and graph_record.get("metadata", {}).get("retriever_id")
            == retrievers.get("graph_rag")
        )

    gates = {
        "exact_frozen_case_set": (
            bool(expected_ids)
            and tuple(trace_by_case) == expected_ids
            and tuple(decision_by_case) == expected_ids
        ),
        "exact_four_main_modes_once_per_case": (
            len(predictions) == len(expected_ids) * len(mode_values)
            and all(
                counts[(case_id, mode)] == 1
                for case_id in expected_ids
                for mode in mode_values
            )
        ),
        "zero_execution_errors": all(not row.get("error") for row in predictions),
        "shared_retrieval_bundle_identity": identity_ok,
        "flat_rag_independent_retrieval": flat_independent,
        "complete_patient_kg_path_source_provenance": provenance_ok,
        "candidate_only_or_abstain_with_valid_citations": decision_ok,
        "zero_gold_label_leakage": no_leakage,
        "bounded_dialectical_resolver_active": no_symbolic,
        "terminal_versioned_manifest": (
            manifest.get("protocol_id") == PROTOCOL_ID
            and manifest.get("status") == "completed"
            and manifest.get("modes") == list(mode_values)
            and bool(manifest.get("artifact_sha256"))
        ),
        "retrieval_quality_gate": (
            retrieval_audit is None
            or bool((retrieval_audit.get("gates") or {}).get(
                "development_authorized"
            ))
        ),
    }
    return {
        "audit_id": "current-case-structural-validation",
        "case_ids": list(expected_ids),
        "counts": {
            "predictions": len(predictions),
            "traces": len(traces),
            "adjudications": len(decisions),
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }
