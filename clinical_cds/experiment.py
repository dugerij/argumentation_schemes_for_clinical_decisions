from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from clinical_cds.clinical_adjudication import (
    AdjudicationDecision,
)
from clinical_cds.io import load_jsonl, write_jsonl
from clinical_cds.method_contract import (
    CURRENT_ARGUMENT_METHOD,
)
from clinical_cds.presentation import (
    presentation_argument_trace,
    presentation_prediction,
)
from clinical_cds.retrieval import patient_evidence, render_case
from clinical_cds.runner import (
    ExperimentRunner,
    ExperimentRun,
    provenance_retrieval_bundle_sha256,
)
from clinical_cds.schema import (
    ClinicalCase,
    ExperimentMode,
    PredictedObservation,
    PredictionRecord,
    RetrievalBundle,
)


PROTOCOL_ID = "evidence-grounded-comparison-current"
MAIN_MODES = (
    ExperimentMode.DIRECT,
    ExperimentMode.FLAT_RAG,
    ExperimentMode.GRAPH_RAG,
    ExperimentMode.EVIDENCE_GROUNDED_ARGUMENTATION,
)


@dataclass(frozen=True)
class ExperimentArtifacts(ExperimentRun):
    adjudications_path: Path
    presentation_responses_path: Path
    presentation_traces_path: Path


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol(argument_method=None) -> dict[str, Any]:
    method = argument_method or CURRENT_ARGUMENT_METHOD
    resolver_id = method.resolver_id
    return {
        "protocol_id": PROTOCOL_ID,
        "main_modes": [mode.value for mode in MAIN_MODES],
        "argument_method": method.to_dict(),
        "retrieval_control": {
            "graph_rag_generator_critic": (
                "one_byte_identical_case_specific_kg_bundle"
            ),
            "flat_rag_role": "independent_comparator_only",
            "patients": "queries_only_not_indexed_documents",
            "knowledge_corpus": (
                "24_direct_diagnostic_guideline_graphs_plus_"
                "provenance_backed_gastritis_extension"
            ),
        },
        "primary_decision": {
            "resolver_id": resolver_id,
            "allowed_output": "accepted_child_or_family_fallback_or_abstain",
            "requires_valid_argument_and_evidence_ids": True,
            "gold_label_access": "none",
            "llm_final_reasoner_authority": "none",
        },
        "presentation_contract": {
            "internal_identifiers": "private_machine_audit_only",
            "generator_candidate_contract": (
                "one_active_hypothesis_exact_cited_move"
            ),
            "readable_responses": "diagnosis_and_explanation_without_opaque_ids",
            "readable_stage_traces": (
                "generator_verifier_resolver_without_opaque_ids"
            ),
        },
        "rejected_development_ablation": {
            "mode": ExperimentMode.SYMBOLIC_ARGUMENT.value,
            "status": "preserved_but_excluded_from_primary_and_paid_comparison",
        },
    }


def protocol_sha256(argument_method=None) -> str:
    return _canonical_sha256(protocol(argument_method))


def _adjudication_error(
    case_id: str,
    model_id: str,
    bundle_sha256: str,
    error: str,
) -> AdjudicationDecision:
    return AdjudicationDecision(
        case_id=case_id,
        selected_candidate_id="",
        selected_diagnosis="",
        abstained=True,
        confidence="low",
        supporting_argument_ids=(),
        evidence_ids=(),
        candidate_assessments=(),
        latency_seconds=0.0,
        model_id=model_id,
        retrieval_bundle_sha256=bundle_sha256,
        error=error,
    )


def _argument_prediction(
    case: ClinicalCase,
    run_id: str,
    team: Any,
    trace: dict[str, Any],
    decision: AdjudicationDecision,
    bundle: RetrievalBundle,
    runner: ExperimentRunner,
) -> PredictionRecord:
    valid_ids = (
        tuple(item[0] for item in patient_evidence(case))
        + bundle.evidence_ids
        + tuple(team.citation_allowlist)
    )
    nodes = {
        str(node.get("argument_id") or ""): node
        for node in (trace.get("argument_graph") or {}).get("nodes") or []
    }
    selected_nodes = [
        nodes[argument_id]
        for argument_id in decision.supporting_argument_ids
        if argument_id in nodes
    ]
    observations = tuple(
        PredictedObservation(
            text=str(node.get("premise") or ""),
            source_id=(node.get("evidence_ids") or [None])[0],
        )
        for node in selected_nodes
        if str(node.get("premise") or "").strip()
    )
    reasoning = " ".join(
        str(node.get("conclusion") or "").strip()
        for node in selected_nodes
        if str(node.get("conclusion") or "").strip()
    )
    error = decision.error or team.error
    resolution = dict(team.dialectical_resolution or {})
    if not decision.abstained:
        abstention_category = "not_abstained"
    elif error:
        abstention_category = "execution_failure"
    elif not decision.selected_diagnosis and str(
        resolution.get("action") or resolution.get("outcome") or ""
    ) in {"abstain", "differential_abstain", "insufficient_abstain"}:
        abstention_category = "insufficient_evidence"
    else:
        abstention_category = "other_abstention"
    return PredictionRecord(
        run_id=run_id,
        case_id=case.case_id,
        dataset=case.dataset,
        task=case.task,
        mode=ExperimentMode.EVIDENCE_GROUNDED_ARGUMENTATION,
        model_id=f"deterministic:{team.argument_method.resolver_id}",
        gold_label=case.gold_label,
        predicted_label=("" if decision.abstained else decision.selected_diagnosis),
        reasoning=reasoning,
        citations=decision.evidence_ids,
        observations=observations,
        abstained=decision.abstained,
        latency_seconds=round(team.latency_seconds + decision.latency_seconds, 6),
        prompt_hash=team.shared_argument_critic_artifact_sha256,
        cache_hit=team.cache_hit,
        valid_evidence_ids=valid_ids,
        quality_flags=case.quality_flags,
        error=error,
        metadata={
            "protocol_id": PROTOCOL_ID,
            "argument_method_id": team.argument_method.method_id,
            "resolver_id": team.argument_method.resolver_id,
            "supporting_argument_ids": list(decision.supporting_argument_ids),
            "ordered_retrieval_bundle_sha256": team.retrieval_bundle_sha256,
            "candidate_inventory_sha256": team.candidate_inventory_sha256,
            "reasoner_artifact_sha256": team.reasoner_artifact_sha256,
            "critic_artifact_sha256": team.critic_artifact_sha256,
            "shared_argument_critic_artifact_sha256": (
                team.shared_argument_critic_artifact_sha256
            ),
            "resolver_input_sha256": decision.adjudicator_input_sha256,
            "symbolic_resolver_executed": False,
            "dialectical_resolver_executed": True,
            "gold_label_access": "evaluation_output_only",
            "resolver_action": str(resolution.get("action") or ""),
            "resolver_outcome": str(resolution.get("outcome") or ""),
            "resolver_reason": str(resolution.get("reason") or ""),
            "leading_diagnoses": list(resolution.get("leading_diagnoses") or ()),
            "possible_diagnoses": list(
                resolution.get("possible_diagnoses")
                or ([decision.selected_diagnosis] if decision.selected_diagnosis else [])
            ),
            "abstention_category": abstention_category,
            "patient_input_sha256": hashlib.sha256(
                render_case(case).encode("utf-8")
            ).hexdigest(),
        },
    )


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_experiment(
    *,
    cases: Iterable[ClinicalCase],
    runner: ExperimentRunner,
    output_dir: Path,
    run_name: str,
    resume: bool = False,
    fail_fast: bool = False,
    before_adjudication: Callable[[], None] | None = None,
    adjudicate: Callable[[dict[str, Any], Any], AdjudicationDecision] | None = None,
) -> ExperimentArtifacts:
    """Execute only the four current conditions with atomic checkpoints."""
    case_list = tuple(cases)
    if not case_list:
        raise ValueError("At least one case is required.")
    if not run_name.strip():
        raise ValueError("A run name is required.")
    if runner.argument_method != CURRENT_ARGUMENT_METHOD:
        raise ValueError("The comparison runner requires the supported argumentation method.")
    if runner.flat_retriever is runner.retriever:
        raise ValueError(
            "The current comparison requires an independent Flat RAG "
            "retriever; it cannot reuse the GraphRAG retriever instance."
        )
    if before_adjudication is not None or adjudicate is not None:
        raise ValueError(
            "The bounded resolver path has no separate Final Reasoner boundary."
        )

    run_dir = Path(output_dir) / run_name
    if run_dir.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite existing current run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=resume)
    predictions_path = run_dir / "predictions.jsonl"
    traces_path = run_dir / "argument_traces.jsonl"
    adjudications_path = run_dir / "adjudications.jsonl"
    presentation_responses_path = run_dir / "presentation_responses.jsonl"
    presentation_traces_path = run_dir / "presentation_traces.jsonl"
    manifest_path = run_dir / "manifest.json"

    protocol_config = protocol(runner.argument_method)
    protocol_hash = protocol_sha256(runner.argument_method)
    resolver_id = runner.argument_method.resolver_id
    model_roles = {
        "standard": runner.model.model_id,
        "generator": runner.reasoner_model.model_id,
        "critic": runner.verifier_model.model_id,
        "final_trace_reasoner": "none",
        "resolver": resolver_id,
    }
    reproducibility = {
        "policy_id": "pinned-greedy-sequential-v1",
        "request_processing": "sequential",
        "models": {
            role: {
                "model_id": model.model_id,
                "cache_identity": getattr(model, "cache_identity", model.model_id),
                "decoding": getattr(model, "decoding_config", {}),
            }
            for role, model in (
                ("standard", runner.model),
                ("generator", runner.reasoner_model),
                ("critic", runner.verifier_model),
            )
        },
    }
    reproducibility["fingerprint_sha256"] = _canonical_sha256(reproducibility)
    expected_manifest = {
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": protocol_hash,
        "run_id": run_name,
        "status": "running",
        "case_count": len(case_list),
        "case_ids_sha256": _canonical_sha256(
            [case.case_id for case in case_list]
        ),
        "modes": [mode.value for mode in MAIN_MODES],
        "model_roles": model_roles,
        "inference_stage_order": [
            "direct_flat_and_graph_comparators", "eight_family_evidence_profiling",
            "eight_to_four_evidence_aware_activation",
            "direct_cited_differential", "adversarial_full_context_verification",
            "independent_proposition_bound_attack_validation",
            "abstention_only_protected_graph_incumbent",
            "deterministic_family_first_maintain_switch_fallback_or_abstain",
        ],
        "sequential_model_boundary": True,
        "reproducibility": reproducibility,
        "retriever_id": runner.retriever.retriever_id,
        "retrievers": {
            "flat_rag": runner.flat_retriever.retriever_id,
            "graph_rag": runner.retriever.retriever_id,
            "argument_generator": runner.retriever.retriever_id,
            "argument_critic": runner.retriever.retriever_id,
        },
        "top_k": runner.top_k,
        "protocol": protocol_config,
        "outputs": {
            "predictions": predictions_path.name,
            "argument_traces": traces_path.name,
            "adjudications": adjudications_path.name,
            "presentation_responses": presentation_responses_path.name,
            "presentation_traces": presentation_traces_path.name,
        },
        "artifact_roles": {
            "predictions": "private_machine_audit",
            "argument_traces": "private_machine_audit",
            "adjudications": "private_machine_audit",
            "presentation_responses": "identifier_free_readable_output",
            "presentation_traces": "identifier_free_readable_output",
        },
    }
    if resume:
        if not manifest_path.is_file():
            raise ValueError("Cannot resume without a current manifest.")
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for field in (
            "protocol_id",
            "protocol_sha256",
            "run_id",
            "case_count",
            "case_ids_sha256",
            "modes",
            "model_roles",
            "reproducibility",
            "retriever_id",
            "retrievers",
            "top_k",
        ):
            if existing_manifest.get(field) != expected_manifest[field]:
                raise ValueError(f"Cannot resume: current manifest field changed: {field}")
    else:
        _write_manifest(manifest_path, expected_manifest)

    records_by_key = {
        (record.case_id, record.mode): record
        for record in (
            PredictionRecord.from_dict(row)
            for row in load_jsonl(predictions_path)
        )
    }
    traces_by_case = {
        str(row.get("case_id") or ""): row
        for row in load_jsonl(traces_path)
    }
    decisions_by_case = {
        str(row.get("case_id") or ""): row
        for row in load_jsonl(adjudications_path)
    }

    case_number_by_id = {
        item.case_id: index
        for index, item in enumerate(case_list, start=1)
    }

    def persist_outputs() -> None:
        ordered_records = [
            records_by_key[(item.case_id, mode)].to_dict()
            for item in case_list
            for mode in MAIN_MODES
            if (item.case_id, mode) in records_by_key
        ]
        write_jsonl(predictions_path, ordered_records)
        write_jsonl(
            traces_path,
            [
                traces_by_case[item.case_id]
                for item in case_list
                if item.case_id in traces_by_case
            ],
        )
        write_jsonl(
            adjudications_path,
            [
                decisions_by_case[item.case_id]
                for item in case_list
                if item.case_id in decisions_by_case
            ],
        )
        write_jsonl(
            presentation_responses_path,
            [
                presentation_prediction(
                    records_by_key[(item.case_id, mode)].to_dict(),
                    case_number=case_number_by_id[item.case_id],
                )
                for item in case_list
                for mode in MAIN_MODES
                if (item.case_id, mode) in records_by_key
            ],
        )
        write_jsonl(
            presentation_traces_path,
            [
                presentation_argument_trace(
                    traces_by_case[item.case_id],
                    case_number=case_number_by_id[item.case_id],
                )
                for item in case_list
                if item.case_id in traces_by_case
            ],
        )

    prepared: list[tuple[ClinicalCase, Any, dict[str, Any], RetrievalBundle, str]] = []

    def resolve_case(
        case: ClinicalCase,
        team: Any,
        trace: dict[str, Any],
        bundle: RetrievalBundle,
        bundle_hash: str,
    ) -> None:
        if team.error:
            decision = _adjudication_error(
                case.case_id,
                f"deterministic:{resolver_id}",
                bundle_hash,
                team.error,
            )
        else:
            resolution = dict(team.dialectical_resolution)
            accepted = resolution.get("outcome") == "accepted"
            active = set(team.dialectical_trace.get("active_candidate_ids", ()))
            direct = team.dialectical_trace.get("direct_differential") or {}
            attack = team.dialectical_trace.get("attack") or {}
            assessments = tuple({
                "candidate_id": item.candidate_id,
                "verdict": (
                    str(resolution.get("action") or "abstain")
                    if item.candidate_id == team.selected_candidate_id
                    else "rejected_alternative" if item.candidate_id in active
                    else "not_evaluated_due_to_budget"
                ),
                "reason": (
                    str(resolution.get("reason") or "")
                    if item.candidate_id == team.selected_candidate_id
                    else str(attack.get("explanation") or direct.get("rationale") or "")
                ),
            } for item in team.candidate_inventory)
            decision = AdjudicationDecision(
                case_id=case.case_id,
                selected_candidate_id=(team.selected_candidate_id if accepted else ""),
                selected_diagnosis=(
                    str(resolution.get("selected_diagnosis") or "")
                    if accepted else ""
                ),
                abstained=not accepted,
                confidence="not_applicable",
                supporting_argument_ids=(
                    team.selected_argument_ids if accepted else ()
                ),
                evidence_ids=(team.selected_evidence_ids if accepted else ()),
                candidate_assessments=assessments,
                latency_seconds=0.0,
                model_id=f"deterministic:{resolver_id}",
                retrieval_bundle_sha256=bundle_hash,
                adjudicator_input_sha256=_canonical_sha256(
                    team.dialectical_trace
                ),
                adjudicator_prompt_sha256="",
            )
        if decision.retrieval_bundle_sha256 != bundle_hash:
            raise ValueError("Resolver input is not bound to the fixed bundle.")
        trace["resolver_decision"] = decision.to_dict()
        trace["llm_adjudication"] = {
            "executed": False,
            "authority": "none",
            "subordinated_to": resolver_id,
        }
        traces_by_case[case.case_id] = trace
        decisions_by_case[case.case_id] = decision.to_dict()
        records_by_key[
            (case.case_id, ExperimentMode.EVIDENCE_GROUNDED_ARGUMENTATION)
        ] = _argument_prediction(
            case,
            run_name,
            team,
            trace,
            decision,
            bundle,
            runner,
        )
        persist_outputs()
        if fail_fast and any(
            record.error
            for key, record in records_by_key.items()
            if key[0] == case.case_id
        ):
            raise RuntimeError(f"current case failed: {case.case_id}")

    for case in case_list:
        complete = (
            all((case.case_id, mode) in records_by_key for mode in MAIN_MODES)
            and case.case_id in traces_by_case
            and case.case_id in decisions_by_case
        )
        if complete:
            continue

        bundle = runner.retriever.retrieve(case, top_k=runner.top_k)
        flat_bundle = runner.flat_retriever.retrieve(case, top_k=runner.top_k)
        bundle_hash = provenance_retrieval_bundle_sha256(bundle)
        flat_bundle_hash = provenance_retrieval_bundle_sha256(flat_bundle)
        if any(
            not fact.node_id
            or not fact.diagnostic_path
            or not fact.source_chunk_id
            for fact in bundle.facts
        ):
            raise ValueError(
                f"Incomplete KG node/path/source provenance for {case.case_id}."
            )

        for mode in (
            ExperimentMode.DIRECT,
            ExperimentMode.FLAT_RAG,
            ExperimentMode.GRAPH_RAG,
        ):
            record = runner.predict(
                case,
                mode,
                run_id=run_name,
                bundle=(
                    RetrievalBundle(facts=(), query_tokens=())
                    if mode == ExperimentMode.DIRECT
                    else flat_bundle
                    if mode == ExperimentMode.FLAT_RAG
                    else bundle
                ),
            )
            if mode == ExperimentMode.FLAT_RAG:
                if (
                    record.metadata.get("ordered_retrieval_bundle_sha256")
                    != flat_bundle_hash
                ):
                    raise ValueError("Flat RAG changed its independent bundle.")
                if (
                    record.metadata.get("retriever_id")
                    != runner.flat_retriever.retriever_id
                ):
                    raise ValueError("Flat RAG used the wrong retriever.")
            elif mode == ExperimentMode.GRAPH_RAG:
                if (
                    record.metadata.get("ordered_retrieval_bundle_sha256")
                    != bundle_hash
                ):
                    raise ValueError("Graph RAG changed the fixed KG bundle.")
                if (
                    record.metadata.get("retriever_id")
                    != runner.retriever.retriever_id
                ):
                    raise ValueError("Graph RAG used the wrong retriever.")
            records_by_key[(case.case_id, mode)] = record

        try:
            team = runner.run_argument_team(
                case, bundle, run_id=run_name,
                graph_incumbent=records_by_key[
                    (case.case_id, ExperimentMode.GRAPH_RAG)
                ],
            )
        except Exception as exc:
            if fail_fast:
                raise
            team = runner.argument_team_error_result(
                case,
                bundle,
                run_id=run_name,
                error=exc,
            )
        if team.retrieval_bundle_sha256 != bundle_hash:
            raise ValueError("Generator/critic did not receive the fixed bundle.")
        if not team.dialectical_resolution:
            raise ValueError("The bounded dialectical resolver did not execute.")
        trace = team.to_dict()
        trace["symbolic_resolver_executed"] = False
        trace["dialectical_resolver_executed"] = True
        trace["primary_decision"] = resolver_id
        trace["retrieval_bundle_consumers"] = {
            "graph_rag": bundle_hash,
            "argument_generator": bundle_hash,
            "argument_critic": bundle_hash,
            "deterministic_resolver": bundle_hash,
        }
        trace["independent_flat_retrieval"] = {
            "retriever_id": runner.flat_retriever.retriever_id,
            "ordered_retrieval_bundle_sha256": flat_bundle_hash,
            "shared_with_graph_or_argumentation": False,
        }
        trace["citation_allowlist_consumers"] = {
            "graph_rag": team.citation_allowlist_sha256,
            "argument_generator": team.citation_allowlist_sha256,
            "argument_critic": team.citation_allowlist_sha256,
            "deterministic_resolver": team.citation_allowlist_sha256,
        }
        prepared.append((case, team, trace, bundle, bundle_hash))
        resolve_case(case, team, trace, bundle, bundle_hash)

    records = tuple(
        records_by_key[(case.case_id, mode)]
        for case in case_list
        for mode in MAIN_MODES
    )
    manifest = {
        **expected_manifest,
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "record_count": len(records),
        "error_count": sum(record.error is not None for record in records),
        "artifact_sha256": {
            "predictions": _file_sha256(predictions_path),
            "argument_traces": _file_sha256(traces_path),
            "adjudications": _file_sha256(adjudications_path),
            "presentation_responses": _file_sha256(
                presentation_responses_path
            ),
            "presentation_traces": _file_sha256(presentation_traces_path),
        },
    }
    _write_manifest(manifest_path, manifest)
    return ExperimentArtifacts(
        run_id=run_name,
        output_dir=run_dir,
        predictions_path=predictions_path,
        argument_traces_path=traces_path,
        manifest_path=manifest_path,
        records=records,
        adjudications_path=adjudications_path,
        presentation_responses_path=presentation_responses_path,
        presentation_traces_path=presentation_traces_path,
    )
