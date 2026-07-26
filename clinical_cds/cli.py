from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

from clinical_cds.direct import (
    DirectDataset,
    load_direct_dataset,
    prepare_direct_archive,
    select_direct_partition,
)
from clinical_cds.evaluation import (
    load_prediction_records,
    normalized_label_key,
    write_evaluation_artifacts,
)
from clinical_cds.medqa import load_medqa_cases
from clinical_cds.model import ollama_model_from_env
from clinical_cds.normalization import UMLSNormalizer
from clinical_cds.patient import load_patient_case
from clinical_cds.perturbation import (
    build_section_removal_pairs,
    evaluate_perturbations,
    write_perturbation_artifacts,
)
from clinical_cds.retrieval import KnowledgeRetriever
from clinical_cds.runner import ExperimentRunner, run_experiment
from clinical_cds.schema import ClinicalCase, ExperimentMode
from clinical_cds.terminology.local_umls import (
    DEFAULT_LOCAL_UMLS_DB_PATH,
    DEFAULT_LOCAL_UMLS_META_DIR,
    LocalUMLSBuildConfig,
    build_local_umls_database,
)
from clinical_cds.terminology.vocabularies import SOURCE_PRIORITY
from clinical_cds.trace_visualization import (
    load_argument_trace,
    plot_argument_trace,
)


DEFAULT_DIRECT_ROOT = Path("data/mimic_iv_ext_direct/unpacked")
DEFAULT_DIRECT_ARCHIVE = Path(
    "data/mimic_iv_ext_direct/raw/mimic-iv-ext-direct-1.0.0.zip"
)
DEFAULT_MEDQA_ROOT = Path("data/medqa")
DEFAULT_EXPERIMENT_ROOT = Path("output/experiments")
DEFAULT_CACHE_ROOT = Path("output/cache/model_responses")


def _add_umls_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--umls-db",
        type=Path,
        default=None,
        help="Local UMLS SQLite index used for terminology normalization.",
    )


def _add_agent_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--reasoner-model",
        default=None,
        help="Optional reasoner model; defaults to --model.",
    )
    parser.add_argument(
        "--verifier-model",
        default=None,
        help="Optional verifier model; defaults to --model.",
    )


def _add_progress_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the stage progress bar written to standard error.",
    )


def _add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        choices=("direct", "medqa"),
        default="direct",
    )
    parser.add_argument("--direct-root", type=Path, default=DEFAULT_DIRECT_ROOT)
    parser.add_argument("--medqa-root", type=Path, default=DEFAULT_MEDQA_ROOT)
    parser.add_argument("--medqa-split", choices=("dev", "test"), default="test")
    parser.add_argument("--medqa-graph-covered-only", action="store_true")
    parser.add_argument(
        "--partition",
        choices=("all", "development", "test"),
        default="test",
        help="Deterministic DiReCT partition; ignored by datasets with official splits.",
    )
    parser.add_argument(
        "--strict-direct",
        action="store_true",
        help="Exclude DiReCT cases without a graph-consistent final conclusion.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m clinical_cds",
        description="Run diagnosis-only graph and argumentation experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare-direct",
        help="Extract the licensed DiReCT ZIP and its two RAR archives.",
    )
    prepare.add_argument("--archive", type=Path, default=DEFAULT_DIRECT_ARCHIVE)
    prepare.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mimic_iv_ext_direct"),
    )
    prepare.add_argument("--overwrite", action="store_true")

    build_umls = subparsers.add_parser(
        "build-umls",
        help="Build the local terminology index from licensed UMLS META files.",
    )
    build_umls.add_argument(
        "--meta-dir",
        type=Path,
        default=DEFAULT_LOCAL_UMLS_META_DIR,
    )
    build_umls.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_LOCAL_UMLS_DB_PATH,
    )
    build_umls.add_argument(
        "--sources",
        default=",".join(SOURCE_PRIORITY),
    )
    build_umls.add_argument("--languages", default="ENG")
    build_umls.add_argument("--batch-size", type=int, default=5000)

    audit = subparsers.add_parser(
        "audit-direct",
        help="Validate DiReCT annotations and guideline coverage without model calls.",
    )
    audit.add_argument("--direct-root", type=Path, default=DEFAULT_DIRECT_ROOT)
    audit.add_argument("--output", type=Path, default=None)
    _add_umls_argument(audit)

    medqa_audit = subparsers.add_parser(
        "audit-medqa",
        help="Count diagnostic MedQA questions and DiReCT graph coverage.",
    )
    medqa_audit.add_argument("--direct-root", type=Path, default=DEFAULT_DIRECT_ROOT)
    medqa_audit.add_argument("--medqa-root", type=Path, default=DEFAULT_MEDQA_ROOT)
    _add_umls_argument(medqa_audit)

    run = subparsers.add_parser(
        "run-experiment",
        help="Run one model over the requested diagnostic ablation conditions.",
    )
    _add_dataset_arguments(run)
    _add_umls_argument(run)
    _add_agent_model_arguments(run)
    _add_progress_argument(run)
    run.add_argument(
        "--modes",
        default="all",
        help=(
            "all or a comma-separated list of direct, flat_rag, graph_rag, "
            "structured_argument, symbolic_argument."
        ),
    )
    run.add_argument("--model", default=None)
    run.add_argument("--seed", type=int, default=17)
    run.add_argument("--top-k", type=int, default=6)
    run.add_argument("--timeout-seconds", type=float, default=600.0)
    run.add_argument("--context-window", type=int, default=8192)
    run.add_argument("--max-output-tokens", type=int, default=1024)
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--output-dir", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    run.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_ROOT)
    run.add_argument("--run-name", default=None)
    run.add_argument("--allow-remote-data", action="store_true")
    run.add_argument("--continue-on-error", action="store_true")

    evaluate = subparsers.add_parser(
        "evaluate-run",
        help="Recompute deterministic metrics, paired tests, and plots.",
    )
    _add_dataset_arguments(evaluate)
    _add_umls_argument(evaluate)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, default=None)

    perturb = subparsers.add_parser(
        "run-perturbations",
        help="Measure response to removal of the most-supported clinical section.",
    )
    perturb.add_argument("--direct-root", type=Path, default=DEFAULT_DIRECT_ROOT)
    perturb.add_argument(
        "--partition",
        choices=("development", "test"),
        default="test",
    )
    perturb.add_argument(
        "--mode",
        choices=tuple(mode.value for mode in ExperimentMode),
        default=ExperimentMode.SYMBOLIC_ARGUMENT.value,
    )
    perturb.add_argument("--strict-direct", action="store_true")
    perturb.add_argument("--model", default=None)
    perturb.add_argument("--seed", type=int, default=17)
    perturb.add_argument("--top-k", type=int, default=6)
    perturb.add_argument("--timeout-seconds", type=float, default=600.0)
    perturb.add_argument("--context-window", type=int, default=8192)
    perturb.add_argument("--max-output-tokens", type=int, default=1024)
    perturb.add_argument("--limit", type=int, default=None)
    perturb.add_argument("--output-dir", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    perturb.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_ROOT)
    perturb.add_argument("--run-name", default=None)
    perturb.add_argument("--allow-remote-data", action="store_true")
    _add_umls_argument(perturb)
    _add_agent_model_arguments(perturb)
    _add_progress_argument(perturb)

    diagnose = subparsers.add_parser(
        "diagnose",
        help="Return a diagnosis for a submitted patient-state JSON file.",
    )
    diagnose.add_argument("--patient", type=Path, required=True)
    diagnose.add_argument("--direct-root", type=Path, default=DEFAULT_DIRECT_ROOT)
    diagnose.add_argument(
        "--mode",
        choices=tuple(mode.value for mode in ExperimentMode),
        default=ExperimentMode.SYMBOLIC_ARGUMENT.value,
    )
    diagnose.add_argument("--model", default=None)
    diagnose.add_argument("--seed", type=int, default=17)
    diagnose.add_argument("--top-k", type=int, default=6)
    diagnose.add_argument("--timeout-seconds", type=float, default=600.0)
    diagnose.add_argument("--context-window", type=int, default=8192)
    diagnose.add_argument("--max-output-tokens", type=int, default=1024)
    diagnose.add_argument("--output-dir", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    diagnose.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_ROOT)
    diagnose.add_argument("--allow-remote-data", action="store_true")
    _add_umls_argument(diagnose)
    _add_agent_model_arguments(diagnose)
    _add_progress_argument(diagnose)

    trace_plot = subparsers.add_parser(
        "plot-trace",
        help="Render a compact reasoner-verifier-resolver trace for one case.",
    )
    trace_plot.add_argument("--traces", type=Path, required=True)
    trace_plot.add_argument("--case-id", default=None)
    trace_plot.add_argument("--trace-id", default=None)
    trace_plot.add_argument("--output", type=Path, default=None)
    return parser


def _parse_modes(value: str) -> tuple[ExperimentMode, ...]:
    if value.strip().casefold() == "all":
        return tuple(ExperimentMode)
    modes: list[ExperimentMode] = []
    for item in value.split(","):
        normalized = item.strip()
        if not normalized:
            continue
        mode = ExperimentMode(normalized)
        if mode not in modes:
            modes.append(mode)
    if not modes:
        raise ValueError("At least one experiment mode is required.")
    return tuple(modes)


def _stable_limit(
    cases: Iterable[ClinicalCase],
    limit: int | None,
    seed: int,
) -> tuple[ClinicalCase, ...]:
    case_list = tuple(cases)
    if limit is None:
        return case_list
    if limit < 1:
        raise ValueError("--limit must be positive.")
    ranked = sorted(
        case_list,
        key=lambda case: hashlib.sha256(
            f"{seed}|{case.case_id}".encode("utf-8")
        ).hexdigest(),
    )
    return tuple(ranked[:limit])


def _is_graph_consistent(case: ClinicalCase) -> bool:
    excluded = {
        "missing_guideline_graph",
        "conclusion_outside_category_graph",
        "conclusion_not_leaf",
    }
    return not bool(set(case.quality_flags) & excluded)


def _load_normalizer(args: argparse.Namespace) -> UMLSNormalizer | None:
    db_path = getattr(args, "umls_db", None)
    if db_path is None:
        return None
    return UMLSNormalizer.from_path(db_path)


def _load_agent_model(
    args: argparse.Namespace,
    *,
    role: str,
    base_model: object,
):
    model_name = getattr(args, f"{role}_model", None)
    if not model_name:
        return base_model
    return ollama_model_from_env(
        model_name=model_name,
        seed=args.seed,
        allow_remote=args.allow_remote_data,
        timeout_seconds=args.timeout_seconds,
        context_window=args.context_window,
        max_output_tokens=args.max_output_tokens,
    )


def _load_experiment_cases(
    args: argparse.Namespace,
    direct: DirectDataset,
    normalizer: UMLSNormalizer | None = None,
) -> tuple[ClinicalCase, ...]:
    if args.dataset == "direct":
        cases = select_direct_partition(direct.cases, args.partition)
        if args.strict_direct:
            cases = tuple(case for case in cases if _is_graph_consistent(case))
        return cases
    if args.dataset == "medqa":
        return load_medqa_cases(
            args.medqa_root,
            split=args.medqa_split,
            graphs=direct.graphs,
            diagnostic_only=True,
            graph_covered_only=args.medqa_graph_covered_only,
            normalizer=normalizer,
        )
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def _audit_direct(args: argparse.Namespace) -> int:
    direct = load_direct_dataset(args.direct_root)
    normalizer = _load_normalizer(args)
    retriever = KnowledgeRetriever(
        direct.graphs,
        normalizer=normalizer,
    )
    retrieval_hits = 0
    consistent_retrieval_hits = 0
    consistent_case_count = 0
    for case in direct.cases:
        retrieved_labels = {
            normalized_label_key(label, normalizer)
            for fact in retriever.retrieve(case, top_k=6).facts
            for label in (fact.diagnosis_label, *fact.diagnostic_path)
        }
        hit = normalized_label_key(case.gold_label, normalizer) in retrieved_labels
        retrieval_hits += hit
        if _is_graph_consistent(case):
            consistent_case_count += 1
            consistent_retrieval_hits += hit
    payload = {
        **direct.audit.to_dict(),
        "graph_node_count": sum(len(graph.nodes) for graph in direct.graphs),
        "graph_edge_count": sum(len(graph.edges) for graph in direct.graphs),
        "annotation_node_count": sum(
            len(case.annotation_nodes) for case in direct.cases
        ),
        "gold_observation_count": sum(
            len(case.gold_observations) for case in direct.cases
        ),
        "development_case_count": len(
            select_direct_partition(direct.cases, "development")
        ),
        "test_case_count": len(select_direct_partition(direct.cases, "test")),
        "retriever_id": retriever.retriever_id,
        "normalizer_id": (
            normalizer.normalizer_id
            if normalizer is not None
            else "lexical-v1"
        ),
        "top_6_gold_path_coverage": retrieval_hits / len(direct.cases),
        "top_6_graph_consistent_gold_path_coverage": (
            consistent_retrieval_hits / consistent_case_count
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


def _audit_medqa(args: argparse.Namespace) -> int:
    direct = load_direct_dataset(args.direct_root)
    normalizer = _load_normalizer(args)
    rows: dict[str, dict[str, int]] = {}
    for split in ("dev", "test"):
        diagnostic = load_medqa_cases(
            args.medqa_root,
            split=split,
            graphs=direct.graphs,
            diagnostic_only=True,
            normalizer=normalizer,
        )
        rows[split] = {
            "diagnostic_question_count": len(diagnostic),
            "gold_label_graph_covered_count": sum(
                bool(case.metadata.get("direct_graph_covered"))
                for case in diagnostic
            ),
        }
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


def _run_experiment(args: argparse.Namespace) -> int:
    direct = load_direct_dataset(args.direct_root)
    normalizer = _load_normalizer(args)
    cases = _stable_limit(
        _load_experiment_cases(args, direct, normalizer),
        args.limit,
        args.seed,
    )
    if not cases:
        raise ValueError("No cases matched the requested dataset filters.")
    model = ollama_model_from_env(
        model_name=args.model,
        seed=args.seed,
        allow_remote=args.allow_remote_data,
        timeout_seconds=args.timeout_seconds,
        context_window=args.context_window,
        max_output_tokens=args.max_output_tokens,
    )
    boundary_check = getattr(model, "assert_data_boundary", None)
    if callable(boundary_check):
        boundary_check(cases[0].dataset)
    runner = ExperimentRunner(
        model=model,
        reasoner_model=_load_agent_model(
            args,
            role="reasoner",
            base_model=model,
        ),
        verifier_model=_load_agent_model(
            args,
            role="verifier",
            base_model=model,
        ),
        retriever=KnowledgeRetriever(
            direct.graphs,
            normalizer=normalizer,
        ),
        cache_dir=args.cache_dir,
        top_k=args.top_k,
    )
    run = run_experiment(
        cases=cases,
        modes=_parse_modes(args.modes),
        runner=runner,
        output_dir=args.output_dir,
        run_name=args.run_name,
        fail_fast=not args.continue_on_error,
        show_progress=not args.no_progress,
    )
    artifacts = write_evaluation_artifacts(
        records=run.records,
        cases=cases,
        graphs=direct.graphs,
        output_dir=run.output_dir / "evaluation",
        normalizer=normalizer,
    )
    print(
        json.dumps(
            {
                "run_id": run.run_id,
                "case_count": len(cases),
                "prediction_count": len(run.records),
                "predictions": str(run.predictions_path),
                "argument_traces": (
                    str(run.argument_traces_path)
                    if run.argument_traces_path.exists()
                    else None
                ),
                "manifest": str(run.manifest_path),
                "evaluation": {
                    name: str(path) for name, path in artifacts.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _evaluate_run(args: argparse.Namespace) -> int:
    direct = load_direct_dataset(args.direct_root)
    normalizer = _load_normalizer(args)
    cases = _load_experiment_cases(args, direct, normalizer)
    records = load_prediction_records(args.predictions)
    output_dir = args.output_dir or args.predictions.parent / "evaluation"
    artifacts = write_evaluation_artifacts(
        records=records,
        cases=cases,
        graphs=direct.graphs,
        output_dir=output_dir,
        normalizer=normalizer,
    )
    print(
        json.dumps(
            {name: str(path) for name, path in artifacts.items()},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _diagnose(args: argparse.Namespace) -> int:
    direct = load_direct_dataset(args.direct_root)
    normalizer = _load_normalizer(args)
    case = load_patient_case(args.patient)
    model = ollama_model_from_env(
        model_name=args.model,
        seed=args.seed,
        allow_remote=args.allow_remote_data,
        timeout_seconds=args.timeout_seconds,
        context_window=args.context_window,
        max_output_tokens=args.max_output_tokens,
    )
    model.assert_data_boundary(case.dataset)
    runner = ExperimentRunner(
        model=model,
        reasoner_model=_load_agent_model(
            args,
            role="reasoner",
            base_model=model,
        ),
        verifier_model=_load_agent_model(
            args,
            role="verifier",
            base_model=model,
        ),
        retriever=KnowledgeRetriever(
            direct.graphs,
            normalizer=normalizer,
        ),
        cache_dir=args.cache_dir,
        top_k=args.top_k,
    )
    run = run_experiment(
        cases=(case,),
        modes=(ExperimentMode(args.mode),),
        runner=runner,
        output_dir=args.output_dir,
        show_progress=not args.no_progress,
    )
    record = run.records[0]
    print(
        json.dumps(
            {
                "run_id": run.run_id,
                "case_id": record.case_id,
                "mode": record.mode.value,
                "answer": record.predicted_label,
                "reasoning": record.reasoning,
                "citations": list(record.citations),
                "abstained": record.abstained,
                "error": record.error,
                "record_path": str(run.predictions_path),
                "argument_trace_path": (
                    str(run.argument_traces_path)
                    if run.argument_traces_path.exists()
                    else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if record.error else 0


def _run_perturbations(args: argparse.Namespace) -> int:
    direct = load_direct_dataset(args.direct_root)
    normalizer = _load_normalizer(args)
    cases = select_direct_partition(direct.cases, args.partition)
    if args.strict_direct:
        cases = tuple(case for case in cases if _is_graph_consistent(case))
    cases = _stable_limit(cases, args.limit, args.seed)
    pairs = build_section_removal_pairs(cases)
    if not pairs:
        raise ValueError("No cases support the requested section-removal analysis.")

    model = ollama_model_from_env(
        model_name=args.model,
        seed=args.seed,
        allow_remote=args.allow_remote_data,
        timeout_seconds=args.timeout_seconds,
        context_window=args.context_window,
        max_output_tokens=args.max_output_tokens,
    )
    model.assert_data_boundary(pairs[0].base_case.dataset)
    runner = ExperimentRunner(
        model=model,
        reasoner_model=_load_agent_model(
            args,
            role="reasoner",
            base_model=model,
        ),
        verifier_model=_load_agent_model(
            args,
            role="verifier",
            base_model=model,
        ),
        retriever=KnowledgeRetriever(
            direct.graphs,
            normalizer=normalizer,
        ),
        cache_dir=args.cache_dir,
        top_k=args.top_k,
    )
    paired_cases = tuple(
        case
        for pair in pairs
        for case in (pair.base_case, pair.perturbed_case)
    )
    run = run_experiment(
        cases=paired_cases,
        modes=(ExperimentMode(args.mode),),
        runner=runner,
        output_dir=args.output_dir,
        run_name=args.run_name,
        fail_fast=True,
        show_progress=not args.no_progress,
    )
    metrics = evaluate_perturbations(
        pairs,
        run.records,
        normalizer=normalizer,
    )
    artifacts = write_perturbation_artifacts(
        metrics,
        run.output_dir / "evaluation",
    )
    print(
        json.dumps(
            {
                "run_id": run.run_id,
                "pair_count": len(pairs),
                "prediction_count": len(run.records),
                "predictions": str(run.predictions_path),
                "argument_traces": (
                    str(run.argument_traces_path)
                    if run.argument_traces_path.exists()
                    else None
                ),
                "evaluation": {
                    name: str(path) for name, path in artifacts.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _plot_trace(args: argparse.Namespace) -> int:
    trace = load_argument_trace(
        args.traces,
        case_id=args.case_id,
        trace_id=args.trace_id,
    )
    case_id = str(trace.get("case_id") or "argument-trace")
    safe_case_id = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in case_id
    )
    output_path = args.output or (
        args.traces.parent
        / "evaluation"
        / f"argument_trace_{safe_case_id}.png"
    )
    rendered = plot_argument_trace(trace, output_path)
    print(
        json.dumps(
            {
                "case_id": case_id,
                "argument_trace_plot": str(rendered),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    if args.command == "prepare-direct":
        root = prepare_direct_archive(
            args.archive,
            args.output_dir,
            overwrite=args.overwrite,
        )
        print(json.dumps({"direct_root": str(root)}, indent=2))
        return 0
    if args.command == "build-umls":
        sources = tuple(
            value.strip()
            for value in args.sources.split(",")
            if value.strip()
        )
        languages = tuple(
            value.strip()
            for value in args.languages.split(",")
            if value.strip()
        )
        db_path = build_local_umls_database(
            LocalUMLSBuildConfig(
                meta_dir=args.meta_dir,
                db_path=args.db_path,
                source_vocabularies=sources,
                languages=languages,
                batch_size=args.batch_size,
            )
        )
        print(
            json.dumps(
                {
                    "umls_db": str(db_path),
                    "sources": list(sources),
                    "languages": list(languages),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "audit-direct":
        return _audit_direct(args)
    if args.command == "audit-medqa":
        return _audit_medqa(args)
    if args.command == "run-experiment":
        return _run_experiment(args)
    if args.command == "evaluate-run":
        return _evaluate_run(args)
    if args.command == "run-perturbations":
        return _run_perturbations(args)
    if args.command == "diagnose":
        return _diagnose(args)
    if args.command == "plot-trace":
        return _plot_trace(args)
    raise ValueError(f"Unknown command: {args.command}")
