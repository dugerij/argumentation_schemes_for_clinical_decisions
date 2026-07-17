import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from api.recommendation import RecommendationRequest, generate_recommendation
from eval.case_eval import run_case_eval
from helpers.config import startup_check
from retrieval.cds_graph import build_cds_materialized_graph
from retrieval.cds_query import SUPPORTED_CDS_TASKS


SUPPORTED_DATASET_FORMATS = ("mimic_ext_cds",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Graph-backed clinical argumentation pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    materialize_graph = subparsers.add_parser(
        "materialize-graph",
        help="Build a materialized case graph from a supported note-like dataset.",
    )
    materialize_graph.add_argument("--input-path", required=True)
    materialize_graph.add_argument("--dataset-format", default="mimic_ext_cds", choices=SUPPORTED_DATASET_FORMATS)
    materialize_graph.add_argument("--output-dir", default=None)
    materialize_graph.add_argument("--no-overwrite", action="store_true")

    answer_question = subparsers.add_parser(
        "answer-question",
        help="Answer one case question using the materialized graph and argumentation framework.",
    )
    answer_question.add_argument("--task", required=True, choices=SUPPORTED_CDS_TASKS)
    answer_question.add_argument("--case-id", required=True, type=int)
    answer_question.add_argument("--output-dir", default=None)
    answer_question.add_argument("--question", default=None)
    answer_question.add_argument("--clinical-goal", default=None)
    answer_question.add_argument("--max-rounds", type=int, default=3)
    answer_question.add_argument("--top-k-cases", type=int, default=5)
    answer_question.add_argument("--dry-run", action="store_true")

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Run a small evaluation over cases in the materialized graph.",
    )
    evaluate.add_argument("--task", required=True, choices=SUPPORTED_CDS_TASKS)
    evaluate.add_argument("--output-dir", default=None)
    evaluate.add_argument("--sample-size", type=int, default=5)
    evaluate.add_argument("--max-rounds", type=int, default=3)
    evaluate.add_argument("--top-k-cases", type=int, default=5)

    serve_api = subparsers.add_parser("serve-api", help="Start the FastAPI server.")
    serve_api.add_argument("--host", default="127.0.0.1")
    serve_api.add_argument("--port", type=int, default=8000)
    serve_api.add_argument("--reload", action="store_true")

    return parser.parse_args()


def _output_dir_from_args(value: str | None) -> Path:
    return Path(value or os.environ.get("OUTPUT_BASE_DIR", "output"))


def _require_supported_dataset_format(dataset_format: str) -> None:
    if dataset_format not in SUPPORTED_DATASET_FORMATS:
        raise ValueError(f"Unsupported dataset format: {dataset_format}")


def main() -> None:
    load_dotenv()
    args = parse_args()

    if args.command == "materialize-graph":
        startup_check(require_models=False, require_paths=False)
        _require_supported_dataset_format(args.dataset_format)
        output_dir = _output_dir_from_args(args.output_dir)
        stats = build_cds_materialized_graph(
            zip_path=Path(args.input_path),
            output_dir=output_dir,
            overwrite=not args.no_overwrite,
        )
        print(
            json.dumps(
                {
                    "artifact_path": stats.artifact_path,
                    "manifest_path": stats.manifest_path,
                    "case_count": stats.case_count,
                    "diagnosis_case_count": stats.diagnosis_case_count,
                    "triage_case_count": stats.triage_case_count,
                    "specialty_case_count": stats.specialty_case_count,
                    "token_count": stats.token_count,
                    "build_seconds": stats.build_seconds,
                    "artifact_bytes": stats.artifact_bytes,
                },
                indent=2,
            )
        )
        return

    if args.command == "serve-api":
        startup_check(require_models=True, require_paths=False)
        import uvicorn

        uvicorn.run("api.app:app", host=args.host, port=args.port, reload=args.reload)
        return

    startup_check(require_models=True, require_paths=False)
    output_dir = _output_dir_from_args(getattr(args, "output_dir", None))
    os.environ["OUTPUT_BASE_DIR"] = str(output_dir)

    if args.command == "answer-question":
        result = asyncio.run(
            generate_recommendation(
                RecommendationRequest(
                    scenario=args.question,
                    case_id=args.case_id,
                    task=args.task,
                    clinical_goal=args.clinical_goal,
                    max_rounds=args.max_rounds,
                    top_k_cases=args.top_k_cases,
                    dry_run=args.dry_run,
                )
            )
        )
        print(result.model_dump_json(indent=2))
        return

    if args.command == "evaluate":
        summary = asyncio.run(
            run_case_eval(
                output_dir=output_dir,
                task=args.task,
                sample_size=args.sample_size,
                max_rounds=args.max_rounds,
                top_k_cases=args.top_k_cases,
            )
        )
        print(
            json.dumps(
                {
                    "task": summary.task,
                    "sample_size": summary.sample_size,
                    "correct_count": summary.correct_count,
                    "accuracy": summary.accuracy,
                    "records": [
                        {
                            "case_id": record.case_id,
                            "expected_answer": record.expected_answer,
                            "predicted_answer": record.predicted_answer,
                            "correct": record.correct,
                        }
                        for record in summary.records
                    ],
                },
                indent=2,
            )
        )
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
