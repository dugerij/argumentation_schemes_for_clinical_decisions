import argparse
import asyncio
import os
import random
from pathlib import Path

from dotenv import load_dotenv

from api.recommendation import RecommendationRequest, generate_recommendation
from eval.embedding_benchmark import EmbeddingBenchmarkConfig, run_embedding_benchmark
from eval.medqa_smoke import run_medqa_smoke_eval
from eval.model_benchmark import ModelBenchmarkConfig, run_model_benchmark
from eval.pipeline_check import run_pipeline_check
from helpers.config import parse_optional_int, startup_check


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clinical argumentation pipeline entrypoint.")
    subparsers = parser.add_subparsers(dest="command", required=False)

    subparsers.add_parser("smoke-eval", help="Run the MedQA smoke evaluation.")

    serve_api = subparsers.add_parser("serve-api", help="Start the FastAPI server.")
    serve_api.add_argument("--host", default="10.0.0.201")
    serve_api.add_argument("--port", type=int, default=8000)
    serve_api.add_argument("--reload", action="store_true")

    recommend = subparsers.add_parser("recommend", help="Run a single recommendation request.")
    recommend.add_argument("--scenario", required=True)
    recommend.add_argument("--clinical-goal", default=None)
    recommend.add_argument("--patient-id", default=None)
    recommend.add_argument("--max-rounds", type=int, default=3)
    recommend.add_argument("--dry-run", action="store_true")
    recommend.add_argument("--no-rag", action="store_true")

    benchmark = subparsers.add_parser("benchmark-embeddings", help="Benchmark embedding models.")
    benchmark.add_argument("--generation-model", required=True)
    benchmark.add_argument("--embedding-model", action="append", dest="embedding_models", required=True)
    benchmark.add_argument("--output-root", default="output/embedding_benchmark")
    benchmark.add_argument("--sample-size", type=int, default=5)
    benchmark.add_argument("--questions-path", default=None)
    benchmark.add_argument("--use-umls", action="store_true")
    benchmark.add_argument("--schema-guided", action="store_true")
    benchmark.add_argument("--mimic-csv", default=os.environ.get("MIMIC_DISCHARGE_CSV"))
    benchmark.add_argument("--note-limit", default=os.environ.get("MIMIC_DISCHARGE_LIMIT", "25"))
    benchmark.add_argument("--note-max-chars", default=os.environ.get("MIMIC_DISCHARGE_MAX_CHARS", "all"))

    model_benchmark = subparsers.add_parser("benchmark-models", help="Benchmark generation models with a fixed embedding model.")
    model_benchmark.add_argument("--generation-model", action="append", dest="generation_models", required=True)
    model_benchmark.add_argument("--embedding-model", required=True)
    model_benchmark.add_argument("--output-root", default="output/model_benchmark")
    model_benchmark.add_argument("--sample-size", type=int, default=5)
    model_benchmark.add_argument("--questions-path", default=None)
    model_benchmark.add_argument("--use-umls", action="store_true")
    model_benchmark.add_argument("--schema-guided", action="store_true")
    model_benchmark.add_argument("--mimic-csv", default=os.environ.get("MIMIC_DISCHARGE_CSV"))
    model_benchmark.add_argument("--note-limit", default=os.environ.get("MIMIC_DISCHARGE_LIMIT", "25"))
    model_benchmark.add_argument("--note-max-chars", default=os.environ.get("MIMIC_DISCHARGE_MAX_CHARS", "all"))

    pipeline = subparsers.add_parser("pipeline-check", help="Run an end-to-end pipeline smoke check.")
    pipeline.add_argument("--scenario", required=True)
    pipeline.add_argument("--clinical-goal", default=None)
    pipeline.add_argument("--dry-run", action="store_true")
    pipeline.add_argument("--use-umls", action="store_true")
    pipeline.add_argument("--schema-guided", action="store_true")

    return parser.parse_args()


def main() -> None:
    load_dotenv()
    startup_check()
    args = parse_args()

    command = args.command or "smoke-eval"

    random.seed(int(os.environ.get("RANDOM_SEED", "42")))
    sample_size = int(os.environ.get("EVAL_SAMPLE_SIZE", "1"))
    input_dir = Path(os.environ["INPUT_BASE_DIR"])
    output_dir = Path(os.environ["OUTPUT_BASE_DIR"])

    if command == "smoke-eval":
        asyncio.run(
            run_medqa_smoke_eval(
                input_dir=input_dir,
                output_dir=output_dir,
                sample_size=sample_size,
            )
        )
        return

    if command == "serve-api":
        import uvicorn

        uvicorn.run("api.app:app", host=args.host, port=args.port, reload=args.reload)
        return

    if command == "recommend":
        result = asyncio.run(
            generate_recommendation(
                RecommendationRequest(
                    scenario=args.scenario,
                    clinical_goal=args.clinical_goal,
                    patient_id=args.patient_id,
                    max_rounds=args.max_rounds,
                    use_rag=not args.no_rag,
                    dry_run=args.dry_run,
                )
            )
        )
        print(result.model_dump_json(indent=2))
        return

    if command == "benchmark-embeddings":
        results = asyncio.run(
            run_embedding_benchmark(
                EmbeddingBenchmarkConfig(
                    input_dir=input_dir,
                    output_root=Path(args.output_root),
                    generation_model=args.generation_model,
                    embedding_models=tuple(args.embedding_models),
                    use_umls=args.use_umls,
                    schema_guided=args.schema_guided,
                    mimic_csv=(Path(args.mimic_csv) if args.mimic_csv else None),
                    questions_path=(Path(args.questions_path) if args.questions_path else None),
                    sample_size=args.sample_size,
                    note_limit=parse_optional_int(args.note_limit, default=25),
                    note_max_chars=parse_optional_int(args.note_max_chars, default=None),
                )
            )
        )
        for result in results:
            print(result)
        return

    if command == "benchmark-models":
        results = asyncio.run(
            run_model_benchmark(
                ModelBenchmarkConfig(
                    input_dir=input_dir,
                    output_root=Path(args.output_root),
                    generation_models=tuple(args.generation_models),
                    embedding_model=args.embedding_model,
                    use_umls=args.use_umls,
                    schema_guided=args.schema_guided,
                    mimic_csv=(Path(args.mimic_csv) if args.mimic_csv else None),
                    questions_path=(Path(args.questions_path) if args.questions_path else None),
                    sample_size=args.sample_size,
                    note_limit=parse_optional_int(args.note_limit, default=25),
                    note_max_chars=parse_optional_int(args.note_max_chars, default=None),
                )
            )
        )
        for result in results:
            print(result)
        return

    if command == "pipeline-check":
        result = asyncio.run(
            run_pipeline_check(
                input_dir=input_dir,
                output_dir=output_dir,
                scenario=args.scenario,
                clinical_goal=args.clinical_goal,
                use_umls=args.use_umls,
                schema_guided=args.schema_guided,
                dry_run=args.dry_run,
            )
        )
        print(result)
        return

    raise ValueError(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
