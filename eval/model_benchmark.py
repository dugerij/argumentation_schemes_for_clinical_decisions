from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from eval.embedding_benchmark import (
    DEFAULT_DATA_CANDIDATES,
    discover_questions_path,
    load_benchmark_questions,
    model_slug,
    prepare_mimic_subset,
)
from eval.medqa_smoke import extract_answer, format_options
from helpers.config import env_bool, parse_optional_int, startup_check
from helpers.jsonl import JsonlLogger
from rag.index import ensure_index
from rag.retrieve import query_index_context


DEFAULT_EVENT_LOG = Path("logs/framework/model_benchmark.jsonl")


@dataclass(frozen=True)
class ModelBenchmarkConfig:
    input_dir: Path
    output_root: Path
    generation_models: tuple[str, ...]
    embedding_model: str
    provider: str = "vllm"
    use_umls: bool = False
    schema_guided: bool = False
    note_limit: int | None = 25
    note_max_chars: int | None = 3000
    note_type: str = "DS"
    mimic_csv: Path | None = None
    questions_path: Path | None = None
    sample_size: int = 5


@dataclass(frozen=True)
class ModelBenchmarkResult:
    generation_model: str
    embedding_model: str
    index_dir: str
    question_count: int
    build_seconds: float
    total_query_seconds: float
    mean_query_seconds: float
    exact_match: float | None
    use_umls: bool
    schema_guided: bool
    provider: str


async def run_model_benchmark(
    config: ModelBenchmarkConfig,
    *,
    event_log_path: Path = DEFAULT_EVENT_LOG,
) -> list[ModelBenchmarkResult]:
    run_id = f"model_benchmark_{int(time.time())}"
    logger = JsonlLogger(event_log_path, run_id=run_id)

    config.output_root.mkdir(parents=True, exist_ok=True)
    prepare_mimic_subset(
        csv_path=config.mimic_csv,
        output_dir=config.input_dir,
        note_limit=config.note_limit,
        note_max_chars=config.note_max_chars,
        note_type=config.note_type,
    )
    questions_path = discover_questions_path(config.questions_path)
    questions = load_benchmark_questions(questions_path, sample_size=config.sample_size)

    logger.event(
        "model_benchmark",
        "started",
        provider=config.provider,
        generation_models=list(config.generation_models),
        embedding_model=config.embedding_model,
        use_umls=config.use_umls,
        schema_guided=config.schema_guided,
        question_count=len(questions),
        input_dir=config.input_dir,
        output_root=config.output_root,
        questions_path=questions_path,
    )

    os.environ["INDEX_LLM_PROVIDER"] = config.provider
    os.environ["INDEX_EMBEDDING_PROVIDER"] = config.provider
    os.environ["INDEX_EMBEDDING_MODEL"] = config.embedding_model
    if config.provider in {"vllm", "vllm_offline"}:
        os.environ["VLLM_EMBEDDING_MODEL"] = config.embedding_model

    results: list[ModelBenchmarkResult] = []
    for generation_model in config.generation_models:
        os.environ["INDEX_LLM_MODEL"] = generation_model
        if config.provider in {"vllm", "vllm_offline"}:
            os.environ["VLLM_MODEL"] = generation_model

        slug = model_slug(generation_model)
        index_dir = config.output_root / slug
        logger.event("generation_model", "started", generation_model=generation_model, index_dir=index_dir)

        build_started = time.perf_counter()
        index = await asyncio.to_thread(
            ensure_index,
            input_dir=config.input_dir,
            output_dir=index_dir,
            use_umls=config.use_umls,
            schema_guided=config.schema_guided,
        )
        build_seconds = round(time.perf_counter() - build_started, 2)

        exact_matches: list[float] = []
        query_durations: list[float] = []
        for item in questions:
            prompt = item["question"]
            options = item.get("options") or {}
            if options:
                prompt = f"{prompt}\n\n{format_options(options)}"

            query_started = time.perf_counter()
            response, _context = await query_index_context(index=index, query=prompt)
            query_durations.append(time.perf_counter() - query_started)

            if options and item.get("answer"):
                predicted = extract_answer(response, options)
                exact_matches.append(float(predicted == item["answer"]))

        total_query_seconds = round(sum(query_durations), 2)
        mean_query_seconds = round(total_query_seconds / max(1, len(query_durations)), 2)
        exact_match = round(sum(exact_matches) / len(exact_matches), 4) if exact_matches else None
        result = ModelBenchmarkResult(
            generation_model=generation_model,
            embedding_model=config.embedding_model,
            index_dir=index_dir.as_posix(),
            question_count=len(questions),
            build_seconds=build_seconds,
            total_query_seconds=total_query_seconds,
            mean_query_seconds=mean_query_seconds,
            exact_match=exact_match,
            use_umls=config.use_umls,
            schema_guided=config.schema_guided,
            provider=config.provider,
        )
        results.append(result)
        logger.event("generation_model", "completed", **asdict(result))

    summary_path = config.output_root / "model_benchmark_results.json"
    summary_path.write_text(
        json.dumps([asdict(result) for result in results], indent=2),
        encoding="utf-8",
    )
    logger.event("model_benchmark", "completed", summary_path=summary_path, result_count=len(results))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark generation models for the clinical RAG pipeline.")
    parser.add_argument("--provider", default="vllm", choices=("vllm", "vllm_offline", "ollama", "openai"))
    parser.add_argument("--generation-model", action="append", dest="generation_models", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--input-dir", default=os.environ.get("INPUT_BASE_DIR", "data/evidence/mimic_discharge_subset"))
    parser.add_argument("--output-root", default="output/model_benchmark")
    parser.add_argument("--questions-path", default=None)
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--use-umls", action="store_true")
    parser.add_argument("--schema-guided", action="store_true")
    parser.add_argument("--mimic-csv", default=os.environ.get("MIMIC_DISCHARGE_CSV"))
    parser.add_argument("--note-limit", default=os.environ.get("MIMIC_DISCHARGE_LIMIT", "25"))
    parser.add_argument("--note-max-chars", default=os.environ.get("MIMIC_DISCHARGE_MAX_CHARS", "3000"))
    parser.add_argument("--note-type", default=os.environ.get("MIMIC_DISCHARGE_NOTE_TYPE", "DS"))
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    startup_check()
    args = parse_args()
    results = asyncio.run(
        run_model_benchmark(
            ModelBenchmarkConfig(
                input_dir=Path(args.input_dir),
                output_root=Path(args.output_root),
                generation_models=tuple(args.generation_models),
                embedding_model=args.embedding_model,
                provider=args.provider,
                use_umls=args.use_umls,
                schema_guided=args.schema_guided,
                note_limit=parse_optional_int(args.note_limit, default=25),
                note_max_chars=parse_optional_int(args.note_max_chars, default=3000),
                note_type=args.note_type,
                mimic_csv=(Path(args.mimic_csv) if args.mimic_csv else None),
                questions_path=(Path(args.questions_path) if args.questions_path else None),
                sample_size=args.sample_size,
            )
        )
    )
    print(json.dumps([asdict(result) for result in results], indent=2))


if __name__ == "__main__":
    main()
