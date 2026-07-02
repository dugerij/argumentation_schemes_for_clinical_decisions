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
    discover_questions_path,
    filter_questions_by_note_overlap,
    load_all_questions,
    load_benchmark_questions,
    model_slug,
    prepare_mimic_subset,
)
from eval.medqa_smoke import extract_answer, format_options
from helpers.config import env_bool, parse_optional_int, startup_check
from helpers.jsonl import JsonlLogger
from helpers.paths import MODEL_BENCHMARK_LOG_PATH
from helpers.progress import iter_progress, progress_message
from retrieval.index import ensure_index
from retrieval.query import query_index_context


DEFAULT_EVENT_LOG = MODEL_BENCHMARK_LOG_PATH


@dataclass(frozen=True)
class ModelBenchmarkConfig:
    input_dir: Path
    output_root: Path
    generation_models: tuple[str, ...]
    embedding_model: str
    index_root: Path | None = None
    use_umls: bool = False
    schema_guided: bool = False
    note_limit: int | None = 25
    note_max_chars: int | None = 3000
    note_type: str = "DS"
    mimic_csv: Path | None = None
    questions_path: Path | None = None
    sample_size: int = 5
    require_question_note_overlap: bool = True
    min_overlap_terms: int = 2


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
    provider: str = "ollama"


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
    raw_questions = (
        load_all_questions(questions_path)
        if config.require_question_note_overlap
        else load_benchmark_questions(questions_path, sample_size=config.sample_size)
    )
    if config.require_question_note_overlap:
        questions, question_overlap_details = filter_questions_by_note_overlap(
            questions=raw_questions,
            input_dir=config.input_dir,
            max_questions=config.sample_size,
            min_overlap_terms=config.min_overlap_terms,
        )
        if not questions:
            raise ValueError(
                f"No benchmark questions overlapped with notes in {config.input_dir} using min_overlap_terms={config.min_overlap_terms}."
            )
    else:
        questions = raw_questions[: config.sample_size]
        question_overlap_details = []

    logger.event(
        "model_benchmark",
        "started",
        provider="ollama",
        generation_models=list(config.generation_models),
        embedding_model=config.embedding_model,
        use_umls=config.use_umls,
        schema_guided=config.schema_guided,
        question_count=len(questions),
        raw_question_count=len(raw_questions),
        input_dir=config.input_dir,
        output_root=config.output_root,
        questions_path=questions_path,
        require_question_note_overlap=config.require_question_note_overlap,
        min_overlap_terms=config.min_overlap_terms,
    )
    if question_overlap_details:
        overlap_path = config.output_root / "selected_questions.json"
        overlap_path.write_text(
            json.dumps(question_overlap_details, indent=2),
            encoding="utf-8",
        )
        logger.event("question_filter", "completed", selected_count=len(questions), summary_path=overlap_path)

    os.environ["INDEX_LLM_PROVIDER"] = "ollama"
    os.environ["INDEX_EMBEDDING_PROVIDER"] = "ollama"
    os.environ["INDEX_EMBEDDING_MODEL"] = config.embedding_model

    index_root = config.index_root or config.output_root
    index_root.mkdir(parents=True, exist_ok=True)
    shared_index = None
    shared_index_dir = index_root / model_slug(config.embedding_model)
    shared_build_seconds = 0.0
    if not config.schema_guided:
        build_started = time.perf_counter()
        shared_index = await asyncio.to_thread(
            ensure_index,
            input_dir=config.input_dir,
            output_dir=shared_index_dir,
            use_umls=config.use_umls,
            schema_guided=config.schema_guided,
        )
        shared_build_seconds = round(time.perf_counter() - build_started, 2)

    results: list[ModelBenchmarkResult] = []
    progress_message(
        f"Running model benchmark for {len(config.generation_models)} generation model(s) over {len(questions)} question(s)"
    )
    for generation_model in iter_progress(
        config.generation_models,
        desc="Generation models",
        total=len(config.generation_models),
        unit="model",
    ):
        os.environ["INDEX_LLM_MODEL"] = generation_model

        build_seconds = shared_build_seconds
        if config.schema_guided:
            index_dir = index_root / model_slug(generation_model)
            progress_message(f"Building index for generation model `{generation_model}`")
            build_started = time.perf_counter()
            index = await asyncio.to_thread(
                ensure_index,
                input_dir=config.input_dir,
                output_dir=index_dir,
                use_umls=config.use_umls,
                schema_guided=config.schema_guided,
            )
            build_seconds = round(time.perf_counter() - build_started, 2)
        else:
            index_dir = shared_index_dir
            index = shared_index
        logger.event("generation_model", "started", generation_model=generation_model, index_dir=index_dir)
        if not config.schema_guided:
            progress_message(f"Querying shared index with generation model `{generation_model}`")

        exact_matches: list[float] = []
        query_durations: list[float] = []
        for item in iter_progress(
            questions,
            desc=f"Questions [{generation_model}]",
            total=len(questions),
            unit="question",
        ):
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
            provider="ollama",
        )
        results.append(result)
        logger.event("generation_model", "completed", **asdict(result))
        progress_message(
            f"Completed `{generation_model}`: build={build_seconds}s, mean_query={mean_query_seconds}s, exact_match={exact_match}"
        )

    summary_path = config.output_root / "model_benchmark_results.json"
    summary_path.write_text(
        json.dumps([asdict(result) for result in results], indent=2),
        encoding="utf-8",
    )
    logger.event("model_benchmark", "completed", summary_path=summary_path, result_count=len(results))
    progress_message(f"Model benchmark summary written to {summary_path}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark generation models for the clinical RAG pipeline.")
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
