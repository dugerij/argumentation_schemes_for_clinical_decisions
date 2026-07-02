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

from eval.medqa_smoke import extract_answer, format_options, load_questions
from eval.question_subset import filter_questions_by_note_overlap, load_questions_jsonl
from helpers.config import env_bool, parse_optional_int, startup_check
from helpers.jsonl import JsonlLogger
from helpers.paths import EMBEDDING_BENCHMARK_LOG_PATH
from helpers.progress import iter_progress, progress_message
from ingest.mimic import MimicDischargeSubsetConfig, extract_mimic_discharge_subset
from retrieval.index import ensure_index
from retrieval.query import query_index_with_diagnostics


DEFAULT_EVENT_LOG = EMBEDDING_BENCHMARK_LOG_PATH
DEFAULT_DATA_CANDIDATES = (
    Path("test.jsonl"),
    Path("data/medqa/data_clean/questions/US/test.jsonl"),
    Path("data/eval/test.jsonl"),
)


@dataclass(frozen=True)
class EmbeddingBenchmarkConfig:
    input_dir: Path
    output_root: Path
    generation_model: str
    embedding_models: tuple[str, ...]
    index_root: Path | None = None
    use_umls: bool = False
    schema_guided: bool = False
    note_limit: int | None = 25
    note_max_chars: int | None = 3000
    note_type: str = "DS"
    mimic_csv: Path | None = None
    questions_path: Path | None = None
    sample_size: int = 5
    render_graphs: bool = False
    require_question_note_overlap: bool = True
    min_overlap_terms: int = 2


@dataclass(frozen=True)
class EmbeddingBenchmarkResult:
    embedding_model: str
    index_dir: str
    question_count: int
    build_seconds: float
    total_query_seconds: float
    mean_query_seconds: float
    exact_match: float | None
    mean_top_retrieval_score: float | None
    mean_retrieval_score: float | None
    mean_retrieval_score_margin: float | None
    mean_top_cosine_similarity: float | None
    mean_cosine_similarity: float | None
    mean_cosine_similarity_margin: float | None
    use_umls: bool
    schema_guided: bool
    generation_model: str
    provider: str = "ollama"


def model_slug(value: object | None) -> str:
    text = "unknown" if value is None else str(value)
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text).strip("_") or "unknown"


def discover_questions_path(explicit_path: Path | None = None) -> Path | None:
    if explicit_path is not None:
        return explicit_path if explicit_path.exists() else None
    for candidate in DEFAULT_DATA_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def load_benchmark_questions(path: Path | None, sample_size: int) -> list[dict[str, Any]]:
    if path is None:
        return [
            {
                "id": f"demo-{index}",
                "question": "What is the preferred next step for a patient with chronic kidney disease and uncontrolled hypertension?",
                "answer": "A",
                "options": {
                    "A": "Optimize blood pressure control and renal protection",
                    "B": "Stop all medication",
                    "C": "Ignore the blood pressure",
                },
            }
            for index in range(1, sample_size + 1)
        ]
    return load_questions(path, sample_size=sample_size)


def load_all_questions(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return load_benchmark_questions(path, sample_size=100)
    return load_questions_jsonl(path)


def prepare_mimic_subset(
    *,
    csv_path: Path | None,
    output_dir: Path,
    note_limit: int | None,
    note_max_chars: int | None,
    note_type: str,
) -> None:
    if csv_path is None:
        return
    extract_mimic_discharge_subset(
        MimicDischargeSubsetConfig(
            csv_path=csv_path,
            output_dir=output_dir,
            limit=note_limit,
            note_type=note_type,
            max_chars=note_max_chars,
            overwrite=env_bool("BENCHMARK_OVERWRITE_NOTES", True),
        )
    )


async def run_embedding_benchmark(
    config: EmbeddingBenchmarkConfig,
    *,
    event_log_path: Path = DEFAULT_EVENT_LOG,
) -> list[EmbeddingBenchmarkResult]:
    run_id = f"embedding_benchmark_{int(time.time())}"
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
        "embedding_benchmark",
        "started",
        provider="ollama",
        generation_model=config.generation_model,
        embedding_models=list(config.embedding_models),
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
    os.environ["INDEX_LLM_MODEL"] = config.generation_model

    results: list[EmbeddingBenchmarkResult] = []
    progress_message(
        f"Running embedding benchmark for {len(config.embedding_models)} embedding model(s) over {len(questions)} question(s)"
    )
    index_root = config.index_root or config.output_root
    index_root.mkdir(parents=True, exist_ok=True)
    for embedding_model in iter_progress(
        config.embedding_models,
        desc="Embedding models",
        total=len(config.embedding_models),
        unit="model",
    ):
        os.environ["INDEX_EMBEDDING_MODEL"] = embedding_model

        slug = model_slug(embedding_model)
        index_dir = index_root / slug
        logger.event("embedding_model", "started", embedding_model=embedding_model, index_dir=index_dir)
        progress_message(f"Building index for embedding model `{embedding_model}`")

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
        top_retrieval_scores: list[float] = []
        mean_retrieval_scores: list[float] = []
        retrieval_score_margins: list[float] = []
        top_cosine_similarities: list[float] = []
        mean_cosine_similarities: list[float] = []
        cosine_similarity_margins: list[float] = []
        for item in iter_progress(
            questions,
            desc=f"Questions [{embedding_model}]",
            total=len(questions),
            unit="question",
        ):
            prompt = item["question"]
            options = item.get("options") or {}
            if options:
                prompt = f"{prompt}\n\n{format_options(options)}"

            query_started = time.perf_counter()
            response, _context, diagnostics = await query_index_with_diagnostics(index=index, query=prompt)
            query_durations.append(time.perf_counter() - query_started)
            if diagnostics.get("top_retrieval_score") is not None:
                top_retrieval_scores.append(float(diagnostics["top_retrieval_score"]))
            if diagnostics.get("mean_retrieval_score") is not None:
                mean_retrieval_scores.append(float(diagnostics["mean_retrieval_score"]))
            if diagnostics.get("retrieval_score_margin") is not None:
                retrieval_score_margins.append(float(diagnostics["retrieval_score_margin"]))
            if diagnostics.get("top_cosine_similarity") is not None:
                top_cosine_similarities.append(float(diagnostics["top_cosine_similarity"]))
            if diagnostics.get("mean_cosine_similarity") is not None:
                mean_cosine_similarities.append(float(diagnostics["mean_cosine_similarity"]))
            if diagnostics.get("cosine_similarity_margin") is not None:
                cosine_similarity_margins.append(float(diagnostics["cosine_similarity_margin"]))

            if options and item.get("answer"):
                predicted = extract_answer(response, options)
                exact_matches.append(float(predicted == item["answer"]))

        total_query_seconds = round(sum(query_durations), 2)
        mean_query_seconds = round(total_query_seconds / max(1, len(query_durations)), 2)
        exact_match = round(sum(exact_matches) / len(exact_matches), 4) if exact_matches else None
        result = EmbeddingBenchmarkResult(
            embedding_model=embedding_model,
            index_dir=index_dir.as_posix(),
            question_count=len(questions),
            build_seconds=build_seconds,
            total_query_seconds=total_query_seconds,
            mean_query_seconds=mean_query_seconds,
            exact_match=exact_match,
            mean_top_retrieval_score=round(sum(top_retrieval_scores) / len(top_retrieval_scores), 4) if top_retrieval_scores else None,
            mean_retrieval_score=round(sum(mean_retrieval_scores) / len(mean_retrieval_scores), 4) if mean_retrieval_scores else None,
            mean_retrieval_score_margin=round(sum(retrieval_score_margins) / len(retrieval_score_margins), 4) if retrieval_score_margins else None,
            mean_top_cosine_similarity=round(sum(top_cosine_similarities) / len(top_cosine_similarities), 4) if top_cosine_similarities else None,
            mean_cosine_similarity=round(sum(mean_cosine_similarities) / len(mean_cosine_similarities), 4) if mean_cosine_similarities else None,
            mean_cosine_similarity_margin=round(sum(cosine_similarity_margins) / len(cosine_similarity_margins), 4) if cosine_similarity_margins else None,
            use_umls=config.use_umls,
            schema_guided=config.schema_guided,
            provider="ollama",
            generation_model=config.generation_model,
        )
        results.append(result)
        logger.event("embedding_model", "completed", **asdict(result))
        progress_message(
            f"Completed `{embedding_model}`: build={build_seconds}s, mean_query={mean_query_seconds}s, exact_match={exact_match}"
        )

    summary_path = config.output_root / "embedding_benchmark_results.json"
    summary_path.write_text(
        json.dumps([asdict(result) for result in results], indent=2),
        encoding="utf-8",
    )
    logger.event("embedding_benchmark", "completed", summary_path=summary_path, result_count=len(results))
    progress_message(f"Embedding benchmark summary written to {summary_path}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark embedding models for the clinical RAG pipeline.")
    parser.add_argument("--generation-model", required=True, help="LLM used for graph extraction/querying.")
    parser.add_argument("--embedding-model", action="append", dest="embedding_models", required=True)
    parser.add_argument("--input-dir", default=os.environ.get("INPUT_BASE_DIR", "data/evidence/mimic_discharge_subset"))
    parser.add_argument("--output-root", default="output/embedding_benchmark")
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
        run_embedding_benchmark(
            EmbeddingBenchmarkConfig(
                input_dir=Path(args.input_dir),
                output_root=Path(args.output_root),
                generation_model=args.generation_model,
                embedding_models=tuple(args.embedding_models),
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
