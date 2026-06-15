import json
import random
import re
from pathlib import Path

from helpers.jsonl import JsonlLogger
from helpers.records import write_eval_record
from rag.index import ensure_index, load_graph_tables
from rag.retrieve import drift_search_context


DEFAULT_DATA_PATH = Path("data/medqa/data_clean/questions/US/4_options/phrases_no_exclude_train.jsonl")
DEFAULT_EVENT_LOG = Path("logs/framework/events.jsonl")
DEFAULT_EVAL_LOG = Path("logs/framework/eval_records.jsonl")


def format_options(options: dict[str, str]) -> str:
    return "\n".join(f"{letter}. {text}" for letter, text in options.items())


def load_questions(path: Path, sample_size: int) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]

    if sample_size > len(rows):
        raise ValueError(f"Requested {sample_size} questions, but only {len(rows)} are available.")

    return random.sample(rows, sample_size)


def extract_answer(response: str, options: dict[str, str]) -> str | None:
    response_lower = response.lower()

    for letter, text in options.items():
        if text.lower() in response_lower:
            return text
        if re.search(rf"\b{re.escape(letter.lower())}\b", response_lower):
            return text

    return None


async def run_medqa_smoke_eval(
    config,
    output_dir: Path,
    sample_size: int,
    index_method: str,
    data_path: Path = DEFAULT_DATA_PATH,
    event_log_path: Path = DEFAULT_EVENT_LOG,
    eval_log_path: Path = DEFAULT_EVAL_LOG,
) -> None:
    event_logger = JsonlLogger(event_log_path, run_id="medqa_smoke")
    event_logger.event(
        "medqa_smoke_eval",
        "started",
        data_path=data_path,
        output_dir=output_dir,
        sample_size=sample_size,
        index_method=index_method,
        eval_log_path=eval_log_path,
    )

    with event_logger.timed("ensure_graphrag_index", output_dir=output_dir, index_method=index_method):
        await ensure_index(config, output_dir, index_method)

    with event_logger.timed("load_graph_tables", output_dir=output_dir):
        entities, communities, community_reports = load_graph_tables(output_dir)
    print("Graph loaded.")

    with event_logger.timed("load_medqa_questions", data_path=data_path, sample_size=sample_size):
        questions = load_questions(data_path, sample_size)
    correct = 0

    for index, data in enumerate(questions, start=1):
        options = data["options"]
        expected_answer = data["answer"]
        question = f"{data['question']}\n{format_options(options)}"
        query = f"{question}\n\nAnswer with ONLY the correct option exactly as written."

        question_id = str(data.get("id") or data.get("meta_info") or f"medqa_{index}")
        event_logger.event("question", "started", question_id=question_id, question_index=index)

        try:
            with event_logger.timed("drift_search", question_id=question_id, question_index=index):
                response, context = await drift_search_context(
                    config=config,
                    query=query,
                    entities=entities,
                    communities=communities,
                    community_reports=community_reports,
                    response_type="Single Sentence",
                )
        except Exception as exc:
            write_eval_record(
                eval_log_path,
                run_id=event_logger.run_id,
                question_id=question_id,
                question=question,
                response="",
                expected_answer=expected_answer,
                predicted_answer=None,
                correct=False,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            event_logger.event(
                "question",
                "failed",
                question_id=question_id,
                question_index=index,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise

        predicted = extract_answer(response, options)
        is_correct = predicted == expected_answer
        correct += int(is_correct)
        context_preview = str(context)[:2000] if context is not None else None

        write_eval_record(
            eval_log_path,
            run_id=event_logger.run_id,
            question_id=question_id,
            question=question,
            response=response,
            scores={"exact_match_smoke": float(is_correct)},
            expected_answer=expected_answer,
            predicted_answer=predicted,
            correct=is_correct,
            context_preview=context_preview,
            metadata=data.get("meta_info"),
        )

        event_logger.event(
            "question",
            "completed",
            question_id=question_id,
            question_index=index,
            predicted_answer=predicted,
            expected_answer=expected_answer,
            correct=is_correct,
        )

        print(f"\n--- Q{index} ---")
        print("Question:", question)
        print("Expected:", expected_answer)
        print("Predicted:", predicted)
        print("Correct:", is_correct)
        print("Response:", response)

    accuracy = correct / len(questions)
    event_logger.event("medqa_smoke_eval", "completed", accuracy=accuracy, total=len(questions), correct=correct)
    print(f"\nFinal Accuracy: {accuracy:.2f}")
    print(f"Run ID: {event_logger.run_id}")
    print(f"Event log: {event_log_path}")
    print(f"Evaluation records: {eval_log_path}")
