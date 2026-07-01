import json
import random
import re
import asyncio
from pathlib import Path

from helpers.jsonl import JsonlLogger
from helpers.paths import EVAL_RECORD_LOG_PATH, EVENT_LOG_PATH
from helpers.records import write_eval_record
from retrieval.index import ensure_index
from retrieval.query import query_index_context


DEFAULT_DATA_PATH = Path("data/medqa/data_clean/questions/US/4_options/phrases_no_exclude_train.jsonl")
DEFAULT_EVENT_LOG = EVENT_LOG_PATH
DEFAULT_EVAL_LOG = EVAL_RECORD_LOG_PATH


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
    input_dir: Path,
    output_dir: Path,
    sample_size: int,
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
        eval_log_path=eval_log_path,
    )

    with event_logger.timed(
        "ensure_index",
        input_dir=input_dir,
        output_dir=output_dir,
    ):
        llama_index = await asyncio.to_thread(
            ensure_index,
            input_dir,
            output_dir,
        )
    print("Index loaded.")

    with event_logger.timed("load_medqa_questions", data_path=data_path, sample_size=sample_size):
        questions = load_questions(data_path, sample_size)
    correct = 0

    for question_index, data in enumerate(questions, start=1):
        options = data["options"]
        expected_answer = data["answer"]
        question = f"{data['question']}\n{format_options(options)}"
        query = f"{question}\n\nAnswer with ONLY the correct option exactly as written."

        question_id = str(data.get("id") or data.get("meta_info") or f"medqa_{question_index}")
        event_logger.event("question", "started", question_id=question_id, question_index=question_index)

        try:
            with event_logger.timed("query_index", question_id=question_id, question_index=question_index):
                response, context = await query_index_context(index=llama_index, query=query)
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
                question_index=question_index,
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
            question_index=question_index,
            predicted_answer=predicted,
            expected_answer=expected_answer,
            correct=is_correct,
        )

        print(f"\n--- Q{question_index} ---")
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
