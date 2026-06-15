from pathlib import Path
from typing import Any

from helpers.jsonl import append_jsonl, load_jsonl, utc_now


def write_eval_record(
    path: Path,
    run_id: str,
    question_id: str,
    question: str,
    response: str,
    scores: dict[str, float] | None = None,
    **payload: Any,
) -> None:
    append_jsonl(
        path,
        {
            "timestamp": utc_now(),
            "run_id": run_id,
            "question_id": question_id,
            "question": question,
            "response": response,
            "scores": scores or {},
            **payload,
        },
    )


def load_eval_records(path: Path, run_id: str | None = None) -> list[dict[str, Any]]:
    records = load_jsonl(path)
    if run_id is None:
        return records
    return [record for record in records if record.get("run_id") == run_id]
