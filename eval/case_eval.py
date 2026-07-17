from dataclasses import dataclass
from pathlib import Path

from api.recommendation import RecommendationRequest, generate_recommendation
from retrieval.cds_graph import CdsMaterializedGraphStore
from retrieval.cds_query import SUPPORTED_CDS_TASKS, query_cds_evidence_bundle


@dataclass(frozen=True)
class CaseEvalRecord:
    case_id: int
    expected_answer: str
    predicted_answer: str | None
    correct: bool


@dataclass(frozen=True)
class CaseEvalSummary:
    task: str
    sample_size: int
    correct_count: int
    accuracy: float
    records: tuple[CaseEvalRecord, ...]


def _normalize_task(task: str) -> str:
    normalized = task.strip().lower()
    if normalized not in SUPPORTED_CDS_TASKS:
        raise ValueError(f"Unsupported task: {task}")
    return normalized


def _extract_answer(text: str | None) -> str | None:
    if not text:
        return None
    for line in reversed(text.splitlines()):
        if line.strip().lower().startswith("answer:"):
            answer = line.split(":", 1)[1].strip()
            return answer or None
    compact = " ".join(text.split()).strip()
    return compact or None


def _candidate_cases(store: CdsMaterializedGraphStore, task: str) -> list[int]:
    case_ids: list[int] = []
    for case in store.all_cases():
        result = query_cds_evidence_bundle(store, case.stay_id, task, max_cases=1)
        if result.expected_answer:
            case_ids.append(case.stay_id)
    return case_ids


async def run_case_eval(
    output_dir: Path,
    task: str,
    *,
    sample_size: int,
    max_rounds: int,
    top_k_cases: int,
) -> CaseEvalSummary:
    task = _normalize_task(task)
    store = CdsMaterializedGraphStore.from_persist_dir(output_dir)
    case_ids = _candidate_cases(store, task)[:sample_size]

    records: list[CaseEvalRecord] = []
    for case_id in case_ids:
        query = query_cds_evidence_bundle(store, case_id, task, max_cases=top_k_cases)
        response = await generate_recommendation(
            RecommendationRequest(
                case_id=case_id,
                task=task,
                max_rounds=max_rounds,
                top_k_cases=top_k_cases,
            )
        )
        predicted_answer = _extract_answer(response.final_recommendation)
        expected_answer = query.expected_answer or ""
        correct = bool(predicted_answer) and predicted_answer.strip().casefold() == expected_answer.strip().casefold()
        records.append(
            CaseEvalRecord(
                case_id=case_id,
                expected_answer=expected_answer,
                predicted_answer=predicted_answer,
                correct=correct,
            )
        )

    correct_count = sum(1 for record in records if record.correct)
    sample_count = len(records)
    accuracy = (correct_count / sample_count) if sample_count else 0.0
    return CaseEvalSummary(
        task=task,
        sample_size=sample_count,
        correct_count=correct_count,
        accuracy=accuracy,
        records=tuple(records),
    )
