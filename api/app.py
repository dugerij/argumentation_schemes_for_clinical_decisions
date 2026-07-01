from collections import Counter
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query

from api.recommendation import RecommendationRequest, RecommendationResponse, generate_recommendation
from helpers.jsonl import load_jsonl
from helpers.paths import EVAL_RECORD_LOG_PATH, EVENT_LOG_PATH
from helpers.records import load_eval_records


DEFAULT_EVENT_LOG = EVENT_LOG_PATH
DEFAULT_EVAL_LOG = EVAL_RECORD_LOG_PATH

app = FastAPI(
    title="Clinical Argumentation API",
    description="API for clinical recommendation requests, framework events, evaluation records, and recommendation traces.",
    version="0.1.0",
)


def _slice(records: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return records
    return records[:limit]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/runs")
def runs() -> list[dict[str, Any]]:
    events = load_jsonl(DEFAULT_EVENT_LOG)
    eval_records = load_eval_records(DEFAULT_EVAL_LOG)
    run_ids = sorted({record.get("run_id") for record in [*events, *eval_records] if record.get("run_id")})

    output = []
    for run_id in run_ids:
        run_events = [event for event in events if event.get("run_id") == run_id]
        run_eval_records = [record for record in eval_records if record.get("run_id") == run_id]
        statuses = Counter(event.get("status") for event in run_events)
        output.append(
            {
                "run_id": run_id,
                "event_count": len(run_events),
                "eval_record_count": len(run_eval_records),
                "statuses": dict(statuses),
                "first_event_at": run_events[0].get("timestamp") if run_events else None,
                "last_event_at": run_events[-1].get("timestamp") if run_events else None,
            }
        )
    return output


@app.get("/events")
def events(
    run_id: str | None = None,
    status: str | None = None,
    step: str | None = None,
    limit: int | None = Query(default=100, ge=1),
) -> list[dict[str, Any]]:
    records = load_jsonl(DEFAULT_EVENT_LOG)
    if run_id:
        records = [record for record in records if record.get("run_id") == run_id]
    if status:
        records = [record for record in records if record.get("status") == status]
    if step:
        records = [record for record in records if record.get("step") == step]
    return _slice(records, limit)


@app.get("/eval-records")
def eval_records(
    run_id: str | None = None,
    question_id: str | None = None,
    limit: int | None = Query(default=100, ge=1),
) -> list[dict[str, Any]]:
    records = load_eval_records(DEFAULT_EVAL_LOG, run_id=run_id)
    if question_id:
        records = [record for record in records if record.get("question_id") == question_id]
    return _slice(records, limit)


@app.get("/recommendations")
def recommendations(
    run_id: str | None = None,
    limit: int | None = Query(default=100, ge=1),
) -> list[dict[str, Any]]:
    records = load_eval_records(DEFAULT_EVAL_LOG, run_id=run_id)
    recommendation_records = [
        record
        for record in records
        if record.get("recommendation") or record.get("final_status") or record.get("scores")
    ]
    return _slice(recommendation_records, limit)


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest) -> RecommendationResponse:
    return await generate_recommendation(request)
