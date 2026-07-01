import asyncio
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from argumentation.agents import ArgumentInteraction
from helpers.jsonl import JsonlLogger, new_run_id
from helpers.paths import EVAL_RECORD_LOG_PATH, EVENT_LOG_PATH
from helpers.records import write_eval_record
from retrieval.index import ensure_index
from retrieval.query import query_index_context


EVENT_LOG = EVENT_LOG_PATH
EVAL_LOG = EVAL_RECORD_LOG_PATH


class RecommendationRequest(BaseModel):
    scenario: str = Field(..., min_length=10, description="Clinical scenario or question.")
    patient_id: str | None = Field(default=None, description="Optional patient identifier.")
    clinical_goal: str | None = Field(default=None, description="Optional target clinical goal.")
    max_rounds: int = Field(default=3, ge=1, le=10)
    use_rag: bool = True
    dry_run: bool = False


class RecommendationResponse(BaseModel):
    run_id: str
    patient_id: str | None
    scenario: str
    clinical_goal: str | None
    rag_available: bool
    rag_context: str | None
    final_recommendation: str | None
    argumentation_trace: list[dict[str, Any]]
    event_log: str
    record_log: str
    warnings: list[str] = Field(default_factory=list)


def _build_question(request: RecommendationRequest) -> str:
    if request.clinical_goal:
        return f"{request.scenario}\n\nClinical goal: {request.clinical_goal}"
    return request.scenario


async def generate_recommendation(request: RecommendationRequest) -> RecommendationResponse:
    load_dotenv()
    run_id = new_run_id("recommend")
    event_logger = JsonlLogger(EVENT_LOG, run_id=run_id)
    event_logger.event(
        "recommendation_request",
        "started",
        patient_id=request.patient_id,
        clinical_goal=request.clinical_goal,
        use_rag=request.use_rag,
        dry_run=request.dry_run,
    )

    warnings: list[str] = []
    question = _build_question(request)
    rag_context = None
    rag_available = False

    output_dir = Path(os.environ.get("OUTPUT_BASE_DIR", "output"))
    input_dir = Path(os.environ.get("INPUT_BASE_DIR", "data/evidence/mimic_discharge_subset"))

    if request.use_rag:
        with event_logger.timed("ensure_index", output_dir=output_dir, input_dir=input_dir):
            index = await asyncio.to_thread(
                ensure_index,
                input_dir=input_dir,
                output_dir=output_dir,
            )

        with event_logger.timed("rag_context", output_dir=output_dir):
            response, context = await query_index_context(index=index, query=question)
            rag_context = str(context or response)
            rag_available = True

    if request.dry_run:
        final_recommendation = None
        trace: list[dict[str, Any]] = []
        warnings.append("Dry run requested; no generator/verifier/reasoner model calls were made.")
    else:
        with event_logger.timed("argumentation", max_rounds=request.max_rounds):
            interaction = ArgumentInteraction(
                question=question,
                rag_context=rag_context or "",
                max_rounds=request.max_rounds,
                event_logger=event_logger,
            )
            final_recommendation = interaction.run()
            trace = [asdict(turn) for turn in interaction.dialogue_history]

    write_eval_record(
        EVAL_LOG,
        run_id=run_id,
        question_id=request.patient_id or run_id,
        question=question,
        response=final_recommendation or "",
        recommendation=final_recommendation,
        final_status="dry_run" if request.dry_run else "completed",
        patient_id=request.patient_id,
        clinical_goal=request.clinical_goal,
        rag_available=rag_available,
        rag_context_preview=(rag_context[:2000] if rag_context else None),
        argumentation_trace=trace,
        warnings=warnings,
    )

    event_logger.event("recommendation_request", "completed", warnings=warnings)

    return RecommendationResponse(
        run_id=run_id,
        patient_id=request.patient_id,
        scenario=request.scenario,
        clinical_goal=request.clinical_goal,
        rag_available=rag_available,
        rag_context=rag_context,
        final_recommendation=final_recommendation,
        argumentation_trace=trace,
        event_log=str(EVENT_LOG),
        record_log=str(EVAL_LOG),
        warnings=warnings,
    )
