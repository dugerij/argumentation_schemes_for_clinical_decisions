import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from graphrag.config.load_config import load_config
from pydantic import BaseModel, Field

from argumentation.agents import ArgumentInteraction
from helpers.jsonl import JsonlLogger, new_run_id
from helpers.records import write_eval_record
from rag.index import index_needs_rebuild, load_graph_tables
from rag.retrieve import drift_search_context


CONFIG_PATH = Path("settings.yaml")
EVENT_LOG = Path("logs/framework/events.jsonl")
EVAL_LOG = Path("logs/framework/eval_records.jsonl")


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
    needs_rebuild, rebuild_reasons = index_needs_rebuild(output_dir)

    if request.use_rag and needs_rebuild:
        warnings.append("GraphRAG index is unavailable or incompatible; recommendation will run without retrieved context.")
        event_logger.event("rag_context", "skipped", reasons=rebuild_reasons)
    elif request.use_rag:
        with event_logger.timed("rag_context", output_dir=output_dir):
            config = load_config(CONFIG_PATH)
            entities, communities, community_reports = load_graph_tables(output_dir)
            response, context = await drift_search_context(
                config=config,
                query=question,
                entities=entities,
                communities=communities,
                community_reports=community_reports,
                response_type="Multiple Paragraphs",
            )
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
