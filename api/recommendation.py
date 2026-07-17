from dataclasses import asdict
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from argumentation.agents import ArgumentInteraction
from helpers.jsonl import JsonlLogger, new_run_id
from helpers.paths import EVAL_RECORD_LOG_PATH, EVENT_LOG_PATH
from helpers.records import write_eval_record
from retrieval.cds_graph import CdsMaterializedGraphStore, cds_graph_path
from retrieval.cds_query import SUPPORTED_CDS_TASKS, query_cds_evidence_bundle


EVENT_LOG = EVENT_LOG_PATH
EVAL_LOG = EVAL_RECORD_LOG_PATH


class RecommendationRequest(BaseModel):
    scenario: str | None = Field(default=None, description="Optional direct question override.")
    patient_id: str | None = Field(default=None, description="Optional external identifier.")
    case_id: int | None = Field(default=None, description="Case identifier in the materialized graph.")
    task: str = Field(..., description="Question task.")
    clinical_goal: str | None = Field(default=None, description="Optional target clinical goal.")
    max_rounds: int = Field(default=3, ge=1, le=10)
    top_k_cases: int = Field(default=5, ge=1, le=20)
    dry_run: bool = False


class RecommendationResponse(BaseModel):
    run_id: str
    patient_id: str | None
    case_id: int | None
    task: str
    scenario: str
    clinical_goal: str | None
    rag_available: bool
    retrieval_backend: str | None
    rag_context: str | None
    evidence_bundle: list[dict[str, Any]] = Field(default_factory=list)
    final_recommendation: str | None
    argumentation_trace: list[dict[str, Any]]
    event_log: str
    record_log: str
    warnings: list[str] = Field(default_factory=list)


def _normalize_task(task: str) -> str:
    normalized = task.strip().lower()
    if normalized not in SUPPORTED_CDS_TASKS:
        raise ValueError(f"Unsupported task: {task}")
    return normalized


def _has_graph(output_dir: Path) -> bool:
    return cds_graph_path(output_dir).exists()


async def generate_recommendation(request: RecommendationRequest) -> RecommendationResponse:
    load_dotenv()
    run_id = new_run_id("recommend")
    task = _normalize_task(request.task)
    output_dir = Path(os.environ.get("OUTPUT_BASE_DIR", "output"))
    if not _has_graph(output_dir):
        raise FileNotFoundError(f"Missing materialized graph at {cds_graph_path(output_dir)}")
    if request.case_id is None:
        raise ValueError("RecommendationRequest.case_id is required.")

    event_logger = JsonlLogger(EVENT_LOG, run_id=run_id)
    event_logger.event(
        "recommendation_request",
        "started",
        patient_id=request.patient_id,
        case_id=request.case_id,
        task=task,
        clinical_goal=request.clinical_goal,
        dry_run=request.dry_run,
    )

    with event_logger.timed("load_graph", output_dir=output_dir):
        store = CdsMaterializedGraphStore.from_persist_dir(output_dir)
    with event_logger.timed("retrieve_case_evidence", output_dir=output_dir):
        query = query_cds_evidence_bundle(
            store,
            request.case_id,
            task,
            max_cases=request.top_k_cases,
        )

    question = request.scenario or query.question
    if request.clinical_goal:
        question = f"{question}\n\nClinical goal: {request.clinical_goal}"

    if request.dry_run:
        final_recommendation = None
        trace: list[dict[str, Any]] = []
        warnings = ["Dry run requested; no generator/verifier/reasoner model calls were made."]
    else:
        with event_logger.timed("argumentation", max_rounds=request.max_rounds):
            interaction = ArgumentInteraction(
                question=question,
                rag_context=query.rag_context,
                max_rounds=request.max_rounds,
                evidence_bundle=query.evidence_bundle,
                event_logger=event_logger,
            )
            final_recommendation = interaction.run()
            trace = [asdict(turn) for turn in interaction.dialogue_history]
        warnings = []

    write_eval_record(
        EVAL_LOG,
        run_id=run_id,
        question_id=str(request.case_id),
        question=question,
        response=final_recommendation or "",
        recommendation=final_recommendation,
        final_status="dry_run" if request.dry_run else "completed",
        patient_id=request.patient_id,
        case_id=request.case_id,
        task=task,
        clinical_goal=request.clinical_goal,
        rag_available=True,
        retrieval_backend="materialized_case_graph",
        rag_context_preview=query.rag_context[:2000],
        evidence_bundle=query.evidence_bundle,
        argumentation_trace=trace,
        warnings=warnings,
        expected_answer=query.expected_answer,
    )

    event_logger.event("recommendation_request", "completed", warnings=warnings)

    return RecommendationResponse(
        run_id=run_id,
        patient_id=request.patient_id,
        case_id=request.case_id,
        task=task,
        scenario=question,
        clinical_goal=request.clinical_goal,
        rag_available=True,
        retrieval_backend="materialized_case_graph",
        rag_context=query.rag_context,
        evidence_bundle=query.evidence_bundle,
        final_recommendation=final_recommendation,
        argumentation_trace=trace,
        event_log=str(EVENT_LOG),
        record_log=str(EVAL_LOG),
        warnings=warnings,
    )
