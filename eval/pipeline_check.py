from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from api.recommendation import RecommendationRequest, generate_recommendation
from helpers.config import startup_check
from rag.index import ensure_index
from rag.retrieve import query_index_context


@dataclass(frozen=True)
class PipelineCheckResult:
    index_ready: bool
    rag_response_preview: str
    rag_context_preview: str
    recommendation_run_id: str
    recommendation_preview: str
    trace_turns: int


async def run_pipeline_check(
    *,
    input_dir: Path,
    output_dir: Path,
    scenario: str,
    clinical_goal: str | None = None,
    use_umls: bool = False,
    schema_guided: bool = False,
    dry_run: bool = False,
) -> PipelineCheckResult:
    index = await asyncio.to_thread(
        ensure_index,
        input_dir=input_dir,
        output_dir=output_dir,
        use_umls=use_umls,
        schema_guided=schema_guided,
    )
    response, context = await query_index_context(index=index, query=scenario)
    recommendation = await generate_recommendation(
        RecommendationRequest(
            scenario=scenario,
            clinical_goal=clinical_goal,
            dry_run=dry_run,
            use_rag=True,
        )
    )
    return PipelineCheckResult(
        index_ready=True,
        rag_response_preview=str(response)[:1000],
        rag_context_preview=str(context)[:1000],
        recommendation_run_id=recommendation.run_id,
        recommendation_preview=(recommendation.final_recommendation or "")[:1000],
        trace_turns=len(recommendation.argumentation_trace),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an end-to-end smoke check for indexing, retrieval, and recommendation.")
    parser.add_argument("--input-dir", default=os.environ.get("INPUT_BASE_DIR", "data/evidence/mimic_discharge_subset"))
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_BASE_DIR", "output"))
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--clinical-goal", default=None)
    parser.add_argument("--use-umls", action="store_true")
    parser.add_argument("--schema-guided", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    startup_check()
    args = parse_args()
    result = asyncio.run(
        run_pipeline_check(
            input_dir=Path(args.input_dir),
            output_dir=Path(args.output_dir),
            scenario=args.scenario,
            clinical_goal=args.clinical_goal,
            use_umls=args.use_umls,
            schema_guided=args.schema_guided,
            dry_run=args.dry_run,
        )
    )
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
