"""Deterministic decision record for the argumentation result."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AdjudicationDecision:
    case_id: str
    selected_candidate_id: str
    selected_diagnosis: str
    abstained: bool
    confidence: str
    supporting_argument_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    candidate_assessments: tuple[dict[str, str], ...]
    latency_seconds: float
    model_id: str
    retrieval_bundle_sha256: str = ""
    adjudicator_input_sha256: str = ""
    adjudicator_prompt_sha256: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_diagnosis": self.selected_diagnosis,
            "abstained": self.abstained,
            "confidence": self.confidence,
            "supporting_argument_ids": list(self.supporting_argument_ids),
            "evidence_ids": list(self.evidence_ids),
            "candidate_assessments": list(self.candidate_assessments),
            "latency_seconds": self.latency_seconds,
            "model_id": self.model_id,
            "retrieval_bundle_sha256": self.retrieval_bundle_sha256,
            "adjudicator_input_sha256": self.adjudicator_input_sha256,
            "adjudicator_prompt_sha256": self.adjudicator_prompt_sha256,
            "error": self.error,
        }
