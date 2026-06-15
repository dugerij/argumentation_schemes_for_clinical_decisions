from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationScores:
    correctness: float
    grounding: float
    clinical_reasoning: float
    safety: float
    auditability: float

    @property
    def mean(self) -> float:
        return (
            self.correctness
            + self.grounding
            + self.clinical_reasoning
            + self.safety
            + self.auditability
        ) / 5


def exact_match_score(predicted: str | None, reference: str | None) -> float:
    if not predicted or not reference:
        return 0.0
    return float(predicted.strip().lower() == reference.strip().lower())
