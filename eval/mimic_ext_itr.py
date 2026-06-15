from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OpenEndedClinicalQuestion:
    question_id: str
    question: str
    reference_answer: str | None = None
    patient_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def load_questions(_path: str) -> list[OpenEndedClinicalQuestion]:
    raise NotImplementedError(
        "MIMIC-IV-Ext-ITR loading will be implemented when the approved dataset schema is available."
    )
