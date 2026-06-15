from dataclasses import dataclass

from argumentation.schemes import Argument


@dataclass(frozen=True)
class CriticalQuestion:
    id: str
    scheme_type: str
    text: str


DEFAULT_MEDICAL_CQS = (
    CriticalQuestion("CQ_CONTRAINDICATION", "ASPR", "Is the recommendation contraindicated for this patient?"),
    CriticalQuestion("CQ_ADVERSE_EFFECT", "ASPR", "Does the recommendation introduce a clinically material harm?"),
    CriticalQuestion("CQ_GOAL_FIT", "ASPR", "Does the recommendation plausibly realize the stated clinical goal?"),
    CriticalQuestion("CQ_EVIDENCE", "ASPR", "Is the recommendation grounded in retrieved evidence?"),
    CriticalQuestion("CQ_HISTORY", "ASPR", "Has this or a similar intervention failed for this patient before?"),
)


def unsupported_evidence_attack(attacker: Argument, attacked: Argument) -> tuple[str, str]:
    return attacker.id, attacked.id
