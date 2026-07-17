from dataclasses import dataclass
import re
from typing import Any

from retrieval.cds_graph import CdsCase, CdsMaterializedGraphStore


TASK_DIAGNOSIS = "diagnosis"
TASK_TRIAGE = "triage"
TASK_SPECIALTY = "specialty_referral"
SUPPORTED_CDS_TASKS = (
    TASK_DIAGNOSIS,
    TASK_TRIAGE,
    TASK_SPECIALTY,
)
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "patient",
    "she",
    "that",
    "the",
    "their",
    "to",
    "was",
    "were",
    "with",
}


@dataclass(frozen=True)
class RetrievedCase:
    stay_id: int
    score: float
    shared_tokens: tuple[str, ...]
    task_label: str | None
    summary: str


@dataclass(frozen=True)
class CdsQueryResult:
    question: str
    expected_answer: str | None
    rag_context: str
    evidence_bundle: list[dict[str, Any]]
    retrieved_cases: tuple[RetrievedCase, ...]


def _normalize_task(task: str) -> str:
    normalized = task.strip().lower()
    if normalized not in SUPPORTED_CDS_TASKS:
        raise ValueError(f"Unsupported CDS task: {task}")
    return normalized


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in TOKEN_RE.findall(text.lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def _safe_text(value: str | None) -> str:
    return " ".join((value or "").split()).strip()


def _case_background(case: CdsCase) -> str:
    parts = [
        f"HPI: {_safe_text(case.hpi)}" if case.hpi else "",
        f"Chief complaint: {_safe_text(case.chiefcomplaint)}" if case.chiefcomplaint else "",
        f"Patient info: {_safe_text(case.patient_info)}" if case.patient_info else "",
        f"Initial vitals: {_safe_text(case.initial_vitals)}" if case.initial_vitals else "",
        f"Tests: {_safe_text(case.tests)}" if case.tests else "",
        f"Medication history: {_safe_text(case.past_medication)}" if case.past_medication else "",
    ]
    return "\n".join(part for part in parts if part)


def _task_label(case: CdsCase, task: str) -> str | None:
    task = _normalize_task(task)
    if task == TASK_DIAGNOSIS:
        if case.primary_diagnosis:
            return case.primary_diagnosis[0]
        return None
    if task == TASK_TRIAGE:
        return _safe_text(case.triage) or None
    if task == TASK_SPECIALTY:
        if case.specialty_referral:
            return case.specialty_referral[0]
        return None
    raise ValueError(f"Unsupported CDS task: {task}")


def _question_prompt(task: str) -> str:
    if task == TASK_DIAGNOSIS:
        return (
            "What is the most likely diagnosis for this emergency department case? "
            "Use the patient details and retrieved precedent cases. "
            "Reason step by step in short exam style, then end with a final line exactly in the form "
            "`Answer: <diagnosis>`."
        )
    if task == TASK_TRIAGE:
        return (
            "What is the most likely emergency department triage level for this case? "
            "Use the patient details and retrieved precedent cases. "
            "Reason step by step in short exam style, then end with a final line exactly in the form "
            "`Answer: <triage level>`."
        )
    if task == TASK_SPECIALTY:
        return (
            "Which specialty is the best referral destination for this case? "
            "Use the patient details and retrieved precedent cases. "
            "Reason step by step in short exam style, then end with a final line exactly in the form "
            "`Answer: <specialty>`."
        )
    raise ValueError(f"Unsupported CDS task: {task}")


def _case_question(case: CdsCase, task: str) -> str:
    background = _case_background(case)
    return f"{_question_prompt(task)}\n\nTarget case (stay_id={case.stay_id}):\n{background}"


def _similar_case_summary(case: CdsCase, task: str) -> str:
    label = _task_label(case, task) or "unknown"
    return (
        f"Precedent case stay_id={case.stay_id}\n"
        f"{_case_background(case)}\n"
        f"Observed {task.replace('_', ' ')}: {label}"
    )


def _score_candidate(target_tokens: set[str], candidate_tokens: set[str]) -> float:
    if not target_tokens or not candidate_tokens:
        return 0.0
    overlap = target_tokens & candidate_tokens
    if not overlap:
        return 0.0
    return len(overlap) / len(target_tokens | candidate_tokens)


def _retrieve_precedent_cases(
    store: CdsMaterializedGraphStore,
    target_case: CdsCase,
    task: str,
    max_cases: int,
) -> list[RetrievedCase]:
    candidate_ids = store.search_stay_ids(_case_background(target_case))
    target_tokens = _tokenize(_case_background(target_case))
    results: list[RetrievedCase] = []
    for stay_id in candidate_ids:
        if stay_id == target_case.stay_id:
            continue
        candidate_case = store.get_case(stay_id)
        if candidate_case is None:
            continue
        label = _task_label(candidate_case, task)
        if not label:
            continue
        candidate_tokens = _tokenize(_case_background(candidate_case))
        score = _score_candidate(target_tokens, candidate_tokens)
        if score <= 0.0:
            continue
        shared_tokens = tuple(sorted((target_tokens & candidate_tokens))[:12])
        results.append(
            RetrievedCase(
                stay_id=candidate_case.stay_id,
                score=score,
                shared_tokens=shared_tokens,
                task_label=label,
                summary=_similar_case_summary(candidate_case, task),
            )
        )
    results.sort(key=lambda item: (-item.score, item.stay_id))
    return results[:max_cases]


def query_cds_evidence_bundle(
    store: CdsMaterializedGraphStore,
    stay_id: int,
    task: str,
    *,
    max_cases: int = 5,
) -> CdsQueryResult:
    task = _normalize_task(task)
    case = store.get_case(stay_id)
    if case is None:
        raise ValueError(f"Unknown CDS stay_id: {stay_id}")

    retrieved_cases = _retrieve_precedent_cases(store, case, task, max_cases=max_cases)
    question = _case_question(case, task)
    expected_answer = _task_label(case, task)

    evidence_bundle: list[dict[str, Any]] = [
        {
            "evidence_id": "E1",
            "kind": "target_case",
            "source_name": f"stay:{case.stay_id}",
            "section": "target_case",
            "snippet": _case_background(case),
            "score": 1.0,
            "node_id": f"stay:{case.stay_id}",
        }
    ]

    rag_sections = [
        f"Task: {task}",
        f"Target case stay_id={case.stay_id}",
        _case_background(case),
    ]

    for offset, precedent in enumerate(retrieved_cases, start=2):
        evidence_bundle.append(
            {
                "evidence_id": f"E{offset}",
                "kind": "precedent_case",
                "source_name": f"stay:{precedent.stay_id}",
                "section": task,
                "snippet": precedent.summary,
                "score": round(precedent.score, 4),
                "node_id": f"stay:{precedent.stay_id}",
                "shared_tokens": list(precedent.shared_tokens),
                "task_label": precedent.task_label,
            }
        )
        rag_sections.append(precedent.summary)

    rag_context = "\n\n".join(section for section in rag_sections if section)
    return CdsQueryResult(
        question=question,
        expected_answer=expected_answer,
        rag_context=rag_context,
        evidence_bundle=evidence_bundle,
        retrieved_cases=tuple(retrieved_cases),
    )
