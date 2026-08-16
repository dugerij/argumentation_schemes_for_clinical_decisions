"""Blinded post-hoc LLM judgment of diagnostic-family equivalence."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from clinical_cds.io import write_jsonl
from clinical_cds.model import DiagnosticModel
from clinical_cds.schema import ClinicalCase, DiagnosticGraph, PredictionRecord


JUDGE_ID = "blinded-diagnostic-family-judge-v4-derived-verdict"
FAMILY_RELATIONSHIPS = (
    "exact_or_synonymous", "parent_child", "sibling_branches",
    "acute_or_severity_state", "different_disease",
)
JUDGE_SYSTEM_PROMPT = """You are a blinded medical diagnostic-equivalence
judge. Compare the immutable reference diagnosis with the immutable evaluated
diagnosis. Decide whether the evaluated diagnosis preserves the same CORE
DISEASE IDENTITY. This is narrower than sharing an organ, specialty, symptom,
mechanism, or broad umbrella such as cardiovascular, infectious, respiratory,
neurological, or gastrointestinal disease.

Use exact_or_synonymous, parent_child, sibling_branches, or
acute_or_severity_state only when the diagnoses' LOWEST meaningful
COMMON DIAGNOSTIC ANCESTOR is itself a specific named disease or a clinically
recognized diagnostic syndrome with its own stable identity, and both answers
preserve that identity. A broad organ-system, specialty, generic syndrome class,
or pathological-mechanism ancestor is insufficient. The relationship need not
be direct parent-to-child: sibling branches pass when their common ancestor is
that same specific disease or named diagnostic syndrome. Allowed differences are:
- a named disease versus its clinical parent or recognized subtype;
- two recognized subtypes, phenotypes, stages, severity classes, or risk classes
  whose lowest meaningful common ancestor is that same specific named disease;
- anatomical, temporal, severity, stage, phenotype, or risk qualification of
  that same disease;
- an acute, crisis, exacerbated, or decompensated presentation that explicitly
  retains the identity of that same underlying disease, but not a different
  downstream organ complication;
- standard synonyms, abbreviations, and equivalent clinical formulations.

Examples that PASS:
- epilepsy / temporal lobe epilepsy;
- focal epilepsy / generalized epilepsy;
- sickle cell disease / sickle cell crisis;
- melanoma / acral lentiginous melanoma;
- psoriasis / severe plaque psoriasis.

Use different_disease when moving from one answer to the other requires changing
the core pathological diagnosis. In particular, FAIL:
- sibling diseases whose common ancestor is only a broad category, while
  allowing sibling branches of one specific disease or named diagnostic syndrome;
- a disease versus a different complication, cause, comorbidity, or risk factor;
- a symptom or manifestation versus its possible underlying disease;
- a procedure, treatment, laboratory result, or imaging finding versus a disease;
- two conditions that can coexist in the same encounter.

Examples that FAIL:
- epilepsy / syncope;
- melanoma / basal cell carcinoma;
- psoriasis / cellulitis.

Apply the same rule to all diseases; the examples illustrate the boundary and
are not a label-specific lookup table. Use the clinical record and controlled
paths only to disambiguate label meaning. Classify relationship as
exact_or_synonymous, parent_child, sibling_branches, acute_or_severity_state, or
different_disease. Deterministic code derives PASS from the first four and FAIL
from different_disease; do not output a separate boolean verdict. A sibling branch
under a supplied specific named diagnostic syndrome is not a different disease.
Do not decide whether a narrower label
is sufficiently evidenced and do not decide which answer is clinically better:
the only question is whether core disease identity is preserved. Abstentions are
handled deterministically before this prompt. Never rewrite either diagnosis.
Return only schema-valid JSON."""


@dataclass(frozen=True)
class ClinicalJudgeResult:
    judge_id: str
    case_id: str
    prediction_sha256: str
    same_disease_family: bool
    passed: bool
    rationale: str
    model_id: str
    attempt_count: int
    decision_source: str
    included_in_accuracy: bool = True
    exclusion_reason: str = ""
    primary_passed: bool | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def clinical_judge_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "relationship": {
                "type": "string", "enum": list(FAMILY_RELATIONSHIPS),
            },
            "rationale": {"type": "string", "minLength": 1, "maxLength": 500},
        },
        "required": ["relationship", "rationale"],
    }


def parse_clinical_judgment(payload: Mapping[str, Any]) -> tuple[bool, str]:
    if set(payload) != {"relationship", "rationale"}:
        raise ValueError("Clinical family judgment differs from the frozen schema.")
    relationship = payload["relationship"]
    if relationship not in FAMILY_RELATIONSHIPS:
        raise ValueError("Clinical family relationship is invalid.")
    rationale = payload["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("Clinical family judgment requires a rationale.")
    return relationship != "different_disease", " ".join(rationale.split())


def _prediction_sha256(record: PredictionRecord) -> str:
    frozen = {
        "run_id": record.run_id,
        "case_id": record.case_id,
        "mode": record.mode.value,
        "gold_label": record.gold_label,
        "predicted_label": record.predicted_label,
        "abstained": record.abstained,
        "citations": list(record.citations),
        "possible_diagnoses": list(record.metadata.get("possible_diagnoses") or ()),
    }
    return hashlib.sha256(json.dumps(frozen, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()


def _relevant_paths(reference: str, evaluated: str,
                    graphs: Iterable[DiagnosticGraph]) -> list[list[str]]:
    needles = {reference.casefold(), evaluated.casefold()} - {""}
    paths: list[list[str]] = []
    for graph in graphs:
        for path in graph.diagnostic_paths.values():
            if any(any(needle in label.casefold() or label.casefold() in needle
                       for needle in needles) for label in path):
                value = list(path)
                if value not in paths:
                    paths.append(value)
    return paths[:8]


def _label_key(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _controlled_same_family(
    reference: str, evaluated: str, graphs: Iterable[DiagnosticGraph]
) -> bool:
    """Resolve explicit controlled parent/branch relations before LLM judgment."""
    reference_key = _label_key(reference)
    evaluated_key = _label_key(evaluated)
    if not reference_key or not evaluated_key:
        return False
    for graph in graphs:
        category_key = _label_key(graph.category)
        reference_paths = [tuple(path) for path in graph.diagnostic_paths.values()
                           if any(_label_key(node) == reference_key for node in path)]
        evaluated_paths = [tuple(path) for path in graph.diagnostic_paths.values()
                           if any(_label_key(node) == evaluated_key for node in path)]
        if reference_key == category_key:
            reference_paths = [tuple(path) for path in graph.diagnostic_paths.values()]
        if evaluated_key == category_key:
            evaluated_paths = [tuple(path) for path in graph.diagnostic_paths.values()]
        if not reference_paths or not evaluated_paths:
            continue
        if reference_key == category_key or evaluated_key == category_key:
            return True
        for left in reference_paths:
            left_ancestors = {
                _label_key(node) for node in left
                if not node.casefold().startswith(("suspected ", "strongly suspected "))
            }
            for right in evaluated_paths:
                right_ancestors = {
                    _label_key(node) for node in right
                    if not node.casefold().startswith(("suspected ", "strongly suspected "))
                }
                if left_ancestors & right_ancestors:
                    return True
        if "syndrome" in graph.category.casefold():
            return True
    return False


def judge_prediction(*, record: PredictionRecord, case: ClinicalCase,
                     graphs: Iterable[DiagnosticGraph], model: DiagnosticModel,
                     max_output_tokens: int = 256) -> ClinicalJudgeResult:
    if record.case_id != case.case_id:
        raise ValueError("Prediction and clinical case identities differ.")
    prediction_hash = _prediction_sha256(record)
    possibilities = tuple(dict.fromkeys(str(value).strip() for value in
        (record.metadata.get("possible_diagnoses") or ()) if str(value).strip()))
    if record.predicted_label.strip() and record.predicted_label.strip() not in possibilities:
        possibilities = (record.predicted_label.strip(), *possibilities)
    if len(possibilities) > 1:
        judgments = []
        for diagnosis in possibilities:
            metadata = dict(record.metadata)
            metadata.pop("possible_diagnoses", None)
            judgment = judge_prediction(record=record.__class__(**{
                **asdict(record), "predicted_label": diagnosis, "metadata": metadata,
            }), case=case, graphs=graphs, model=model,
                max_output_tokens=max_output_tokens)
            judgments.append((diagnosis, judgment))
            if judgment.passed:
                return ClinicalJudgeResult(JUDGE_ID, record.case_id, prediction_hash,
                    True, True, f"Accepted possibility '{diagnosis}': {judgment.rationale}",
                    model.model_id, sum(value.attempt_count for _, value in judgments),
                    "prediction_set_family_match",
                    primary_passed=judgments[0][1].passed)
        unavailable = [value for _, value in judgments if not value.included_in_accuracy]
        return ClinicalJudgeResult(JUDGE_ID, record.case_id, prediction_hash,
            False, False, "No returned possibility preserves the reference disease identity.",
            model.model_id, sum(value.attempt_count for _, value in judgments),
            "prediction_set_no_family_match", not bool(unavailable),
            "evaluation_failure" if unavailable else "",
            primary_passed=judgments[0][1].passed)
    if record.abstained or not record.predicted_label.strip():
        evidence_limited = (
            record.error is None
            and record.metadata.get("abstention_category") == "insufficient_evidence"
        )
        return ClinicalJudgeResult(JUDGE_ID, record.case_id, prediction_hash,
            False, False, "The evaluated system abstained.", model.model_id,
            0, "deterministic_abstention", not evidence_limited,
            "insufficient_evidence_abstention" if evidence_limited else "")
    graph_tuple = tuple(graphs)
    normalized_reference = _label_key(record.gold_label)
    normalized_evaluated = _label_key(record.predicted_label)
    if normalized_reference and normalized_reference == normalized_evaluated:
        return ClinicalJudgeResult(JUDGE_ID, record.case_id, prediction_hash,
            True, True, "The normalized diagnostic labels are identical.",
            model.model_id, 0, "deterministic_normalized_identity")
    if _controlled_same_family(
        record.gold_label, record.predicted_label, graph_tuple
    ):
        return ClinicalJudgeResult(
            JUDGE_ID, record.case_id, prediction_hash, True, True,
            "Both labels occupy one explicit controlled diagnostic family.",
            model.model_id, 0, "deterministic_controlled_graph_family",
        )
    state = {
        "clinical_record": case.text,
        "reference_diagnosis": record.gold_label,
        "evaluated_diagnosis": record.predicted_label,
        "controlled_diagnostic_paths": _relevant_paths(
            record.gold_label, record.predicted_label, graph_tuple
        ),
        "decision_rule": (
            "PASS iff both diagnoses preserve one specific disease or named "
            "diagnostic-syndrome identity, including sibling branches and "
            "acute/severity presentations of that identity"
        ),
    }
    last_error: Exception | None = None
    for attempt_count in range(1, 4):
        try:
            raw = model.complete(JUDGE_SYSTEM_PROMPT,
                json.dumps(state, ensure_ascii=False, sort_keys=True),
                output_schema=clinical_judge_schema(), max_output_tokens=max_output_tokens)
            # Some OpenAI-compatible servers can return an otherwise valid
            # structured string containing a literal control character. JSON's
            # data model permits the decoded character; strict=False only
            # tolerates its missing wire escape and changes no schema field.
            same_family, rationale = parse_clinical_judgment(
                json.loads(raw, strict=False))
            break
        except Exception as exc:
            last_error = exc
    else:
        return ClinicalJudgeResult(
            JUDGE_ID, record.case_id, prediction_hash, False, False,
            "The family judge did not return a valid structured judgment.",
            model.model_id, 3, "judge_unavailable", False,
            "evaluation_failure",
        )
    return ClinicalJudgeResult(JUDGE_ID, record.case_id, prediction_hash,
        same_family, same_family, rationale, model.model_id, attempt_count,
        "llm_family_judge")


def write_clinical_judgments(*, records: Iterable[PredictionRecord],
                             cases: Iterable[ClinicalCase],
                             graphs: Iterable[DiagnosticGraph],
                             model: DiagnosticModel,
                             output_path: Path) -> tuple[ClinicalJudgeResult, ...]:
    case_by_id = {case.case_id: case for case in cases}
    graph_tuple = tuple(graphs)
    results = tuple(judge_prediction(record=record, case=case_by_id[record.case_id],
        graphs=graph_tuple, model=model) for record in records)
    write_jsonl(output_path, [result.to_dict() for result in results])
    return results


def summarize_clinical_judgments(
    records: Iterable[PredictionRecord], results: Iterable[ClinicalJudgeResult]
) -> tuple[dict[str, object], ...]:
    record_by_hash = {_prediction_sha256(record): record for record in records}
    grouped: defaultdict[str, list[tuple[PredictionRecord, ClinicalJudgeResult]]] = (
        defaultdict(list)
    )
    for result in results:
        record = record_by_hash.get(result.prediction_sha256)
        if record is None or record.case_id != result.case_id:
            raise ValueError("Clinical judgment does not bind to a frozen prediction.")
        grouped[record.mode.value].append((record, result))
    summaries = []
    for mode in sorted(grouped):
        bound_values = grouped[mode]
        values = [result for _, result in bound_values]
        counts = Counter(
            "excluded" if not value.included_in_accuracy
            else "pass" if value.passed else "fail"
            for value in values
        )
        included = [value for value in values if value.included_in_accuracy]
        passes = sum(value.passed for value in included)
        summaries.append({
            "mode": mode,
            "n": len(values),
            "evaluated_n": len(included),
            "excluded_count": len(values) - len(included),
            "excluded_insufficient_evidence_abstention_count": sum(
                value.exclusion_reason == "insufficient_evidence_abstention"
                for value in values
            ),
            "evaluation_failure_count": sum(
                value.exclusion_reason == "evaluation_failure" for value in values
            ),
            "accuracy_denominator_coverage": len(included) / len(values),
            "diagnostic_answer_count": sum(
                not record.abstained and bool(record.predicted_label.strip())
                for record, _ in bound_values
            ),
            "diagnostic_coverage": sum(
                not record.abstained and bool(record.predicted_label.strip())
                for record, _ in bound_values
            ) / len(values),
            "family_pass_count": passes,
            "family_accuracy": passes / len(included) if included else None,
            "primary_family_pass_count": sum(
                value.passed if value.primary_passed is None else value.primary_passed
                for value in included
            ),
            "primary_family_accuracy": sum(
                value.passed if value.primary_passed is None else value.primary_passed
                for value in included
            ) / len(included) if included else None,
            "primary_answer_count": sum(
                not record.abstained and bool(record.predicted_label.strip())
                for record, _ in bound_values
            ),
            "single_answer_count": sum(
                len(record.metadata.get("possible_diagnoses") or
                    ([record.predicted_label] if record.predicted_label else [])) == 1
                for record, _ in bound_values
            ),
            "tie_count": sum(
                len(record.metadata.get("possible_diagnoses") or ()) > 1
                for record, _ in bound_values
            ),
            "average_prediction_set_size": sum(
                len(record.metadata.get("possible_diagnoses") or
                    ([record.predicted_label] if record.predicted_label else []))
                for record, _ in bound_values
            ) / len(values),
            "decision_counts": dict(sorted(counts.items())),
        })
    return tuple(summaries)
