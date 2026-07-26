from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable

from clinical_cds.normalization import UMLSNormalizer, lexical_diagnosis_key
from clinical_cds.retrieval import tokenize
from clinical_cds.schema import (
    ClinicalCase,
    DiagnosticGraph,
    ExperimentMode,
    PredictedObservation,
    PredictionRecord,
)


GRAPH_CONSISTENCY_FLAGS = {
    "missing_guideline_graph",
    "conclusion_outside_category_graph",
    "conclusion_not_leaf",
}


@dataclass(frozen=True)
class CaseMetrics:
    case_id: str
    dataset: str
    mode: str
    exact_match: float
    hierarchy_score: float
    covered: float
    citation_validity: float
    observation_precision: float | None
    observation_recall: float | None
    observation_f1: float | None
    retrieval_gold_coverage: float
    latency_seconds: float
    cache_hit: float
    graph_consistent: float
    error: float
    argument_schema_validity: float | None
    argument_evidence_validity: float | None
    valid_evidence_reference_fraction: float | None
    verifier_review_coverage: float | None
    symbolic_trace_fidelity: float | None
    argument_resolution_changed: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


def normalized_label_key(
    value: str,
    normalizer: UMLSNormalizer | None = None,
) -> str:
    if normalizer is not None:
        return normalizer.diagnosis_key(value)
    return lexical_diagnosis_key(value)


def exact_label_match(
    predicted: str,
    gold: str,
    normalizer: UMLSNormalizer | None = None,
) -> float:
    if not predicted or not gold:
        return 0.0
    return float(
        normalized_label_key(predicted, normalizer)
        == normalized_label_key(gold, normalizer)
    )


def _hierarchy_relations(
    graphs: Iterable[DiagnosticGraph],
    normalizer: UMLSNormalizer | None = None,
) -> dict[str, set[str]]:
    related: defaultdict[str, set[str]] = defaultdict(set)
    for graph in graphs:
        for path in graph.diagnostic_paths.values():
            keys = [normalized_label_key(label, normalizer) for label in path]
            for left_index, left in enumerate(keys):
                for right in keys[left_index + 1 :]:
                    related[left].add(right)
                    related[right].add(left)
    return dict(related)


def hierarchy_score(
    predicted: str,
    gold: str,
    relations: dict[str, set[str]],
    normalizer: UMLSNormalizer | None = None,
) -> float:
    if exact_label_match(predicted, gold, normalizer):
        return 1.0
    predicted_key = normalized_label_key(predicted, normalizer)
    gold_key = normalized_label_key(gold, normalizer)
    if predicted_key and gold_key in relations.get(predicted_key, set()):
        return 0.5
    return 0.0


def _token_overlap(left: str, right: str) -> float:
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def observation_scores(
    predicted: tuple[PredictedObservation, ...],
    gold_texts: tuple[str, ...],
    *,
    threshold: float = 0.35,
) -> tuple[float, float, float]:
    if not predicted and not gold_texts:
        return 1.0, 1.0, 1.0
    if not predicted:
        return 0.0, 0.0, 0.0
    if not gold_texts:
        return 0.0, 1.0, 0.0

    pairs = [
        (_token_overlap(item.text, gold), predicted_index, gold_index)
        for predicted_index, item in enumerate(predicted)
        for gold_index, gold in enumerate(gold_texts)
    ]
    pairs.sort(reverse=True)
    matched_predicted: set[int] = set()
    matched_gold: set[int] = set()
    for score, predicted_index, gold_index in pairs:
        if score < threshold:
            break
        if predicted_index in matched_predicted or gold_index in matched_gold:
            continue
        matched_predicted.add(predicted_index)
        matched_gold.add(gold_index)

    true_positive = len(matched_predicted)
    precision = true_positive / len(predicted)
    recall = true_positive / len(gold_texts)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def _citation_validity(record: PredictionRecord) -> float:
    if not record.citations:
        return 0.0
    valid = {value.casefold() for value in record.valid_evidence_ids}
    valid_count = sum(
        citation.casefold() in valid
        for citation in record.citations
    )
    return valid_count / len(record.citations)


def _retrieval_gold_coverage(
    record: PredictionRecord,
    normalizer: UMLSNormalizer | None = None,
) -> float:
    if record.mode == ExperimentMode.DIRECT:
        return 0.0
    gold_key = normalized_label_key(record.gold_label, normalizer)
    for fact in record.metadata.get("retrieved_facts", []):
        labels = [fact.get("diagnosis_label") or "", *fact.get("diagnostic_path", [])]
        if gold_key in {
            normalized_label_key(str(label), normalizer)
            for label in labels
        }:
            return 1.0
    return 0.0


def _argument_metric(
    record: PredictionRecord,
    name: str,
) -> float | None:
    if record.mode not in {
        ExperimentMode.STRUCTURED_ARGUMENT,
        ExperimentMode.SYMBOLIC_ARGUMENT,
    }:
        return None
    value = record.metadata.get(name)
    if value is None:
        return None
    return float(value)


def _symbolic_trace_fidelity(
    record: PredictionRecord,
    normalizer: UMLSNormalizer | None = None,
) -> float | None:
    if record.mode != ExperimentMode.SYMBOLIC_ARGUMENT:
        return None
    expected = str(
        record.metadata.get("symbolic_selected_diagnosis") or ""
    )
    if not expected:
        return float(record.abstained and not record.predicted_label)
    return float(
        not record.abstained
        and exact_label_match(
            record.predicted_label,
            expected,
            normalizer,
        )
    )


def evaluate_predictions(
    records: Iterable[PredictionRecord],
    cases: Iterable[ClinicalCase],
    graphs: Iterable[DiagnosticGraph],
    normalizer: UMLSNormalizer | None = None,
) -> tuple[CaseMetrics, ...]:
    case_index = {case.case_id: case for case in cases}
    relations = _hierarchy_relations(graphs, normalizer)
    metrics: list[CaseMetrics] = []
    for record in records:
        case = case_index.get(record.case_id)
        gold_observations = (
            tuple(node.text for node in case.gold_observations)
            if case is not None
            else ()
        )
        if case is not None and case.annotation_nodes:
            precision, recall, f1 = observation_scores(
                record.observations,
                gold_observations,
            )
        else:
            precision, recall, f1 = None, None, None
        graph_consistent = not bool(
            set(record.quality_flags) & GRAPH_CONSISTENCY_FLAGS
        )
        covered = not record.abstained and bool(record.predicted_label)
        metrics.append(
            CaseMetrics(
                case_id=record.case_id,
                dataset=record.dataset,
                mode=record.mode.value,
                exact_match=(
                    exact_label_match(
                        record.predicted_label,
                        record.gold_label,
                        normalizer,
                    )
                    if covered
                    else 0.0
                ),
                hierarchy_score=(
                    hierarchy_score(
                        record.predicted_label,
                        record.gold_label,
                        relations,
                        normalizer,
                    )
                    if covered
                    else 0.0
                ),
                covered=float(covered),
                citation_validity=_citation_validity(record),
                observation_precision=precision,
                observation_recall=recall,
                observation_f1=f1,
                retrieval_gold_coverage=_retrieval_gold_coverage(
                    record,
                    normalizer,
                ),
                latency_seconds=record.latency_seconds,
                cache_hit=float(record.cache_hit),
                graph_consistent=float(graph_consistent),
                error=float(record.error is not None),
                argument_schema_validity=_argument_metric(
                    record,
                    "argument_schema_validity",
                ),
                argument_evidence_validity=_argument_metric(
                    record,
                    "argument_evidence_validity",
                ),
                valid_evidence_reference_fraction=_argument_metric(
                    record,
                    "valid_evidence_reference_fraction",
                ),
                verifier_review_coverage=_argument_metric(
                    record,
                    "verifier_review_coverage",
                ),
                symbolic_trace_fidelity=_symbolic_trace_fidelity(
                    record,
                    normalizer,
                ),
                argument_resolution_changed=_argument_metric(
                    record,
                    "argument_resolution_changed",
                ),
            )
        )
    return tuple(metrics)


def _mean(values: list[float | None]) -> float | None:
    observed = [value for value in values if value is not None]
    return fmean(observed) if observed else None


def summarize_modes(metrics: Iterable[CaseMetrics]) -> list[dict[str, Any]]:
    metric_list = tuple(metrics)
    output: list[dict[str, Any]] = []
    for subset_name, subset_filter in (
        ("all", lambda row: True),
        ("graph_consistent", lambda row: bool(row.graph_consistent)),
    ):
        grouped: defaultdict[tuple[str, str], list[CaseMetrics]] = defaultdict(list)
        for row in metric_list:
            if subset_filter(row):
                grouped[(row.dataset, row.mode)].append(row)
        for (dataset, mode), rows in sorted(grouped.items()):
            exact_values = [row.exact_match for row in rows]
            output.append(
                {
                    "subset": subset_name,
                    "dataset": dataset,
                    "mode": mode,
                    "n": len(rows),
                    "accuracy": _mean(exact_values),
                    "accuracy_sd": pstdev(exact_values) if len(exact_values) > 1 else 0.0,
                    "hierarchy_score": _mean([row.hierarchy_score for row in rows]),
                    "coverage": _mean([row.covered for row in rows]),
                    "selective_accuracy": _mean(
                        [
                            row.exact_match
                            for row in rows
                            if row.covered
                        ]
                    ),
                    "citation_validity": _mean([row.citation_validity for row in rows]),
                    "observation_precision": _mean(
                        [row.observation_precision for row in rows]
                    ),
                    "observation_recall": _mean(
                        [row.observation_recall for row in rows]
                    ),
                    "observation_f1": _mean(
                        [row.observation_f1 for row in rows]
                    ),
                    "retrieval_gold_coverage": _mean(
                        [row.retrieval_gold_coverage for row in rows]
                    ),
                    "mean_latency_seconds": _mean(
                        [row.latency_seconds for row in rows if not row.cache_hit]
                    ),
                    "error_rate": _mean([row.error for row in rows]),
                    "argument_schema_validity": _mean(
                        [row.argument_schema_validity for row in rows]
                    ),
                    "argument_evidence_validity": _mean(
                        [row.argument_evidence_validity for row in rows]
                    ),
                    "valid_evidence_reference_fraction": _mean(
                        [
                            row.valid_evidence_reference_fraction
                            for row in rows
                        ]
                    ),
                    "verifier_review_coverage": _mean(
                        [row.verifier_review_coverage for row in rows]
                    ),
                    "symbolic_trace_fidelity": _mean(
                        [row.symbolic_trace_fidelity for row in rows]
                    ),
                    "argument_resolution_change_rate": _mean(
                        [row.argument_resolution_changed for row in rows]
                    ),
                }
            )
    return output


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _paired_bootstrap(
    baseline: list[float],
    comparison: list[float],
    *,
    samples: int = 2000,
    seed: int = 17,
) -> tuple[float, float, float]:
    differences = [
        comparison_value - baseline_value
        for baseline_value, comparison_value in zip(baseline, comparison, strict=True)
    ]
    if not differences:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    bootstrap_means = [
        fmean(rng.choice(differences) for _ in differences)
        for _ in range(samples)
    ]
    return (
        fmean(differences),
        _quantile(bootstrap_means, 0.025),
        _quantile(bootstrap_means, 0.975),
    )


def _mcnemar_exact(baseline: list[float], comparison: list[float]) -> tuple[int, int, float]:
    baseline_only = sum(
        left == 1.0 and right == 0.0
        for left, right in zip(baseline, comparison, strict=True)
    )
    comparison_only = sum(
        left == 0.0 and right == 1.0
        for left, right in zip(baseline, comparison, strict=True)
    )
    discordant = baseline_only + comparison_only
    if discordant == 0:
        return baseline_only, comparison_only, 1.0
    tail = min(baseline_only, comparison_only)
    cumulative = sum(
        math.comb(discordant, index)
        for index in range(tail + 1)
    ) / (2**discordant)
    return baseline_only, comparison_only, min(1.0, 2.0 * cumulative)


def paired_comparisons(
    metrics: Iterable[CaseMetrics],
    *,
    baseline_mode: str = ExperimentMode.DIRECT.value,
) -> list[dict[str, Any]]:
    metric_list = tuple(metrics)
    output: list[dict[str, Any]] = []
    datasets = sorted({row.dataset for row in metric_list})
    for dataset in datasets:
        rows = [row for row in metric_list if row.dataset == dataset]
        by_mode: defaultdict[str, dict[str, CaseMetrics]] = defaultdict(dict)
        for row in rows:
            by_mode[row.mode][row.case_id] = row
        baseline = by_mode.get(baseline_mode, {})
        for mode in sorted(by_mode):
            if mode == baseline_mode:
                continue
            common_ids = sorted(set(baseline) & set(by_mode[mode]))
            baseline_values = [baseline[case_id].exact_match for case_id in common_ids]
            comparison_values = [
                by_mode[mode][case_id].exact_match for case_id in common_ids
            ]
            difference, lower, upper = _paired_bootstrap(
                baseline_values,
                comparison_values,
            )
            baseline_only, comparison_only, p_value = _mcnemar_exact(
                baseline_values,
                comparison_values,
            )
            output.append(
                {
                    "dataset": dataset,
                    "baseline_mode": baseline_mode,
                    "comparison_mode": mode,
                    "n": len(common_ids),
                    "accuracy_difference": difference,
                    "bootstrap_ci_low": lower,
                    "bootstrap_ci_high": upper,
                    "mcnemar_baseline_only_correct": baseline_only,
                    "mcnemar_comparison_only_correct": comparison_only,
                    "mcnemar_exact_p": p_value,
                }
            )
    return output


def load_prediction_records(path: Path) -> tuple[PredictionRecord, ...]:
    records: list[PredictionRecord] = []
    with Path(path).open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            payload = json.loads(line)
            records.append(
                PredictionRecord(
                    run_id=str(payload["run_id"]),
                    case_id=str(payload["case_id"]),
                    dataset=str(payload["dataset"]),
                    task=str(payload["task"]),
                    mode=ExperimentMode(payload["mode"]),
                    model_id=str(payload["model_id"]),
                    gold_label=str(payload["gold_label"]),
                    predicted_label=str(payload.get("predicted_label") or ""),
                    reasoning=str(payload.get("reasoning") or ""),
                    citations=tuple(payload.get("citations") or ()),
                    observations=tuple(
                        PredictedObservation(
                            text=str(item.get("text") or ""),
                            source_id=item.get("source_id"),
                        )
                        for item in payload.get("observations") or ()
                    ),
                    abstained=bool(payload.get("abstained")),
                    latency_seconds=float(payload.get("latency_seconds") or 0.0),
                    prompt_hash=str(payload.get("prompt_hash") or ""),
                    cache_hit=bool(payload.get("cache_hit")),
                    valid_evidence_ids=tuple(payload.get("valid_evidence_ids") or ()),
                    quality_flags=tuple(payload.get("quality_flags") or ()),
                    error=payload.get("error"),
                    metadata=dict(payload.get("metadata") or {}),
                )
            )
    return tuple(records)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_evaluation_artifacts(
    *,
    records: Iterable[PredictionRecord],
    cases: Iterable[ClinicalCase],
    graphs: Iterable[DiagnosticGraph],
    output_dir: Path,
    normalizer: UMLSNormalizer | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_predictions(
        records,
        cases,
        graphs,
        normalizer=normalizer,
    )
    summary = summarize_modes(metrics)
    comparisons = paired_comparisons(metrics)

    case_metrics_path = output_dir / "case_metrics.csv"
    summary_path = output_dir / "mode_summary.csv"
    comparisons_path = output_dir / "paired_comparisons.csv"
    _write_csv(case_metrics_path, [row.to_dict() for row in metrics])
    _write_csv(summary_path, summary)
    _write_csv(comparisons_path, comparisons)
    paths = {
        "case_metrics": case_metrics_path,
        "mode_summary": summary_path,
        "paired_comparisons": comparisons_path,
    }
    from clinical_cds.reporting import write_report_plots

    paths.update(write_report_plots(summary, comparisons, output_dir))
    return paths
