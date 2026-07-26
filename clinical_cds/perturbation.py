from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from clinical_cds.evaluation import exact_label_match, normalized_label_key
from clinical_cds.normalization import UMLSNormalizer
from clinical_cds.patient import remove_case_section
from clinical_cds.retrieval import section_evidence
from clinical_cds.schema import ClinicalCase, PredictionRecord


@dataclass(frozen=True)
class PerturbationPair:
    base_case: ClinicalCase
    perturbed_case: ClinicalCase
    removed_section: str
    removed_evidence_id: str
    removed_gold_observation_count: int


@dataclass(frozen=True)
class PerturbationMetrics:
    base_case_id: str
    perturbed_case_id: str
    mode: str
    removed_section: str
    removed_evidence_id: str
    removed_gold_observation_count: int
    base_correct: float
    perturbed_correct: float
    answer_changed: float
    base_cited_removed_section: float
    stale_removed_section_citation: float
    base_abstained: float
    perturbed_abstained: float

    def to_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


def build_section_removal_pairs(
    cases: Iterable[ClinicalCase],
) -> tuple[PerturbationPair, ...]:
    pairs: list[PerturbationPair] = []
    for case in cases:
        counts = Counter(
            node.source_section
            for node in case.gold_observations
            if node.source_section in case.sections
        )
        if not counts or len(case.sections) < 2:
            continue
        section_order = {
            section_name: index
            for index, section_name in enumerate(case.sections)
        }
        removed_section, observation_count = sorted(
            counts.items(),
            key=lambda item: (
                -item[1],
                section_order.get(item[0], len(section_order)),
                item[0],
            ),
        )[0]
        evidence_by_section = {
            section_name: evidence_id
            for evidence_id, section_name, _ in section_evidence(case)
        }
        removed_evidence_id = evidence_by_section[removed_section]
        pairs.append(
            PerturbationPair(
                base_case=case,
                perturbed_case=remove_case_section(case, removed_section),
                removed_section=removed_section,
                removed_evidence_id=removed_evidence_id,
                removed_gold_observation_count=observation_count,
            )
        )
    return tuple(pairs)


def evaluate_perturbations(
    pairs: Iterable[PerturbationPair],
    records: Iterable[PredictionRecord],
    *,
    normalizer: UMLSNormalizer | None = None,
) -> tuple[PerturbationMetrics, ...]:
    record_index = {
        (record.case_id, record.mode.value): record
        for record in records
    }
    metrics: list[PerturbationMetrics] = []
    for pair in pairs:
        modes = sorted(
            {
                mode
                for case_id, mode in record_index
                if case_id == pair.base_case.case_id
            }
        )
        for mode in modes:
            base = record_index[(pair.base_case.case_id, mode)]
            perturbed = record_index.get((pair.perturbed_case.case_id, mode))
            if perturbed is None:
                continue
            removed_id = pair.removed_evidence_id.casefold()
            metrics.append(
                PerturbationMetrics(
                    base_case_id=pair.base_case.case_id,
                    perturbed_case_id=pair.perturbed_case.case_id,
                    mode=mode,
                    removed_section=pair.removed_section,
                    removed_evidence_id=pair.removed_evidence_id,
                    removed_gold_observation_count=pair.removed_gold_observation_count,
                    base_correct=(
                        exact_label_match(
                            base.predicted_label,
                            base.gold_label,
                            normalizer,
                        )
                        if not base.abstained
                        else 0.0
                    ),
                    perturbed_correct=(
                        exact_label_match(
                            perturbed.predicted_label,
                            perturbed.gold_label,
                            normalizer,
                        )
                        if not perturbed.abstained
                        else 0.0
                    ),
                    answer_changed=float(
                        normalized_label_key(base.predicted_label, normalizer)
                        != normalized_label_key(
                            perturbed.predicted_label,
                            normalizer,
                        )
                    ),
                    base_cited_removed_section=float(
                        removed_id
                        in {citation.casefold() for citation in base.citations}
                    ),
                    stale_removed_section_citation=float(
                        removed_id
                        in {
                            citation.casefold()
                            for citation in perturbed.citations
                        }
                    ),
                    base_abstained=float(base.abstained),
                    perturbed_abstained=float(perturbed.abstained),
                )
            )
    return tuple(metrics)


def write_perturbation_artifacts(
    metrics: Iterable[PerturbationMetrics],
    output_dir: Path,
) -> dict[str, Path]:
    rows = [metric.to_dict() for metric in metrics]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "perturbation_metrics.csv"
    summary_path = output_dir / "perturbation_summary.json"

    if rows:
        with metrics_path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            "pair_count": len(rows),
            "base_accuracy": fmean(row["base_correct"] for row in rows),
            "perturbed_accuracy": fmean(
                row["perturbed_correct"] for row in rows
            ),
            "accuracy_difference": fmean(
                row["perturbed_correct"] - row["base_correct"]
                for row in rows
            ),
            "answer_change_rate": fmean(
                row["answer_changed"] for row in rows
            ),
            "stale_removed_section_citation_rate": fmean(
                row["stale_removed_section_citation"] for row in rows
            ),
            "perturbed_abstention_rate": fmean(
                row["perturbed_abstained"] for row in rows
            ),
        }
    else:
        metrics_path.write_text("", encoding="utf-8")
        summary = {
            "pair_count": 0,
            "base_accuracy": None,
            "perturbed_accuracy": None,
            "accuracy_difference": None,
            "answer_change_rate": None,
            "stale_removed_section_citation_rate": None,
            "perturbed_abstention_rate": None,
        }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "perturbation_metrics": metrics_path,
        "perturbation_summary": summary_path,
    }
