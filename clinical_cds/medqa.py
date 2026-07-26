from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

from clinical_cds.direct import normalize_label
from clinical_cds.normalization import (
    UMLSNormalizer,
    lexical_diagnosis_key,
)
from clinical_cds.schema import ClinicalCase, DiagnosticGraph


DIAGNOSTIC_QUESTION_RE = re.compile(
    r"\b(?:what|which)\b[^?]{0,120}\b(?:most likely )?(?:clinical )?diagnosis\b"
    r"|\bmost likely diagnosis\?",
    flags=re.IGNORECASE,
)


def _resolve_split_path(path: Path, split: str) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    candidates = [
        path / f"{split}.jsonl",
        path / "data_clean" / "questions" / "US" / f"{split}.jsonl",
        path / "questions" / "US" / f"{split}.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find the MedQA {split} split under {path}.")


def _diagnosis_lookup(
    graphs: Iterable[DiagnosticGraph],
    normalizer: UMLSNormalizer | None = None,
) -> dict[str, tuple[str, str]]:
    lookup: dict[str, tuple[str, str]] = {}
    for graph in graphs:
        for label in graph.diagnosis_labels:
            keys = (
                normalizer.diagnosis_keys(label)
                if normalizer is not None
                else (lexical_diagnosis_key(label),)
            )
            for key in keys:
                lookup[key] = (graph.category, label)
    return lookup


def load_medqa_cases(
    path: Path,
    *,
    split: str = "test",
    graphs: Iterable[DiagnosticGraph] = (),
    diagnostic_only: bool = True,
    graph_covered_only: bool = False,
    normalizer: UMLSNormalizer | None = None,
) -> tuple[ClinicalCase, ...]:
    split_path = _resolve_split_path(Path(path), split)
    diagnosis_lookup = _diagnosis_lookup(graphs, normalizer)
    cases: list[ClinicalCase] = []

    with split_path.open("r", encoding="utf-8") as source:
        for row_index, line in enumerate(source):
            if not line.strip():
                continue
            payload = json.loads(line)
            question = normalize_label(str(payload.get("question") or ""))
            answer = normalize_label(str(payload.get("answer") or ""))
            options = {
                str(key): normalize_label(str(value))
                for key, value in (payload.get("options") or {}).items()
            }
            if not question or not answer or not options:
                continue

            is_diagnostic = bool(DIAGNOSTIC_QUESTION_RE.search(question))
            if diagnostic_only and not is_diagnostic:
                continue
            answer_keys = (
                normalizer.diagnosis_keys(answer)
                if normalizer is not None
                else (lexical_diagnosis_key(answer),)
            )
            coverage = next(
                (
                    diagnosis_lookup[key]
                    for key in answer_keys
                    if key in diagnosis_lookup
                ),
                None,
            )
            if graph_covered_only and coverage is None:
                continue

            digest = hashlib.sha256(
                f"medqa|{split}|{row_index}|{question}".encode("utf-8")
            ).hexdigest()[:12]
            quality_flags = () if coverage else ("outside_direct_graph_coverage",)
            cases.append(
                ClinicalCase(
                    case_id=f"medqa-{split}-{digest}",
                    dataset="medqa",
                    task="multiple_choice",
                    sections={"question": question},
                    options=options,
                    gold_label=answer,
                    disease_category=coverage[0] if coverage else None,
                    metadata={
                        "split": split,
                        "meta_info": payload.get("meta_info"),
                        "answer_idx": payload.get("answer_idx"),
                        "diagnostic_question": is_diagnostic,
                        "direct_graph_covered": coverage is not None,
                    },
                    quality_flags=quality_flags,
                )
            )
    return tuple(cases)
