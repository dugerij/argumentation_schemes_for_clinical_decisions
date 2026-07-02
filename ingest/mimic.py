from __future__ import annotations

import csv
import json
import re
import zipfile
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path
from typing import Iterable

from helpers.clinical_domains import DomainMatcher, count_domain_hits, normalize_domain_name
from helpers.progress import iter_progress, progress_message


@dataclass(frozen=True)
class MimicDischargeSubsetConfig:
    csv_path: Path
    output_dir: Path
    limit: int | None = 25
    note_type: str | None = "DS"
    max_chars: int | None = 6000
    overwrite: bool = True


@dataclass(frozen=True)
class MimicExtCardiovascularSubsetConfig:
    dataset_dir: Path
    output_dir: Path
    limit: int | None = 100
    max_chars: int | None = 4000
    overwrite: bool = True


@dataclass(frozen=True)
class MimicDischargeDomainSubsetConfig:
    csv_path: Path
    output_dir: Path
    domain: str
    limit: int | None = None
    note_type: str | None = "DS"
    max_chars: int | None = 6000
    min_domain_hits: int = 1
    overwrite: bool = True


def _normalize(text: str | None) -> str:
    return " ".join((text or "").split())


def _rows_from_csv(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            yield row


def _rows_from_zipped_csv(path: Path) -> Iterable[dict[str, str]]:
    with zipfile.ZipFile(path) as archive:
        csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv") and "/._" not in name]
        if not csv_members:
            raise FileNotFoundError(f"No CSV member found in zip archive: {path}")
        with archive.open(csv_members[0]) as raw_file:
            reader = csv.DictReader(TextIOWrapper(raw_file, encoding="utf-8", newline=""))
            for row in reader:
                yield row


def _first_icd_token(text: str | None) -> str:
    raw = (text or "").strip().upper()
    if not raw:
        return ""
    token = re.split(r"[^A-Z0-9.]+", raw, maxsplit=1)[0]
    return token


def _is_cardiovascular_icd(code: str | None, version: str | None) -> bool:
    token = _first_icd_token(code)
    if not token:
        return False

    normalized_version = _normalize(version)
    if normalized_version == "10":
        return token.startswith("I")

    if normalized_version == "9":
        digits = "".join(ch for ch in token if ch.isdigit())
        if len(digits) < 3:
            return False
        major = int(digits[:3])
        return 390 <= major <= 459

    return False


def _build_note_text(row: dict[str, str], max_chars: int | None) -> tuple[str, bool]:
    text = row.get("text", "") or ""
    truncated = max_chars is not None and len(text) > max_chars
    if truncated:
        text = text[:max_chars].rstrip()

    header = [
        "MIMIC-IV discharge note",
        f"note_id: {_normalize(row.get('note_id'))}",
        f"subject_id: {_normalize(row.get('subject_id'))}",
        f"hadm_id: {_normalize(row.get('hadm_id'))}",
        f"note_type: {_normalize(row.get('note_type'))}",
        f"note_seq: {_normalize(row.get('note_seq'))}",
        f"charttime: {_normalize(row.get('charttime'))}",
        f"storetime: {_normalize(row.get('storetime'))}",
        f"truncated: {str(truncated).lower()}",
        "",
        "SOURCE TEXT:",
    ]
    return "\n".join(header + [text]), truncated


def _build_ext_case_text(
    clinical_row: dict[str, str],
    assessment_row: dict[str, str],
    max_chars: int | None,
) -> tuple[str, bool]:
    full_text = (clinical_row.get("text", "") or "").replace("<br>", "").replace("</br>", "").replace("<comma>", ",")
    body = full_text.strip()

    if not body:
        hpi = clinical_row.get("HPI", "") or ""
        diagnosis_text = clinical_row.get("diagnosis", "") or ""
        primary_diagnosis = clinical_row.get("primary_diagnosis", "") or ""
        secondary_diagnosis = clinical_row.get("secondary_diagnosis", "") or ""
        body = "\n\n".join(
            part.strip()
            for part in [
                "HPI:\n" + hpi.strip(),
                "Tests:\n" + (clinical_row.get("tests", "") or "").strip(),
                "Past medication:\n" + (clinical_row.get("past_medication", "") or "").strip(),
                "Diagnosis text:\n" + diagnosis_text.strip(),
                "Primary diagnosis list:\n" + primary_diagnosis.strip(),
                "Secondary diagnosis list:\n" + secondary_diagnosis.strip(),
            ]
            if part.strip()
        )

    truncated = max_chars is not None and len(body) > max_chars
    if truncated:
        body = body[:max_chars].rstrip()

    header = [
        "MIMIC-IV-Ext cardiovascular case",
        f"stay_id: {_normalize(clinical_row.get('stay_id'))}",
        f"triage: {_normalize(assessment_row.get('triage'))}",
        f"pain: {_normalize(assessment_row.get('pain'))}",
        f"chiefcomplaint: {_normalize(assessment_row.get('chiefcomplaint'))}",
        f"arrival_transport: {_normalize(assessment_row.get('arrival_transport'))}",
        f"disposition: {_normalize(assessment_row.get('disposition'))}",
        f"icd_code: {_normalize(assessment_row.get('icd_code'))}",
        f"icd_title: {_normalize(assessment_row.get('icd_title'))}",
        f"icd_version: {_normalize(assessment_row.get('icd_version'))}",
        f"truncated: {str(truncated).lower()}",
        "",
        "SOURCE TEXT:",
    ]
    return "\n".join(header + [body]), truncated


def extract_mimic_discharge_subset(config: MimicDischargeSubsetConfig) -> list[Path]:
    if not config.csv_path.exists():
        raise FileNotFoundError(f"Missing MIMIC discharge CSV: {config.csv_path}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.overwrite:
        for old_file in config.output_dir.glob("*.txt"):
            old_file.unlink()

    written: list[Path] = []
    rows = _rows_from_csv(config.csv_path)
    progress_message(
        f"Extracting MIMIC discharge notes from {config.csv_path} into {config.output_dir}"
    )

    for row in iter_progress(rows, desc="MIMIC notes", unit="note"):
        if config.note_type and _normalize(row.get("note_type")).upper() != config.note_type.upper():
            continue

        note_id = _normalize(row.get("note_id")) or f"note_{len(written) + 1}"
        subject_id = _normalize(row.get("subject_id")) or "unknown_subject"
        hadm_id = _normalize(row.get("hadm_id")) or "unknown_hadm"
        filename = f"{subject_id}_{hadm_id}_{note_id}.txt".replace("/", "_")
        content, _truncated = _build_note_text(row, config.max_chars)
        output_path = config.output_dir / filename
        output_path.write_text(content, encoding="utf-8")
        written.append(output_path)

        if config.limit is not None and len(written) >= config.limit:
            break

    manifest = {
        "csv_path": config.csv_path.as_posix(),
        "output_dir": config.output_dir.as_posix(),
        "limit": config.limit,
        "note_type": config.note_type,
        "max_chars": config.max_chars,
        "document_count": len(written),
        "files": [path.name for path in written],
    }
    (config.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    progress_message(f"Wrote {len(written)} note files to {config.output_dir}")

    return written


def extract_mimic_discharge_domain_subset(
    config: MimicDischargeDomainSubsetConfig,
    *,
    matcher: DomainMatcher | None = None,
) -> list[Path]:
    if not config.csv_path.exists():
        raise FileNotFoundError(f"Missing MIMIC discharge CSV: {config.csv_path}")

    domain = normalize_domain_name(config.domain)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.overwrite:
        for old_file in config.output_dir.glob("*.txt"):
            old_file.unlink()

    written: list[Path] = []
    manifest_rows: list[dict[str, object]] = []
    rows = _rows_from_csv(config.csv_path)
    progress_message(
        f"Extracting {domain} MIMIC discharge notes from {config.csv_path} into {config.output_dir}"
    )

    for row in iter_progress(rows, desc=f"{domain} MIMIC notes", unit="note"):
        if config.note_type and _normalize(row.get("note_type")).upper() != config.note_type.upper():
            continue

        if matcher is None:
            hit_count = count_domain_hits(row.get("text", "") or "", domain)
            matched_terms: list[str] = []
        else:
            hit_count, matched_terms = matcher.match_details(row.get("text", "") or "")
        if hit_count < config.min_domain_hits:
            continue

        note_id = _normalize(row.get("note_id")) or f"note_{len(written) + 1}"
        subject_id = _normalize(row.get("subject_id")) or "unknown_subject"
        hadm_id = _normalize(row.get("hadm_id")) or "unknown_hadm"
        filename = f"{subject_id}_{hadm_id}_{note_id}.txt".replace("/", "_")
        content, _truncated = _build_note_text(row, config.max_chars)
        output_path = config.output_dir / filename
        output_path.write_text(content, encoding="utf-8")
        written.append(output_path)
        manifest_rows.append(
            {
                "file": output_path.name,
                "note_id": note_id,
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "domain_hit_count": hit_count,
                "matched_terms": matched_terms[:20],
            }
        )

        if config.limit is not None and len(written) >= config.limit:
            break

    manifest = {
        "csv_path": config.csv_path.as_posix(),
        "output_dir": config.output_dir.as_posix(),
        "domain": domain,
        "limit": config.limit,
        "note_type": config.note_type,
        "max_chars": config.max_chars,
        "min_domain_hits": config.min_domain_hits,
        "document_count": len(written),
        "files": manifest_rows,
    }
    (config.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    progress_message(f"Wrote {len(written)} {domain} note files to {config.output_dir}")
    return written


def extract_mimic_ext_cardiovascular_subset(config: MimicExtCardiovascularSubsetConfig) -> list[Path]:
    dataset_dir = config.dataset_dir
    assessment_path = dataset_dir / "initial_assessment_info.csv"
    clinical_zip_path = dataset_dir / "clinical_data.csv.zip"

    if not assessment_path.exists():
        raise FileNotFoundError(f"Missing MIMIC-IV-Ext assessment CSV: {assessment_path}")
    if not clinical_zip_path.exists():
        raise FileNotFoundError(f"Missing MIMIC-IV-Ext clinical data zip: {clinical_zip_path}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.overwrite:
        for old_file in config.output_dir.glob("*.txt"):
            old_file.unlink()

    progress_message(
        f"Extracting cardiovascular cases from {dataset_dir} into {config.output_dir}"
    )

    cardiovascular_cases: dict[str, dict[str, str]] = {}
    for row in iter_progress(
        _rows_from_csv(assessment_path),
        desc="MIMIC-IV-Ext assessments",
        unit="row",
    ):
        if _is_cardiovascular_icd(row.get("icd_code"), row.get("icd_version")):
            stay_id = _normalize(row.get("stay_id"))
            if stay_id:
                cardiovascular_cases[stay_id] = row

    written: list[Path] = []
    matched_rows = 0
    for row in iter_progress(
        _rows_from_zipped_csv(clinical_zip_path),
        desc="MIMIC-IV-Ext clinical rows",
        unit="row",
    ):
        stay_id = _normalize(row.get("stay_id"))
        assessment = cardiovascular_cases.get(stay_id)
        if assessment is None:
            continue

        matched_rows += 1
        content, _truncated = _build_ext_case_text(row, assessment, config.max_chars)
        filename = f"{stay_id}_{_first_icd_token(assessment.get('icd_code')) or 'unknown_icd'}.txt"
        output_path = config.output_dir / filename.replace("/", "_")
        output_path.write_text(content, encoding="utf-8")
        written.append(output_path)

        if config.limit is not None and len(written) >= config.limit:
            break

    manifest = {
        "dataset_dir": dataset_dir.as_posix(),
        "output_dir": config.output_dir.as_posix(),
        "limit": config.limit,
        "max_chars": config.max_chars,
        "cardiovascular_case_count": len(cardiovascular_cases),
        "matched_diagnosis_rows": matched_rows,
        "document_count": len(written),
        "selection_rule": "ICD-10 I00-I99 or ICD-9 390-459 from initial_assessment_info.csv",
        "files": [path.name for path in written],
    }
    (config.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    progress_message(f"Wrote {len(written)} cardiovascular case files to {config.output_dir}")
    return written
