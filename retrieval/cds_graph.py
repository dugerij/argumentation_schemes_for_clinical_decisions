import ast
from collections import Counter
from dataclasses import dataclass
import gzip
import io
import json
import pickle
from pathlib import Path
import re
import time
from typing import Any
import zipfile

import pandas as pd


CDS_GRAPH_FNAME = "cds_case_graph.pkl.gz"
CDS_GRAPH_MANIFEST_FNAME = "cds_case_graph_manifest.json"
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
class CdsCase:
    stay_id: int
    hpi: str
    patient_info: str
    initial_vitals: str
    tests: str
    past_medication: str
    chiefcomplaint: str
    triage: str
    primary_diagnosis: tuple[str, ...]
    secondary_diagnosis: tuple[str, ...]
    specialty_referral: tuple[str, ...]


@dataclass(frozen=True)
class CdsGraphStats:
    artifact_path: str
    manifest_path: str
    case_count: int
    diagnosis_case_count: int
    triage_case_count: int
    specialty_case_count: int
    token_count: int
    build_seconds: float
    artifact_bytes: int


def cds_graph_path(output_dir: Path) -> Path:
    return output_dir / CDS_GRAPH_FNAME


def cds_graph_manifest_path(output_dir: Path) -> Path:
    return output_dir / CDS_GRAPH_MANIFEST_FNAME


def _read_csv_from_outer_zip(zip_path: Path, member_suffix: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as archive:
        member_name = next(name for name in archive.namelist() if name.endswith(member_suffix))
        with archive.open(member_name) as raw_file:
            return pd.read_csv(raw_file)


def _read_csv_from_nested_zip(zip_path: Path, outer_member_suffix: str, inner_member_suffix: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as archive:
        outer_member_name = next(name for name in archive.namelist() if name.endswith(outer_member_suffix))
        nested_bytes = archive.read(outer_member_name)
    with zipfile.ZipFile(io.BytesIO(nested_bytes)) as nested_archive:
        inner_member_name = next(name for name in nested_archive.namelist() if name.endswith(inner_member_suffix))
        with nested_archive.open(inner_member_name) as inner_file:
            return pd.read_csv(inner_file)


def _parse_list_column(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return [text]
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [str(parsed).strip()]


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in TOKEN_RE.findall(text.lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def _build_case_text(record: dict[str, Any]) -> str:
    parts = [
        str(record.get("hpi") or ""),
        str(record.get("chiefcomplaint") or ""),
        str(record.get("patient_info") or ""),
        str(record.get("initial_vitals") or ""),
        str(record.get("tests") or ""),
        str(record.get("past_medication") or ""),
    ]
    return "\n".join(part for part in parts if part)


def _case_to_record(case: CdsCase) -> dict[str, Any]:
    return {
        "stay_id": case.stay_id,
        "hpi": case.hpi,
        "patient_info": case.patient_info,
        "initial_vitals": case.initial_vitals,
        "tests": case.tests,
        "past_medication": case.past_medication,
        "chiefcomplaint": case.chiefcomplaint,
        "triage": case.triage,
        "primary_diagnosis": list(case.primary_diagnosis),
        "secondary_diagnosis": list(case.secondary_diagnosis),
        "specialty_referral": list(case.specialty_referral),
    }


def _record_to_case(record: dict[str, Any]) -> CdsCase:
    return CdsCase(
        stay_id=int(record["stay_id"]),
        hpi=str(record.get("hpi") or ""),
        patient_info=str(record.get("patient_info") or ""),
        initial_vitals=str(record.get("initial_vitals") or ""),
        tests=str(record.get("tests") or ""),
        past_medication=str(record.get("past_medication") or ""),
        chiefcomplaint=str(record.get("chiefcomplaint") or ""),
        triage=str(record.get("triage") or ""),
        primary_diagnosis=tuple(str(item) for item in record.get("primary_diagnosis") or []),
        secondary_diagnosis=tuple(str(item) for item in record.get("secondary_diagnosis") or []),
        specialty_referral=tuple(str(item) for item in record.get("specialty_referral") or []),
    )


def build_cds_materialized_graph(zip_path: Path, output_dir: Path, *, overwrite: bool = True) -> CdsGraphStats:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = cds_graph_path(output_dir)
    manifest_path = cds_graph_manifest_path(output_dir)
    if artifact_path.exists() and not overwrite:
        raise FileExistsError(f"CDS graph already exists: {artifact_path}")

    started_at = time.perf_counter()

    clinical_data = _read_csv_from_nested_zip(zip_path, "clinical_data.csv.zip", "clinical_data.csv")
    diagnosis = _read_csv_from_outer_zip(zip_path, "diagnosis.csv")
    initial_assessment = _read_csv_from_outer_zip(zip_path, "initial_assessment_info.csv")
    specialty_referral = _read_csv_from_outer_zip(zip_path, "specialty_referral.csv")

    cases = diagnosis.merge(
        clinical_data[["stay_id", "tests", "past_medication"]],
        on="stay_id",
        how="left",
    ).merge(
        initial_assessment[["stay_id", "triage", "chiefcomplaint"]],
        on="stay_id",
        how="left",
    ).merge(
        specialty_referral[["stay_id", "specialty"]],
        on="stay_id",
        how="left",
    )

    token_to_stay_ids: dict[str, set[int]] = {}
    case_records: dict[int, dict[str, Any]] = {}
    diagnosis_label_counts: Counter[str] = Counter()
    triage_counts: Counter[str] = Counter()
    specialty_counts: Counter[str] = Counter()

    for _, row in cases.iterrows():
        case = CdsCase(
            stay_id=int(row["stay_id"]),
            hpi=str(row.get("HPI") or ""),
            patient_info=str(row.get("patient_info") or ""),
            initial_vitals=str(row.get("initial_vitals") or ""),
            tests=str(row.get("tests") or ""),
            past_medication=str(row.get("past_medication") or ""),
            chiefcomplaint=str(row.get("chiefcomplaint") or ""),
            triage=str(row.get("triage") or ""),
            primary_diagnosis=tuple(_parse_list_column(row.get("primary_diagnosis"))),
            secondary_diagnosis=tuple(_parse_list_column(row.get("secondary_diagnosis"))),
            specialty_referral=tuple(_parse_list_column(row.get("specialty"))),
        )
        case_records[case.stay_id] = _case_to_record(case)

        for label in case.primary_diagnosis:
            diagnosis_label_counts[label] += 1
        if case.triage:
            triage_counts[case.triage] += 1
        for label in case.specialty_referral:
            specialty_counts[label] += 1

        for token in _tokenize(_build_case_text(case_records[case.stay_id])):
            token_to_stay_ids.setdefault(token, set()).add(case.stay_id)

    payload = {
        "zip_path": str(zip_path),
        "cases": case_records,
        "token_to_stay_ids": {token: sorted(stay_ids) for token, stay_ids in token_to_stay_ids.items()},
        "diagnosis_label_counts": dict(diagnosis_label_counts),
        "triage_counts": dict(triage_counts),
        "specialty_counts": dict(specialty_counts),
        "built_at": time.time(),
    }

    with gzip.open(artifact_path, "wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)

    stats = CdsGraphStats(
        artifact_path=str(artifact_path),
        manifest_path=str(manifest_path),
        case_count=len(case_records),
        diagnosis_case_count=sum(1 for record in case_records.values() if record["primary_diagnosis"]),
        triage_case_count=sum(1 for record in case_records.values() if record["triage"]),
        specialty_case_count=sum(1 for record in case_records.values() if record["specialty_referral"]),
        token_count=len(payload["token_to_stay_ids"]),
        build_seconds=round(time.perf_counter() - started_at, 3),
        artifact_bytes=artifact_path.stat().st_size,
    )
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_path": stats.artifact_path,
                "zip_path": str(zip_path),
                "case_count": stats.case_count,
                "diagnosis_case_count": stats.diagnosis_case_count,
                "triage_case_count": stats.triage_case_count,
                "specialty_case_count": stats.specialty_case_count,
                "token_count": stats.token_count,
                "build_seconds": stats.build_seconds,
                "artifact_bytes": stats.artifact_bytes,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return stats


class CdsMaterializedGraphStore:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self._cases = {
            int(stay_id): _record_to_case(record)
            for stay_id, record in (payload.get("cases") or {}).items()
        }

    @classmethod
    def from_persist_dir(cls, output_dir: str | Path) -> "CdsMaterializedGraphStore":
        artifact_path = cds_graph_path(Path(output_dir))
        if not artifact_path.exists():
            raise FileNotFoundError(f"Missing CDS graph artifact: {artifact_path}")
        with gzip.open(artifact_path, "rb") as file:
            payload = pickle.load(file)
        return cls(payload)

    def all_cases(self) -> list[CdsCase]:
        return list(self._cases.values())

    def get_case(self, stay_id: int) -> CdsCase | None:
        return self._cases.get(int(stay_id))

    def search_stay_ids(self, text: str) -> list[int]:
        matched: set[int] = set()
        token_index = self.payload.get("token_to_stay_ids", {})
        for token in _tokenize(text):
            matched.update(int(stay_id) for stay_id in token_index.get(token, []))
        return sorted(matched)
