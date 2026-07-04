from __future__ import annotations

"""Discharge-note ingestion for the notes-first workflow."""

import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from helpers.progress import iter_progress, progress_message


@dataclass(frozen=True)
class MimicDischargeSubsetConfig:
    """Configuration for extracting plain-text evidence files from discharge notes."""

    csv_path: Path
    output_dir: Path
    limit: int | None = 25
    note_type: str | None = "DS"
    max_chars: int | None = None
    overwrite: bool = True


def _normalize(text: str | None) -> str:
    """Collapse whitespace and coerce missing values to an empty string."""

    return " ".join((text or "").split())


def _rows_from_csv(path: Path) -> Iterable[dict[str, str]]:
    """Yield rows from a CSV or gzipped CSV file as dictionaries."""

    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        yield from reader


def _build_note_text(row: dict[str, str], max_chars: int | None) -> tuple[str, bool]:
    """Render a note row into the on-disk evidence format used by the indexer."""

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


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    """Create the output directory and optionally clear prior note files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for old_file in output_dir.glob("*.txt"):
            old_file.unlink()


def _note_filename(row: dict[str, str], index: int) -> tuple[str, str, str, str]:
    """Build the stable note filename and the identifiers used to derive it."""

    note_id = _normalize(row.get("note_id")) or f"note_{index}"
    subject_id = _normalize(row.get("subject_id")) or "unknown_subject"
    hadm_id = _normalize(row.get("hadm_id")) or "unknown_hadm"
    filename = f"{subject_id}_{hadm_id}_{note_id}.txt".replace("/", "_")
    return filename, note_id, subject_id, hadm_id


def _write_note_file(
    row: dict[str, str],
    *,
    output_dir: Path,
    max_chars: int | None,
    index: int,
) -> tuple[Path, str, str, str]:
    """Write one extracted note file and return its path plus key identifiers."""

    filename, note_id, subject_id, hadm_id = _note_filename(row, index)
    content, _truncated = _build_note_text(row, max_chars)
    output_path = output_dir / filename
    output_path.write_text(content, encoding="utf-8")
    return output_path, note_id, subject_id, hadm_id


def _iter_matching_discharge_rows(
    rows: Iterable[dict[str, str]],
    *,
    note_type: str | None,
) -> Iterable[dict[str, str]]:
    """Filter raw rows down to the note type the workflow wants to keep."""

    for row in rows:
        if note_type and _normalize(row.get("note_type")).upper() != note_type.upper():
            continue
        yield row


def extract_mimic_discharge_subset(config: MimicDischargeSubsetConfig) -> list[Path]:
    """Extract discharge notes into plain-text files for indexing and retrieval.

    The source CSV remains the canonical downloaded dataset. This function creates
    lightweight `.txt` files under the evidence directory so downstream retrieval
    code can work with one note per file and a small manifest describing the run.
    """

    if not config.csv_path.exists():
        raise FileNotFoundError(f"Missing MIMIC discharge CSV: {config.csv_path}")

    _prepare_output_dir(config.output_dir, overwrite=config.overwrite)
    progress_message(f"Extracting MIMIC discharge notes from {config.csv_path} into {config.output_dir}")

    written: list[Path] = []
    rows = _iter_matching_discharge_rows(_rows_from_csv(config.csv_path), note_type=config.note_type)
    for row in iter_progress(rows, desc="MIMIC notes", unit="note"):
        output_path, _note_id, _subject_id, _hadm_id = _write_note_file(
            row,
            output_dir=config.output_dir,
            max_chars=config.max_chars,
            index=len(written) + 1,
        )
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
    (config.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    progress_message(f"Wrote {len(written)} note files to {config.output_dir}")
    return written
