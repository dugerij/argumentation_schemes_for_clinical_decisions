from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class MimicDischargeSubsetConfig:
    csv_path: Path
    output_dir: Path
    limit: int | None = 25
    note_type: str | None = "DS"
    max_chars: int | None = 6000
    overwrite: bool = True


def _normalize(text: str | None) -> str:
    return " ".join((text or "").split())


def _rows_from_csv(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            yield row


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


def extract_mimic_discharge_subset(config: MimicDischargeSubsetConfig) -> list[Path]:
    if not config.csv_path.exists():
        raise FileNotFoundError(f"Missing MIMIC discharge CSV: {config.csv_path}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.overwrite:
        for old_file in config.output_dir.glob("*.txt"):
            old_file.unlink()

    written: list[Path] = []
    rows = _rows_from_csv(config.csv_path)

    for row in rows:
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

    return written
