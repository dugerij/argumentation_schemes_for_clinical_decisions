from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

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
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}

def tokenize(text: str) -> set[str]:
    """Return lowercase word tokens with common stopwords removed."""

    return {
        token
        for token in TOKEN_RE.findall(text.lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def load_questions_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_questions_jsonl(path: Path, questions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for item in questions:
            file.write(json.dumps(item, ensure_ascii=True) + "\n")


def question_text(item: dict[str, Any]) -> str:
    parts = [str(item.get("question", ""))]
    options = item.get("options") or {}
    if isinstance(options, dict):
        parts.extend(str(value) for value in options.values())
    phrases = item.get("metamap_phrases") or []
    if isinstance(phrases, list):
        parts.extend(str(value) for value in phrases)
    return "\n".join(parts)


def load_note_vocabulary(input_dir: Path) -> set[str]:
    vocabulary: set[str] = set()
    for path in sorted(input_dir.glob("*.txt")):
        vocabulary.update(tokenize(path.read_text(encoding="utf-8")))
    return vocabulary


def filter_questions_by_note_overlap(
    *,
    questions: list[dict[str, Any]],
    input_dir: Path,
    max_questions: int,
    min_overlap_terms: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    note_vocabulary = load_note_vocabulary(input_dir)
    scored: list[tuple[int, dict[str, Any], list[str]]] = []
    rejected: list[dict[str, Any]] = []

    for item in questions:
        overlap = sorted(tokenize(question_text(item)) & note_vocabulary)
        if len(overlap) >= min_overlap_terms:
            scored.append((len(overlap), item, overlap[:20]))
        else:
            rejected.append(
                {
                    "question": item.get("question"),
                    "overlap_count": len(overlap),
                    "overlap_terms": overlap[:20],
                }
            )

    scored.sort(key=lambda row: (-row[0], str(row[1].get("question", ""))))
    retained = []
    retained_metadata = []
    for overlap_count, item, overlap_terms in scored[:max_questions]:
        retained.append(item)
        retained_metadata.append(
            {
                "question": item.get("question"),
                "overlap_count": overlap_count,
                "overlap_terms": overlap_terms,
            }
        )
    return retained, retained_metadata + rejected
