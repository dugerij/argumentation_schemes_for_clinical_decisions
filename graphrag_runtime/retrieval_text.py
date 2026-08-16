"""Deterministic clinical text segmentation for retrieval embeddings."""

from __future__ import annotations

import re


CLAUSE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|[;\r\n]+")
CLINICAL_BLOCK_BOUNDARY = re.compile(
    r"(?=\b(?:ECG|EKG|EEG|EGD|Endoscopy|Echocardiogram|Echo|CXR|CT|MRI|"
    r"Imaging|Laboratory|Labs?|Pathology|Biopsy)\s*:)",
    re.IGNORECASE,
)


def atomic_retrieval_segments(
    value: str,
    *,
    maximum_words: int = 56,
    overlap_words: int = 8,
) -> tuple[str, ...]:
    """Split clinical prose into bounded, lossless-overlap retrieval segments."""
    if maximum_words < 8 or overlap_words < 0 or overlap_words >= maximum_words:
        raise ValueError("Atomic retrieval window configuration is invalid.")
    normalized = " ".join(str(value).split())
    if not normalized:
        return ()
    blocks = []
    for clause in CLAUSE_BOUNDARY.split(normalized):
        blocks.extend(CLINICAL_BLOCK_BOUNDARY.split(clause))
    output: list[str] = []
    stride = maximum_words - overlap_words
    for block in blocks:
        words = block.split()
        if not words:
            continue
        if len(words) <= maximum_words:
            output.append(" ".join(words))
            continue
        for start in range(0, len(words), stride):
            window = words[start:start + maximum_words]
            if not window:
                break
            output.append(" ".join(window))
            if start + maximum_words >= len(words):
                break
    return tuple(dict.fromkeys(output))
