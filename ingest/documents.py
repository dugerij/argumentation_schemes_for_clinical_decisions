from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceDocument:
    """Normalized document record passed into indexing."""

    id: str
    text: str
    source_type: str
    source_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def load_text_file(path: Path, source_type: str) -> SourceDocument:
    return SourceDocument(
        id=path.stem,
        text=path.read_text(encoding="utf-8"),
        source_type=source_type,
        source_path=path,
    )
