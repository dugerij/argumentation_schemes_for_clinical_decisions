from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class UMLSConcept:
    cui: str
    preferred_term: str
    semantic_type: str
    source_vocabulary: str = "UMLS"
    source_code: str | None = None
    category: str | None = None
    aliases: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
