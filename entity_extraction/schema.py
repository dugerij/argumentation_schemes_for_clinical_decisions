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


@dataclass(frozen=True)
class ClinicalEntityMention:
    text: str
    concept: UMLSConcept | None
    start_char: int | None = None
    end_char: int | None = None
    category: str | None = None
    negated: bool = False
    temporality: str | None = None
    experiencer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClinicalKnowledgeContext:
    patient_id: str
    extracted_facts: list[str]
    past_failures_history: list[dict[str, Any]]
    side_effects_history: list[dict[str, Any]]
    contraindications_db: list[dict[str, Any]]
    umls_mappings: dict[str, UMLSConcept] = field(default_factory=dict)
