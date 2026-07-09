from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from retrieval.concepts.schema import ClinicalEntityMention


MedicalEntityType = Literal[
    "DISEASE",
    "SYMPTOM",
    "FINDING",
    "MEDICATION",
    "THERAPY",
    "LAB_TEST",
    "LAB_RESULT",
    "VITAL_SIGN",
    "PROCEDURE",
    "ANATOMY",
    "RISK_FACTOR",
    "CONTRAINDICATION",
    "ALLERGY",
    "CLINICAL_GOAL",
    "COMPLICATION",
    "TOPIC",
]

MedicalRelationType = Literal[
    "MENTIONS",
    "TREATS",
    "INDICATES",
    "SUPPORTS",
    "MONITORS",
    "HAS_RESULT",
    "CAUSES",
    "CONTRAINDICATES",
    "RECOMMENDS",
    "ASSOCIATED_WITH",
    "RISK_FOR",
    "CONTINUED",
    "HELD",
    "RESTARTED",
    "DISCONTINUED",
    "ADMINISTERED",
    "DISCHARGED_ON",
    "IMPROVES",
    "WORSENS",
    "PART_OF",
    "LOCATED_IN",
    "AFFECTS",
]

ENTITY_LABELS: tuple[str, ...] = (
    "DISEASE",
    "SYMPTOM",
    "FINDING",
    "MEDICATION",
    "THERAPY",
    "LAB_TEST",
    "LAB_RESULT",
    "VITAL_SIGN",
    "PROCEDURE",
    "ANATOMY",
    "RISK_FACTOR",
    "CONTRAINDICATION",
    "ALLERGY",
    "CLINICAL_GOAL",
    "COMPLICATION",
    "TOPIC",
)

RELATION_LABELS: tuple[str, ...] = (
    "MENTIONS",
    "TREATS",
    "INDICATES",
    "SUPPORTS",
    "MONITORS",
    "HAS_RESULT",
    "CAUSES",
    "CONTRAINDICATES",
    "RECOMMENDS",
    "ASSOCIATED_WITH",
    "RISK_FOR",
    "CONTINUED",
    "HELD",
    "RESTARTED",
    "DISCONTINUED",
    "ADMINISTERED",
    "DISCHARGED_ON",
    "IMPROVES",
    "WORSENS",
    "PART_OF",
    "LOCATED_IN",
    "AFFECTS",
)

UMLS_CATEGORY_TO_ENTITY: dict[str, str] = {
    "diagnosis": "DISEASE",
    "clinical_finding": "FINDING",
    "medication": "MEDICATION",
    "therapy_or_procedure": "THERAPY",
    "lab_or_measurement": "LAB_TEST",
    "anatomy": "ANATOMY",
    "biomedical_topic": "TOPIC",
}

ENTITY_RELATION_HINTS: dict[tuple[str, str], tuple[str, ...]] = {
    ("TOPIC", "TOPIC"): ("MENTIONS",),
    ("DISEASE", "MEDICATION"): ("TREATS", "ASSOCIATED_WITH"),
    ("MEDICATION", "DISEASE"): ("TREATS", "ASSOCIATED_WITH"),
    ("DISEASE", "THERAPY"): ("TREATS", "RECOMMENDS", "ASSOCIATED_WITH"),
    ("THERAPY", "DISEASE"): ("TREATS", "RECOMMENDS", "ASSOCIATED_WITH"),
    ("FINDING", "DISEASE"): ("INDICATES", "ASSOCIATED_WITH"),
    ("SYMPTOM", "DISEASE"): ("INDICATES", "ASSOCIATED_WITH"),
    ("DISEASE", "FINDING"): ("ASSOCIATED_WITH", "WORSENS"),
    ("DISEASE", "SYMPTOM"): ("ASSOCIATED_WITH", "WORSENS"),
    ("LAB_TEST", "DISEASE"): ("SUPPORTS", "MONITORS", "ASSOCIATED_WITH"),
    ("DISEASE", "LAB_TEST"): ("MONITORS", "ASSOCIATED_WITH"),
    ("LAB_TEST", "LAB_RESULT"): ("HAS_RESULT",),
    ("VITAL_SIGN", "LAB_RESULT"): ("HAS_RESULT",),
    ("MEDICATION", "FINDING"): ("CAUSES", "AFFECTS", "ASSOCIATED_WITH"),
    ("FINDING", "MEDICATION"): ("AFFECTS", "ASSOCIATED_WITH"),
    ("PROCEDURE", "DISEASE"): ("TREATS", "MONITORS", "ASSOCIATED_WITH"),
    ("DISEASE", "PROCEDURE"): ("RECOMMENDS", "ASSOCIATED_WITH"),
    ("RISK_FACTOR", "DISEASE"): ("RISK_FOR", "ASSOCIATED_WITH"),
    ("CONTRAINDICATION", "MEDICATION"): ("CONTRAINDICATES",),
    ("ALLERGY", "MEDICATION"): ("CONTRAINDICATES",),
    ("CLINICAL_GOAL", "THERAPY"): ("SUPPORTS", "RECOMMENDS"),
    ("CLINICAL_GOAL", "MEDICATION"): ("SUPPORTS", "RECOMMENDS"),
    ("COMPLICATION", "DISEASE"): ("ASSOCIATED_WITH", "WORSENS"),
    ("ANATOMY", "DISEASE"): ("LOCATED_IN", "PART_OF"),
}


@dataclass(frozen=True)
class ClinicalSchemaGuidance:
    validation_schema: list[tuple[str, str, str]]
    concept_hints_by_source: dict[str, str]
    concept_count_by_source: dict[str, int]
    candidate_count_by_source: dict[str, int]
    relation_count: int


def entity_type_for_category(category: str | None) -> str:
    if not category:
        return "TOPIC"
    return UMLS_CATEGORY_TO_ENTITY.get(category, "TOPIC")


def relation_hints_for_entity_types(source_type: str, target_type: str) -> tuple[str, ...]:
    if source_type == target_type:
        return ("ASSOCIATED_WITH",)
    return ENTITY_RELATION_HINTS.get((source_type, target_type), ())


def build_validation_schema(entity_types: set[str]) -> list[tuple[str, str, str]]:
    schema: set[tuple[str, str, str]] = set()
    observed = entity_types | {"TOPIC"}
    for source_type in observed:
        for target_type in observed:
            for relation in relation_hints_for_entity_types(source_type, target_type):
                schema.add((source_type, relation, target_type))
    return sorted(schema)


def format_concept_hint_block(
    mentions: list[ClinicalEntityMention],
    max_mentions: int = 20,
    max_relations: int = 20,
) -> str:
    if not mentions:
        return ""

    top_mentions = mentions[:max_mentions]
    lines = ["UMLS concept hints:"]
    for mention in top_mentions:
        concept = mention.concept
        if concept is None:
            continue
        entity_type = entity_type_for_category(concept.category)
        lines.append(
            f"- {entity_type}: {concept.preferred_term} "
            f"(CUI {concept.cui}, source {concept.source_vocabulary}, semantic type {concept.semantic_type})"
        )

    entity_types = {entity_type_for_category(mention.concept.category if mention.concept else None) for mention in top_mentions}
    relation_hints: list[tuple[str, str, str]] = []
    for source_type in sorted(entity_types):
        for target_type in sorted(entity_types):
            for relation in relation_hints_for_entity_types(source_type, target_type):
                triple = (source_type, relation, target_type)
                if triple not in relation_hints:
                    relation_hints.append(triple)
                if len(relation_hints) >= max_relations:
                    break
            if len(relation_hints) >= max_relations:
                break
        if len(relation_hints) >= max_relations:
            break

    if relation_hints:
        lines.append("Suggested relation families:")
        for source_type, relation, target_type in relation_hints:
            lines.append(f"- {source_type} {relation} {target_type}")

    return "\n".join(lines)
