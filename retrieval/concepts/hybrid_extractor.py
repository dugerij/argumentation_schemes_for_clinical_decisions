from __future__ import annotations

import re
from typing import Any

from llama_index.core.graph_stores.types import EntityNode, Relation
from llama_index.core.indices.property_graph.base import KG_NODES_KEY, KG_RELATIONS_KEY
from llama_index.core.schema import MetadataMode

from retrieval.concepts.candidates import extract_candidate_terms
from retrieval.concepts.medical_schema import entity_type_for_category
from retrieval.property_graph import (
    ClinicalEntityNode,
    canonical_entity_id,
    merge_graph_properties,
)


SECTION_HEADERS: dict[str, str] = {
    "allergies": "allergies",
    "chief complaint": "chief_complaint",
    "history of present illness": "history_of_present_illness",
    "past medical history": "past_medical_history",
    "physical exam": "physical_exam",
    "pertinent results": "pertinent_results",
    "studies": "studies",
    "brief hospital course": "brief_hospital_course",
    "discharge medications": "discharge_medications",
    "discharge diagnosis": "discharge_diagnosis",
}

ABBREVIATION_EXPANSIONS: dict[str, str] = {
    "aki": "acute kidney injury",
    "af": "atrial fibrillation",
    "bnp": "brain natriuretic peptide",
    "bp": "blood pressure",
    "cad": "coronary artery disease",
    "ckd": "chronic kidney disease",
    "cr": "creatinine",
    "creat": "creatinine",
    "cxr": "chest x-ray",
    "dm": "diabetes mellitus",
    "dm2": "type 2 diabetes mellitus",
    "ecg": "electrocardiogram",
    "ekg": "electrocardiogram",
    "hfref": "heart failure with reduced ejection fraction",
    "hr": "heart rate",
    "ntg": "nitroglycerin",
    "osa": "obstructive sleep apnea",
    "paf": "atrial fibrillation",
    "rr": "respiratory rate",
    "sao2": "oxygen saturation",
}

LAB_ALIASES: dict[str, tuple[str, ...]] = {
    "brain natriuretic peptide": ("bnp", "probnp"),
    "calcium": ("ca", "calcium"),
    "chloride": ("cl", "chloride"),
    "creatinine": ("cr", "creat", "creatinine"),
    "glucose": ("glucose",),
    "hemoglobin": ("hgb", "hemoglobin"),
    "international normalized ratio": ("inr",),
    "magnesium": ("mg", "magnesium"),
    "phosphate": ("phos", "phosphate"),
    "platelet count": ("plt", "platelets"),
    "potassium": ("k", "potassium"),
    "sodium": ("na", "sodium"),
    "thyroid stimulating hormone": ("tsh",),
    "troponin": ("troponin", "ctropnt", "tnt"),
    "urea nitrogen": ("urean", "bun"),
    "white blood cell count": ("wbc",),
}

VITAL_PATTERNS: tuple[tuple[str, str], ...] = (
    ("blood pressure", r"\bBP\s+(?P<value>\d{2,3}/\d{2,3})\b"),
    ("heart rate", r"\bHR\s+(?P<value>\d{2,3})\b"),
    ("respiratory rate", r"\bRR\s+(?P<value>\d{1,3})\b"),
    ("oxygen saturation", r"\bSaO2\s+(?P<value>\d{1,3}%?)\b"),
    ("temperature", r"\bT\s+(?P<value>\d{2,3}(?:\.\d)?)\b"),
)

ACTION_LABELS: tuple[tuple[str, str], ...] = (
    ("discharged on", "DISCHARGED_ON"),
    ("restarted", "RESTARTED"),
    ("continued", "CONTINUED"),
    ("held", "HELD"),
    ("discontinued", "DISCONTINUED"),
    ("stopped", "DISCONTINUED"),
    ("given", "ADMINISTERED"),
    ("received", "ADMINISTERED"),
    ("started", "ADMINISTERED"),
    ("diuresed with", "ADMINISTERED"),
)

POSITIVE_ACTIONS = {"ADMINISTERED", "CONTINUED", "DISCHARGED_ON", "RESTARTED"}
NEGATIVE_ACTIONS = {"DISCONTINUED", "HELD"}
LAB_VALUE_RE = r"(?P<value><0\.01|[<>]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?%?)\*?"
CONDITION_TRIGGER_TERMS = {
    "disease",
    "failure",
    "fibrillation",
    "hypertension",
    "injury",
    "infection",
    "pain",
    "syndrome",
    "diabetes",
    "edema",
}
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9/-]*")


def _compact_text(text: str, limit: int = 180) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3].rstrip()}..."


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        key = value.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(value.strip())
    return deduped


class HybridClinicalPathExtractor:
    """Deterministic clinical graph extractor layered under optional LLM enrichment.

    The extractor favors local UMLS normalization for entity discovery, then adds
    explicit chunk-to-entity and entity-to-entity relations for clinical content
    that the schema-only LLM path was missing.
    """

    def __init__(
        self,
        *,
        umls_client: Any | None = None,
        candidate_limit: int = 24,
    ) -> None:
        self.umls_client = umls_client
        self.candidate_limit = candidate_limit
        self._lab_patterns = self._compile_lab_patterns()

    def __call__(self, nodes, **_: Any):
        for node in nodes:
            self._extract_into_node(node)
        return list(nodes)

    def _extract_into_node(self, node: Any) -> None:
        text = node.get_content(metadata_mode=MetadataMode.NONE) if hasattr(node, "get_content") else getattr(node, "text", "")
        metadata = getattr(node, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            node.metadata = metadata

        existing_nodes = list(metadata.get(KG_NODES_KEY, []))
        existing_relations = list(metadata.get(KG_RELATIONS_KEY, []))

        source_name = str(metadata.get("source_name") or "")
        source_path = str(metadata.get("source_path") or "")
        entity_nodes: dict[str, EntityNode] = {}
        relation_keys: set[tuple[str, str, str]] = set()
        relation_objects: dict[tuple[str, str, str], Relation] = {}
        primary_section = self._primary_section(text)

        for existing in existing_nodes:
            if not isinstance(existing, EntityNode):
                continue
            normalized_name = " ".join(existing.name.split()).strip()
            canonical_id = getattr(existing, "canonical_id", None) or canonical_entity_id(existing.label, normalized_name)
            entity_nodes[canonical_id] = ClinicalEntityNode(
                name=normalized_name,
                label=existing.label,
                canonical_id=canonical_id,
                properties=dict(existing.properties or {}),
            )
        for existing in existing_relations:
            key = (existing.label, existing.source_id, existing.target_id)
            relation_keys.add(key)
            relation_objects[key] = existing

        def ensure_entity(name: str, label: str, **properties: Any) -> EntityNode:
            normalized_name = " ".join(name.split()).strip()
            internal_id = canonical_entity_id(label, normalized_name)
            entity_properties = {k: v for k, v in properties.items() if v is not None}
            entity_properties["mention_count"] = int(entity_properties.get("mention_count", 1) or 1)
            if source_name:
                entity_properties.setdefault("source_name", source_name)
                entity_properties["source_names"] = [source_name]
            if source_path:
                entity_properties.setdefault("source_path", source_path)
                entity_properties["source_paths"] = [source_path]
            if entity_properties.get("section"):
                entity_properties["sections"] = [entity_properties["section"]]

            if internal_id in entity_nodes:
                entity_nodes[internal_id].properties = merge_graph_properties(
                    entity_nodes[internal_id].properties,
                    entity_properties,
                )
                return entity_nodes[internal_id]
            entity = ClinicalEntityNode(
                name=normalized_name,
                label=label,
                canonical_id=internal_id,
                properties=entity_properties,
            )
            entity_nodes[internal_id] = entity
            return entity

        def add_relation(label: str, source_id: str, target_id: str, **properties: Any) -> None:
            key = (label, source_id, target_id)
            relation_properties = {k: v for k, v in properties.items() if v is not None}
            if source_name:
                relation_properties.setdefault("source_name", source_name)
                relation_properties["source_names"] = [source_name]
            if source_path:
                relation_properties.setdefault("source_path", source_path)
                relation_properties["source_paths"] = [source_path]
            if relation_properties.get("section"):
                relation_properties["sections"] = [relation_properties["section"]]
            if relation_properties.get("evidence"):
                relation_properties["evidence_count"] = int(relation_properties.get("evidence_count", 1) or 1)
                relation_properties["evidence_samples"] = [relation_properties["evidence"]]
            if key in relation_keys:
                relation_objects[key].properties = merge_graph_properties(
                    relation_objects[key].properties,
                    relation_properties,
                )
                return
            relation_keys.add(key)
            relation_objects[key] = Relation(
                label=label,
                source_id=source_id,
                target_id=target_id,
                properties=relation_properties,
            )

        condition_entities = self._extract_umls_entities(
            text,
            allowed_categories={"clinical_finding", "diagnosis"},
            section=primary_section,
            ensure_entity=ensure_entity,
        )
        medication_entities = self._extract_umls_entities(
            text,
            allowed_categories={"medication", "therapy_or_procedure"},
            section=primary_section,
            ensure_entity=ensure_entity,
        )

        for entity in [*condition_entities, *medication_entities]:
            add_relation(
                "MENTIONS",
                node.id_,
                entity.id,
                section=entity.properties.get("section"),
            )

        for line in text.splitlines():
            self._extract_lab_results_from_line(
                line,
                node_id=node.id_,
                section=primary_section,
                ensure_entity=ensure_entity,
                add_relation=add_relation,
            )

        self._extract_vitals(
            text,
            node_id=node.id_,
            section=primary_section,
            ensure_entity=ensure_entity,
            add_relation=add_relation,
        )

        for sentence in self._sentences(text):
            action_matches = [label for phrase, label in ACTION_LABELS if phrase in sentence.lower()]
            if not action_matches:
                continue

            sentence_conditions = self._extract_umls_entities(
                sentence,
                allowed_categories={"clinical_finding", "diagnosis"},
                section=primary_section,
                ensure_entity=ensure_entity,
            )
            sentence_medications = self._extract_umls_entities(
                sentence,
                allowed_categories={"medication", "therapy_or_procedure"},
                section=primary_section,
                ensure_entity=ensure_entity,
            )
            if not sentence_medications:
                continue

            evidence = _compact_text(sentence)
            for medication in sentence_medications:
                add_relation(
                    "MENTIONS",
                    node.id_,
                    medication.id,
                    section=primary_section,
                    evidence=evidence,
                )
                for action_label in action_matches:
                    add_relation(
                        action_label,
                        node.id_,
                        medication.id,
                        section=primary_section,
                        evidence=evidence,
                    )
                    for condition in sentence_conditions:
                        if action_label in POSITIVE_ACTIONS:
                            add_relation(
                                "TREATS",
                                medication.id,
                                condition.id,
                                section=primary_section,
                                evidence=evidence,
                            )
                        elif action_label in NEGATIVE_ACTIONS:
                            add_relation(
                                "CONTRAINDICATES",
                                condition.id,
                                medication.id,
                                section=primary_section,
                                evidence=evidence,
                            )

        metadata["hybrid_entity_count"] = len(entity_nodes)
        metadata["hybrid_relation_count"] = len(relation_objects)
        metadata["hybrid_umls_concept_count"] = len(
            {
                entity.properties.get("cui")
                for entity in entity_nodes.values()
                if entity.properties.get("cui")
            }
        )
        metadata[KG_NODES_KEY] = list(entity_nodes.values())
        metadata[KG_RELATIONS_KEY] = list(relation_objects.values())

    def _extract_umls_entities(
        self,
        text: str,
        *,
        allowed_categories: set[str],
        section: str | None,
        ensure_entity,
    ) -> list[EntityNode]:
        if self.umls_client is None:
            return []

        candidates = extract_candidate_terms(text, limit=self.candidate_limit)
        if allowed_categories & {"clinical_finding", "diagnosis"}:
            candidates = [*self._condition_phrase_candidates(text), *candidates]
        abbreviation_terms = [
            ABBREVIATION_EXPANSIONS[token.lower()]
            for token in re.findall(r"\b[A-Za-z][A-Za-z0-9/-]{1,8}\b", text)
            if token.lower() in ABBREVIATION_EXPANSIONS
        ]
        token_terms = [
            token
            for token in TOKEN_RE.findall(text)
            if len(token) >= 5
        ]
        entities: list[EntityNode] = []

        for candidate in _dedupe_preserve_order([*abbreviation_terms, *token_terms, *candidates]):
            concept = self._best_match(candidate)
            if concept is None or concept.category not in allowed_categories:
                continue
            label = entity_type_for_category(concept.category)
            entity = ensure_entity(
                concept.preferred_term or candidate,
                label,
                cui=concept.cui,
                semantic_type=concept.semantic_type,
                source_vocabulary=concept.source_vocabulary,
                section=section,
                category=concept.category,
            )
            entities.append(entity)

        deduped: dict[str, EntityNode] = {}
        for entity in entities:
            deduped[str(getattr(entity, "id", f"{entity.label}:{entity.name}"))] = entity
        return list(deduped.values())

    def _best_match(self, term: str):
        variants = [term]
        lowered = term.strip().lower()
        expanded = ABBREVIATION_EXPANSIONS.get(lowered)
        if expanded:
            variants.append(expanded)
        variants.extend(
            [
                term.replace("/", " "),
                term.replace("-", " "),
            ]
        )

        for variant in _dedupe_preserve_order(variants):
            concept = self.umls_client.best_match(variant)
            if concept is not None:
                return concept
        return None

    def _extract_lab_results_from_line(
        self,
        line: str,
        *,
        node_id: str,
        section: str | None,
        ensure_entity,
        add_relation,
    ) -> None:
        for canonical_name, pattern in self._lab_patterns:
            for match in pattern.finditer(line):
                value = match.group("value")
                lab_concept = self._best_match(canonical_name)
                lab_name = lab_concept.preferred_term if lab_concept is not None else canonical_name.title()
                lab_node = ensure_entity(
                    lab_name,
                    "LAB_TEST",
                    cui=(lab_concept.cui if lab_concept is not None else None),
                    semantic_type=(lab_concept.semantic_type if lab_concept is not None else None),
                    source_vocabulary=(lab_concept.source_vocabulary if lab_concept is not None else None),
                    section=section,
                )
                result_node = ensure_entity(
                    f"{lab_name} = {value}",
                    "LAB_RESULT",
                    test_name=lab_name,
                    value=value,
                    section=section,
                )
                add_relation("MENTIONS", node_id, lab_node.id, section=section)
                add_relation("MENTIONS", node_id, result_node.id, section=section)
                add_relation("HAS_RESULT", lab_node.id, result_node.id, section=section)

    def _extract_vitals(
        self,
        text: str,
        *,
        node_id: str,
        section: str | None,
        ensure_entity,
        add_relation,
    ) -> None:
        for canonical_name, raw_pattern in VITAL_PATTERNS:
            pattern = re.compile(raw_pattern, flags=re.IGNORECASE)
            for match in pattern.finditer(text):
                value = match.group("value")
                vital_node = ensure_entity(
                    canonical_name.title(),
                    "VITAL_SIGN",
                    section=section,
                )
                result_node = ensure_entity(
                    f"{canonical_name.title()} = {value}",
                    "LAB_RESULT",
                    test_name=canonical_name.title(),
                    value=value,
                    section=section,
                )
                add_relation("MENTIONS", node_id, vital_node.id, section=section)
                add_relation("MENTIONS", node_id, result_node.id, section=section)
                add_relation("HAS_RESULT", vital_node.id, result_node.id, section=section)

    def _primary_section(self, text: str) -> str | None:
        for line in text.splitlines():
            normalized = line.strip().rstrip(":").lower()
            if normalized in SECTION_HEADERS:
                return SECTION_HEADERS[normalized]
        return None

    def _compile_lab_patterns(self) -> list[tuple[str, re.Pattern[str]]]:
        patterns: list[tuple[str, re.Pattern[str]]] = []
        for canonical_name, aliases in LAB_ALIASES.items():
            alias_pattern = "|".join(re.escape(alias) for alias in aliases)
            patterns.append(
                (
                    canonical_name,
                    re.compile(
                        rf"\b(?:{alias_pattern})\b\s*[-: ]\s*{LAB_VALUE_RE}",
                        flags=re.IGNORECASE,
                    ),
                )
            )
        return patterns

    def _sentences(self, text: str) -> list[str]:
        sentences = re.split(r"(?<=[\n.;])\s+", text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def _condition_phrase_candidates(self, text: str) -> list[str]:
        tokens = TOKEN_RE.findall(text)
        phrases: list[str] = []
        max_span = 5
        for start in range(len(tokens)):
            for span in range(2, max_span + 1):
                window = tokens[start : start + span]
                if len(window) != span:
                    continue
                lowered = [token.lower() for token in window]
                if not any(token in CONDITION_TRIGGER_TERMS for token in lowered):
                    continue
                phrases.append(" ".join(window))
        phrases.sort(key=lambda value: (-len(value.split()), -len(value), value.lower()))
        return _dedupe_preserve_order(phrases[: self.candidate_limit])
