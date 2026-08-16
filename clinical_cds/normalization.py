from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from clinical_cds.direct import label_key
from clinical_cds.schema import DiagnosticGraph
from clinical_cds.terminology.candidates import extract_candidate_terms
from clinical_cds.terminology.local_umls import LocalUMLSClient
from clinical_cds.terminology.schema import UMLSConcept


GRAPH_UMLS_CROSSWALK_ID = "graph-umls-crosswalk-v1"
LEXICAL_DIAGNOSIS_ALIASES = {
    "nonstelevationmyocardialinfarction": "nstemi",
    "nonstelevationmyocardialinfarctionnstemi": "nstemi",
    "lowriskpulmonaryembolism": "lowriskpe",
    "pulmonaryembolismlowrisk": "lowriskpe",
    "stemi": "stemiacs",
    "copdasthma": "asthmacopd",
    "severeasthmaexacerbation": "severeasthma",
    "pituitarymacroadenoma": "pituitarymacroadenomas",
    "pituitarymicroadenoma": "pituitarymicroadenomas",
    "gastroesophagealrefluxdisease": "gastrooesophagealrefluxdisease",
}
DIAGNOSIS_FAMILY_ALIASES = {
    "rrms": "multiplesclerosis",
    "relapsingremittingmultiplesclerosis": "multiplesclerosis",
    "lowriskpe": "pulmonaryembolism",
    "submassivepe": "pulmonaryembolism",
    "bacterialpneumonia": "pneumonia",
    "viralpneumonia": "pneumonia",
    "severecopd": "copd",
    "copdexacerbation": "copd",
    "acuteexacerbationofcopd": "copd",
    "asthmacopd": "copd",
    "copdasthma": "copd",
    "mildcopd": "copd",
    "moderatecopd": "copd",
    "veryseverecopd": "copd",
    "hfpef": "heartfailure",
    "hfmref": "heartfailure",
    "hfref": "heartfailure",
    "pituitarymacroadenomas": "pituitaryadenoma",
    "pituitarymicroadenomas": "pituitaryadenoma",
    "pituitarysilenadenomas": "pituitaryadenoma",
    "pituitarysilentadenomas": "pituitaryadenoma",
    "nsteacs": "acutecoronarysyndrome",
    "nstemi": "acutecoronarysyndrome",
    "stemiacs": "acutecoronarysyndrome",
    "ua": "acutecoronarysyndrome",
}
DIAGNOSIS_MODIFIERS = {
    "bacterialpneumonia": frozenset({"bacterial"}),
    "viralpneumonia": frozenset({"viral"}),
    "relapsingremittingmultiplesclerosis": frozenset({"relapsing_remitting"}),
    "rrms": frozenset({"relapsing_remitting"}),
    "lowriskpe": frozenset({"low_risk"}),
    "submassivepe": frozenset({"intermediate_risk"}),
    "severecopd": frozenset({"severe"}),
    "copdexacerbation": frozenset({"exacerbation"}),
    "acuteexacerbationofcopd": frozenset({"acute", "exacerbation"}),
    "mildcopd": frozenset({"mild"}),
    "moderatecopd": frozenset({"moderate"}),
    "veryseverecopd": frozenset({"very_severe"}),
    "massivepe": frozenset({"high_risk"}),
    "hfpef": frozenset({"preserved_ef"}),
    "hfmref": frozenset({"mildly_reduced_ef"}),
    "hfref": frozenset({"reduced_ef"}),
    "typeidiabetes": frozenset({"type_1"}),
    "typeiidiabetes": frozenset({"type_2"}),
    "asthmacopd": frozenset({"overlap"}),
    "nsteacs": frozenset({"non_st_elevation"}),
    "nstemi": frozenset({"non_st_elevation"}),
    "stemiacs": frozenset({"st_elevation"}),
    "ua": frozenset({"unstable_angina"}),
}
CONTRADICTORY_MODIFIER_GROUPS = (
    frozenset({"bacterial", "viral"}),
    frozenset({"low_risk", "intermediate_risk", "high_risk"}),
    frozenset({"mild", "moderate", "severe", "very_severe"}),
    frozenset({"preserved_ef", "mildly_reduced_ef", "reduced_ef"}),
    frozenset({"type_1", "type_2"}),
    frozenset({"st_elevation", "non_st_elevation", "unstable_angina"}),
    frozenset({"primary", "secondary"}),
    frozenset({"acute", "chronic"}),
)
GENERIC_DIAGNOSIS_QUALIFIERS = {
    "acute": "acute",
    "chronic": "chronic",
    "primary": "primary",
    "secondary": "secondary",
    "mild": "mild",
    "moderate": "moderate",
    "severe": "severe",
    "bacterial": "bacterial",
    "viral": "viral",
    "fungal": "fungal",
    "hemorrhagic": "hemorrhagic",
    "haemorrhagic": "hemorrhagic",
    "ischemic": "ischemic",
    "ischaemic": "ischemic",
}
GENERIC_DIAGNOSIS_QUALIFIER_PHRASES = {
    "very severe": "very_severe",
    "low risk": "low_risk",
    "intermediate risk": "intermediate_risk",
    "high risk": "high_risk",
    "preserved ejection fraction": "preserved_ef",
    "mildly reduced ejection fraction": "mildly_reduced_ef",
    "reduced ejection fraction": "reduced_ef",
    "non st elevation": "non_st_elevation",
    "st elevation": "st_elevation",
    "unstable angina": "unstable_angina",
}
GRAPH_LABEL_UMLS_QUERIES = {
    "alzheimer": "Alzheimer disease",
    "asthmacopd": "asthma COPD overlap syndrome",
    "hfmref": "heart failure",
    "hfpef": "heart failure with preserved ejection fraction",
    "hfref": "heart failure with reduced ejection fraction",
    "ltbi": "latent tuberculosis infection",
    "lowriskpe": "pulmonary embolism",
    "massivepe": "pulmonary embolism",
    "mildcopd": "chronic obstructive pulmonary disease",
    "moderatecopd": "chronic obstructive pulmonary disease",
    "nsteacs": "acute coronary syndrome",
    "pituitarymacroadenomas": "pituitary macroadenoma",
    "pituitarymicroadenomas": "pituitary microadenoma",
    "pituitarysilentadenomas": "non-functioning pituitary adenoma",
    "persistentatrialfibrillation": "persistent atrial fibrillation",
    "pulmonaryembolism": "pulmonary embolism",
    "severeasthma": "severe asthma",
    "severecopd": "chronic obstructive pulmonary disease",
    "specifictypesofdiabetes": "diabetes mellitus",
    "stemiacs": "ST elevation myocardial infarction",
    "submassivepe": "pulmonary embolism",
    "suspectedepilepsy": "epilepsy",
    "thyroidcancer": "thyroid cancer",
    "thyroidnodules": "thyroid nodule",
    "typeidiabetes": "type 1 diabetes mellitus",
    "typeiidiabetes": "type 2 diabetes mellitus",
    "ua": "unstable angina",
    "uppergastrointestinalbleeding": "upper gastrointestinal bleeding",
    "veryseverecopd": "chronic obstructive pulmonary disease",
}
DIAGNOSIS_CATEGORIES = {"diagnosis"}
RETRIEVAL_CATEGORIES = {
    "diagnosis",
    "clinical_finding",
    "lab_or_measurement",
    "therapy_or_procedure",
}


def lexical_diagnosis_key(value: str) -> str:
    key = label_key(value)
    return LEXICAL_DIAGNOSIS_ALIASES.get(key, key)


def diagnosis_family_key(value: str) -> str:
    lexical_key = lexical_diagnosis_key(value)
    for prefix in ("stronglysuspected", "suspected"):
        if lexical_key.startswith(prefix):
            lexical_key = lexical_key[len(prefix) :]
            break
    explicit = DIAGNOSIS_FAMILY_ALIASES.get(lexical_key)
    if explicit is not None:
        return explicit
    # Severity qualifies a diagnosis without replacing its underlying family.
    # Keep this separate from exact identity: "severe hypertension" remains a
    # distinct diagnosis key but belongs to the hypertension family.
    for severity in ("verysevere", "moderate", "severe", "mild"):
        if lexical_key.startswith(severity) and len(lexical_key) > len(severity):
            return DIAGNOSIS_FAMILY_ALIASES.get(
                lexical_key[len(severity):],
                lexical_key[len(severity):],
            )
    return lexical_key


def diagnosis_modifiers(value: str) -> frozenset[str]:
    modifiers = set(DIAGNOSIS_MODIFIERS.get(
        lexical_diagnosis_key(value),
        frozenset(),
    ))
    # Candidate extraction may retain surrounding words (for example
    # "known HFrEF"). Preserve a recognized qualified diagnosis token even
    # when the complete phrase is not itself a dictionary key.
    modifiers.update(
        modifier
        for token in re.split(r"[^a-z0-9-]+", value.casefold())
        for modifier in DIAGNOSIS_MODIFIERS.get(label_key(token), frozenset())
    )
    normalized = label_key(value)
    words = " ".join(
        token for token in re.split(r"[^a-z0-9]+", value.casefold())
        if token
    )
    modifiers.update(
        modifier
        for token, modifier in GENERIC_DIAGNOSIS_QUALIFIERS.items()
        if token in words.split()
    )
    modifiers.update(
        modifier
        for phrase, modifier in GENERIC_DIAGNOSIS_QUALIFIER_PHRASES.items()
        if phrase in words
    )
    if normalized.startswith("suspected"):
        modifiers.add("suspected")
    elif normalized.startswith("stronglysuspected"):
        modifiers.update({"strongly_suspected", "suspected"})
    return frozenset(modifiers)


def graph_label_umls_query(value: str) -> str:
    explicit = GRAPH_LABEL_UMLS_QUERIES.get(label_key(value))
    if explicit is not None:
        return explicit
    normalized = " ".join(value.split()).strip()
    lowered = normalized.casefold()
    for prefix in ("strongly suspected ", "suspected "):
        if lowered.startswith(prefix):
            remaining = normalized[len(prefix) :]
            return GRAPH_LABEL_UMLS_QUERIES.get(
                label_key(remaining),
                remaining,
            )
    return normalized


def modifiers_contradict(left: str, right: str) -> bool:
    left_modifiers = diagnosis_modifiers(left)
    right_modifiers = diagnosis_modifiers(right)
    return any(
        bool(left_modifiers & group)
        and bool(right_modifiers & group)
        and (left_modifiers & group) != (right_modifiers & group)
        for group in CONTRADICTORY_MODIFIER_GROUPS
    )


DIRECT_HIERARCHY_COMPATIBILITY = 0.75
DISTANT_HIERARCHY_COMPATIBILITY = 0.5


def normalized_diagnosis_key(
    value: str,
    normalizer: "UMLSNormalizer | None" = None,
) -> str:
    if normalizer is None:
        return lexical_diagnosis_key(value)
    return normalizer.diagnosis_key(value)


def diagnosis_path_compatibility(
    diagnosis: str,
    diagnostic_path: Sequence[str],
    normalizer: "UMLSNormalizer | None" = None,
) -> float:
    """Compare a diagnosis with an ordered root-to-leaf diagnostic path."""

    path = tuple(str(item) for item in diagnostic_path if str(item).strip())
    if not path:
        return 0.0
    diagnosis_key = normalized_diagnosis_key(diagnosis, normalizer)
    path_keys = tuple(
        normalized_diagnosis_key(item, normalizer)
        for item in path
    )
    matching_positions = [
        index
        for index, path_key in enumerate(path_keys)
        if path_key == diagnosis_key
    ]
    if not matching_positions:
        return 0.0
    distance_from_focal = len(path_keys) - 1 - max(matching_positions)
    if distance_from_focal == 0:
        return 1.0
    if distance_from_focal == 1:
        return DIRECT_HIERARCHY_COMPATIBILITY
    return DISTANT_HIERARCHY_COMPATIBILITY


@dataclass
class UMLSNormalizer:
    client: LocalUMLSClient
    candidate_limit: int = 12
    aliases_per_concept: int = 3

    @classmethod
    def from_path(
        cls,
        db_path: Path,
        *,
        lookup_cache_db_path: Path | None = None,
    ) -> "UMLSNormalizer":
        return cls(client=LocalUMLSClient(
            db_path=Path(db_path),
            lookup_cache_db_path=lookup_cache_db_path,
        ))

    @property
    def normalizer_id(self) -> str:
        sources = ",".join(self.client.source_vocabularies)
        alias_mode = (
            "full-aliases"
            if self.client.supports_full_alias_lookup
            else "preferred-terms"
        )
        return (
            f"umls-local-v3-{alias_mode}:"
            f"{self.client.database_id}:{sources}"
        )

    def concept(
        self,
        term: str,
        *,
        categories: set[str] | None = None,
    ) -> UMLSConcept | None:
        match = self.client.best_match(term)
        if match is None:
            return None
        if categories is not None and match.category not in categories:
            return None
        return match

    def diagnosis_key(self, term: str) -> str:
        query = graph_label_umls_query(term)
        uses_crosswalk_query = (
            label_key(term) in GRAPH_LABEL_UMLS_QUERIES
            or query != " ".join(term.split()).strip()
        )
        match = (
            self.concept(query, categories=DIAGNOSIS_CATEGORIES)
            if uses_crosswalk_query
            else self.concept(term, categories=DIAGNOSIS_CATEGORIES)
        )
        if match is not None:
            base = f"cui:{match.cui.casefold()}"
            modifiers = diagnosis_modifiers(term)
            return (
                base
                if not modifiers
                else base + "|mod:" + ",".join(sorted(modifiers))
            )
        return lexical_diagnosis_key(term)

    def diagnosis_keys(self, term: str) -> tuple[str, ...]:
        lexical_key = lexical_diagnosis_key(term)
        concept_key = self.diagnosis_key(term)
        return tuple(dict.fromkeys((lexical_key, concept_key)))

    def match_diagnosis(
        self,
        term: str,
        labels: Iterable[str],
    ) -> str | None:
        label_list = tuple(labels)
        lexical_key = lexical_diagnosis_key(term)
        for label in label_list:
            if lexical_diagnosis_key(label) == lexical_key:
                return label

        concept_key = self.diagnosis_key(term)
        if not concept_key.startswith("cui:"):
            return None
        for label in label_list:
            if self.diagnosis_key(label) == concept_key:
                return label
        return None

    def expand_text(self, text: str) -> tuple[str, ...]:
        expansions: list[str] = []
        seen_terms: set[str] = set()
        seen_cuis: set[str] = set()
        for candidate in extract_candidate_terms(text, limit=self.candidate_limit):
            match = self.concept(candidate, categories=RETRIEVAL_CATEGORIES)
            if match is None or match.cui in seen_cuis:
                continue
            seen_cuis.add(match.cui)
            terms = (
                match.preferred_term,
                *self.client.concept_terms(
                    match.cui,
                    limit=self.aliases_per_concept,
                ),
            )
            required_modifiers = diagnosis_modifiers(candidate)
            for term in terms:
                normalized = " ".join(term.split()).strip()
                key = normalized.casefold()
                # A broad UMLS preferred term is useful as a family fallback,
                # but it must not replace or augment a qualified source concept
                # while silently dropping severity, subtype, acuity, risk,
                # aetiology, or other retained diagnostic distinctions.
                if required_modifiers and not required_modifiers.issubset(
                    diagnosis_modifiers(normalized)
                ):
                    continue
                if not normalized or key in seen_terms:
                    continue
                seen_terms.add(key)
                expansions.append(normalized)
        return tuple(expansions)


@dataclass(frozen=True)
class DiagnosisCrosswalkEntry:
    graph_label: str
    lookup_term: str
    canonical_key: str
    family_key: str
    modifiers: tuple[str, ...]
    cui: str
    preferred_term: str
    semantic_type: str
    source_vocabulary: str
    match_type: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_diagnosis_crosswalk(
    graphs: Iterable[DiagnosticGraph],
    normalizer: UMLSNormalizer,
) -> tuple[DiagnosisCrosswalkEntry, ...]:
    labels = sorted(
        {
            str(label).strip()
            for graph in graphs
            for path in graph.diagnostic_paths.values()
            for label in path
            if str(label).strip()
        }
    )
    entries = []
    for graph_label in labels:
        lookup_term = graph_label_umls_query(graph_label)
        uses_reviewed_query = (
            label_key(graph_label) in GRAPH_LABEL_UMLS_QUERIES
            or lookup_term != graph_label
        )
        concept = normalizer.concept(lookup_term)
        if (
            concept is not None
            and not uses_reviewed_query
            and concept.category not in DIAGNOSIS_CATEGORIES
        ):
            concept = None
        entries.append(
            DiagnosisCrosswalkEntry(
                graph_label=graph_label,
                lookup_term=lookup_term,
                canonical_key=normalizer.diagnosis_key(graph_label),
                family_key=diagnosis_family_key(graph_label),
                modifiers=tuple(sorted(diagnosis_modifiers(graph_label))),
                cui=concept.cui if concept is not None else "",
                preferred_term=(
                    concept.preferred_term if concept is not None else ""
                ),
                semantic_type=(
                    concept.semantic_type if concept is not None else ""
                ),
                source_vocabulary=(
                    concept.source_vocabulary if concept is not None else ""
                ),
                match_type=(
                    "reviewed_query"
                    if uses_reviewed_query and concept is not None
                    else "exact_umls"
                    if concept is not None
                    else "lexical_fallback"
                ),
            )
        )
    return tuple(entries)
