"""Contract for the supported evidence-grounded argumentation method."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from clinical_cds.version import CURRENT_RELEASE


@dataclass(frozen=True)
class ArgumentMethodContract:
    method_id: str
    prompt_version: str
    architecture: str
    candidate_inventory_id: str
    resolver_id: str
    uses_flat_rag_seed: bool
    flat_rag_role: str
    retrieval_bundle_role: str
    critic_relation_types: tuple[str, ...]
    resolver_decisions: tuple[str, ...]
    resolution_rule_id: str
    shared_artifact_control: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


ARGUMENT_METHOD = ArgumentMethodContract(
    method_id="direct-differential-rag-argumentation",
    prompt_version=CURRENT_RELEASE.dialogue_id,
    architecture=(
        "family-first-retrieval-eight-family-inventory-four-family-activation-"
        "family-then-subtype-cited-differential-adversarial-verification-"
        "independent-attack-validation-protected-incumbent-deterministic-resolution"
    ),
    candidate_inventory_id="kg-ranked-fixed-top8",
    resolver_id="direct-differential-family-first-resolver-v1",
    uses_flat_rag_seed=False,
    flat_rag_role="independent_comparator_only",
    retrieval_bundle_role="byte_identical_case_specific_kg_bundle",
    critic_relation_types=(
        "citation_failure",
        "wrong_subject_or_encounter",
        "diagnostic_insufficiency",
        "explicit_contradiction",
        "better_supported_alternative",
        "unsupported_specificity",
        "material_unexplained_evidence",
    ),
    resolver_decisions=("maintain", "switch", "family_fallback",
                        "protected_incumbent", "abstain", "error"),
    resolution_rule_id="direct-differential-family-first-resolution-v1",
    shared_artifact_control=(
        "immutable-family-context-bounded-citations-grounded-attack-"
        "cited-incumbent-deterministic-resolution"
    ),
)

CURRENT_ARGUMENT_METHOD = ARGUMENT_METHOD
DEFAULT_ARGUMENT_METHOD = ARGUMENT_METHOD


def argument_method_contract(
    value: str | ArgumentMethodContract,
) -> ArgumentMethodContract:
    if value == ARGUMENT_METHOD or value == ARGUMENT_METHOD.method_id:
        return ARGUMENT_METHOD
    raise ValueError(f"Unsupported argumentation method: {value!r}")
