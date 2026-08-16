import pytest

from clinical_cds.argumentation_v3 import (
    AttackScope, AttackType, AttackValidation, DirectDecision,
    DifferentialAttack, ProtectedIncumbent, ResolutionAction,
    attack_validation_schema, differential_attack_schema, direct_differential_schema,
    evidence_aware_activation_schema, parse_evidence_aware_activation,
    parse_attack_validation, parse_differential_attack, parse_direct_differential,
    resolve_direct_differential,
)


def proposal_payload(**overrides):
    payload = {"candidate_id": "D1", "child_label": "",
        "decision": "diagnosis_supported", "decisive_pair_ids": ["P1"],
        "strongest_alternative_id": "D2", "alternative_pair_ids": ["P2"],
        "unexplained_patient_evidence_ids": ["S3"], "rationale": "D1 explains the encounter."}
    payload.update(overrides)
    return payload


def parse_proposal(payload=None):
    return parse_direct_differential(payload or proposal_payload(), ("D1", "D2"),
        {"P1": "D1", "P2": "D2"}, ("S1", "S2", "S3"), {"D1": ("Child 1",), "D2": ()})


def test_direct_differential_keeps_only_candidate_owned_citations():
    parsed = parse_proposal(proposal_payload(decisive_pair_ids=["P2", "P1", "P1"]))
    assert parsed.decisive_pair_ids == ("P1",)


def test_evidence_free_supported_answer_fails_closed():
    parsed = parse_proposal(proposal_payload(decisive_pair_ids=["P2"]))
    assert parsed.decision == DirectDecision.INSUFFICIENT
    assert parsed.candidate_id == ""


def test_risk_factor_only_citation_cannot_support_a_diagnosis():
    # P1 is owned by D1 but is not in the establishing set (e.g. it is a
    # risk-factor pairing, or one that fails the typed-binding critical
    # questions). A decision cited only by such pairs must fail closed
    # instead of being accepted as SUPPORTED.
    parsed = parse_direct_differential(
        proposal_payload(decisive_pair_ids=["P1"]),
        ("D1", "D2"), {"P1": "D1", "P2": "D2"}, ("S1", "S2", "S3"),
        {"D1": ("Child 1",), "D2": ()},
        establishing_pair_ids=frozenset(),
    )
    assert parsed.decision == DirectDecision.INSUFFICIENT
    assert parsed.candidate_id == ""
    assert parsed.decisive_pair_ids == ()


def test_mixed_citation_with_one_establishing_pair_still_supports():
    parsed = parse_direct_differential(
        proposal_payload(decisive_pair_ids=["P1", "P1b"]),
        ("D1", "D2"), {"P1": "D1", "P1b": "D1", "P2": "D2"},
        ("S1", "S2", "S3"), {"D1": ("Child 1",), "D2": ()},
        establishing_pair_ids=frozenset({"P1b"}),
    )
    assert parsed.decision == DirectDecision.SUPPORTED
    assert parsed.candidate_id == "D1"
    assert set(parsed.decisive_pair_ids) == {"P1", "P1b"}


def test_better_alternative_attack_requires_establishing_citation():
    # The alternative is cited (P2 owned by D2) but P2 is not in the
    # establishing set, so the alternative is not independently
    # established and the attack must not gain authority.
    parsed = parse_differential_attack(
        {"attack": True, "attack_type": "better_supported_alternative",
         "target_candidate_id": "D1", "alternative_candidate_id": "D2",
         "evidence_pair_ids": ["P1", "P2"],
         "explanation": "D2 has evidence but it does not establish it."},
        ("D1", "D2"), {"P1": "D1", "P2": "D2"}, "D1",
        establishing_pair_ids=frozenset(),
    )
    assert parsed.attack is False
    assert parsed.attack_type == AttackType.NONE


def test_none_of_candidates_with_grounded_allowed_alternative_recovers_parent():
    parsed = parse_proposal(proposal_payload(
        candidate_id="D1",
        child_label="Child 1",
        decision="none_of_supplied_candidates",
        decisive_pair_ids=["P2"],
        strongest_alternative_id="D1",
        alternative_pair_ids=["P1"],
    ))

    assert parsed.decision == DirectDecision.SUPPORTED
    assert parsed.candidate_id == "D1"
    assert parsed.decisive_pair_ids == ("P1",)
    assert parsed.child_label == ""


def test_unsustained_attack_has_no_force():
    parsed = parse_differential_attack({"attack": False,
        "attack_type": "better_supported_alternative", "target_candidate_id": "D1",
        "alternative_candidate_id": "D2", "evidence_pair_ids": ["P2"],
        "explanation": "Considered and rejected."}, ("D1", "D2"), {"P1": "D1", "P2": "D2"}, "D1")
    assert not parsed.attack and parsed.attack_type == AttackType.NONE


def test_non_alternative_attack_requires_target_owned_evidence():
    parsed = parse_differential_attack({"attack": True,
        "attack_type": "explicit_contradiction", "target_candidate_id": "D1",
        "alternative_candidate_id": "D2", "evidence_pair_ids": ["P2"],
        "explanation": "Evidence for D2 is not a contradiction of D1."},
        ("D1", "D2"), {"P1": "D1", "P2": "D2"}, "D1")
    assert parsed.attack is False
    assert parsed.attack_type == AttackType.NONE


def test_grounded_better_alternative_switches_deterministically():
    proposal = parse_proposal()
    attack = parse_differential_attack({"attack": True,
        "attack_type": "better_supported_alternative", "target_candidate_id": "D1",
        "alternative_candidate_id": "D2", "evidence_pair_ids": ["P1", "P2"],
        "explanation": "D2 explains the decisive finding better."},
        ("D1", "D2"), {"P1": "D1", "P2": "D2"}, "D1")
    validation = AttackValidation(True, True, True, True, AttackScope.FAMILY, "Valid.")
    result = resolve_direct_differential(proposal, attack, {"D1": "Family 1", "D2": "Family 2"}, validation)
    assert result.action == ResolutionAction.SWITCH
    assert result.selected_diagnosis == "Family 2"


def test_better_alternative_without_target_falsifier_has_no_attack_authority():
    parsed = parse_differential_attack({"attack": True,
        "attack_type": "better_supported_alternative", "target_candidate_id": "D1",
        "alternative_candidate_id": "D2", "evidence_pair_ids": ["P2"],
        "explanation": "D2 has evidence but does not refute D1."},
        ("D1", "D2"), {"P1": "D1", "P2": "D2"}, "D1")
    assert parsed.attack is False


def test_supported_competitor_without_refutation_yields_differential_abstention():
    proposal = parse_proposal()
    attack = DifferentialAttack(True, AttackType.BETTER_SUPPORTED_ALTERNATIVE,
        "D1", "D2", ("P1", "P2"), "D2 is independently supported.")
    validation = AttackValidation(
        valid=False,
        citations_support_stated_attack=True,
        targets_proposed_diagnosis=False,
        not_merely_coexisting_or_alternative=False,
        scope=AttackScope.NONE,
        explanation="D2 is supported but compatible with D1.",
        directly_falsifies_target=False,
        falsified_condition_is_necessary=False,
        independently_establishes_alternative=True,
    )
    result = resolve_direct_differential(
        proposal, attack, {"D1": "Family 1", "D2": "Family 2"}, validation
    )
    assert result.action == ResolutionAction.ABSTAIN
    assert result.leading_diagnoses == ("Family 1", "Family 2")


def test_unsupported_specificity_falls_back_to_family():
    proposal = parse_proposal(proposal_payload(child_label="Child 1"))
    attack = DifferentialAttack(True, AttackType.UNSUPPORTED_SPECIFICITY, "D1", "", ("P1",), "Child is not unique.")
    validation = AttackValidation(True, True, True, True, AttackScope.CHILD_ONLY, "Child only.")
    result = resolve_direct_differential(proposal, attack, {"D1": "Family 1", "D2": "Family 2"}, validation)
    assert result.action == ResolutionAction.FAMILY_FALLBACK
    assert result.selected_diagnosis == "Family 1"


def test_unvalidated_coexisting_possibility_cannot_defeat_supported_diagnosis():
    proposal = parse_proposal()
    attack = DifferentialAttack(True, AttackType.EXPLICIT_CONTRADICTION, "D1", "",
                                ("P1", "P2"), "D2 may coexist.")
    invalid = AttackValidation(False, False, False, False, AttackScope.NONE,
                               "Coexistence is not contradiction.")
    result = resolve_direct_differential(
        proposal, attack, {"D1": "Family 1", "D2": "Family 2"}, invalid
    )
    assert result.action == ResolutionAction.FAMILY_FALLBACK
    assert result.selected_diagnosis == "Family 1"


def test_insufficient_differential_retains_two_evidence_cited_possibilities():
    proposal = parse_proposal(proposal_payload(decision="insufficient"))
    result = resolve_direct_differential(
        proposal,
        DifferentialAttack(False, AttackType.NONE, "", "", (), "No attack."),
        {"D1": "Family 1", "D2": "Family 2"},
    )
    assert result.action == ResolutionAction.ABSTAIN
    assert result.leading_diagnoses == ("Family 1", "Family 2")


def test_insufficient_differential_uses_verified_cited_incumbent_at_family_level():
    proposal = parse_proposal(proposal_payload(decision="insufficient"))
    incumbent = ProtectedIncumbent("D1", "Family 1", ("P1",))

    result = resolve_direct_differential(
        proposal,
        DifferentialAttack(False, AttackType.NONE, "", "", (), "No attack."),
        {"D1": "Family 1", "D2": "Family 2"},
        protected_incumbent=incumbent,
    )

    assert result.action == ResolutionAction.PROTECTED_INCUMBENT
    assert result.selected_diagnosis == "Family 1"
    assert result.selected_pair_ids == ("P1",)


def test_valid_family_attack_blocks_protected_incumbent():
    proposal = parse_proposal(proposal_payload(decision="insufficient"))
    incumbent = ProtectedIncumbent("D1", "Family 1", ("P1",))
    attack = DifferentialAttack(
        True, AttackType.EXPLICIT_CONTRADICTION, "D1", "", ("P1",),
        "A necessary family proposition is directly contradicted.",
    )
    validation = AttackValidation(
        True, True, True, True, AttackScope.FAMILY, "Valid family attack.",
        directly_falsifies_target=True,
        falsified_condition_is_necessary=True,
    )

    result = resolve_direct_differential(
        proposal, attack, {"D1": "Family 1", "D2": "Family 2"}, validation,
        incumbent,
    )

    assert result.action == ResolutionAction.ABSTAIN


def test_supported_direct_answer_takes_precedence_over_incumbent():
    incumbent = ProtectedIncumbent("D2", "Family 2", ("P2",))

    result = resolve_direct_differential(
        parse_proposal(),
        DifferentialAttack(False, AttackType.NONE, "", "", (), "No attack."),
        {"D1": "Family 1", "D2": "Family 2"},
        protected_incumbent=incumbent,
    )

    assert result.action == ResolutionAction.FAMILY_FALLBACK
    assert result.selected_candidate_id == "D1"


def test_vllm_compatible_arrays_remain_strictly_output_bounded():
    direct = direct_differential_schema(
        ("D1", "D2"),
        tuple(f"P{index}" for index in range(20)),
        tuple(f"S{index}" for index in range(20)),
        (),
    )
    attack = differential_attack_schema(
        ("D1", "D2"), tuple(f"P{index}" for index in range(20))
    )

    properties = direct["properties"]
    assert properties["decisive_pair_ids"]["maxItems"] == 4
    assert properties["alternative_pair_ids"]["maxItems"] == 4
    assert properties["unexplained_patient_evidence_ids"]["maxItems"] == 8
    assert attack["properties"]["evidence_pair_ids"]["maxItems"] == 8


def test_direct_differential_schema_decodes_reasoning_before_the_verdict():
    # Guided decoding fills object keys in schema declaration order. The
    # verdict fields (decision, and everything that follows from it) must
    # come after the fields that ground it (rationale, candidate_id,
    # decisive_pair_ids) -- otherwise the model has to commit to a decision
    # before it has generated any citation or reasoning to base it on.
    schema = direct_differential_schema(("D1", "D2"), ("P1",), ("S1",), ())
    order = list(schema["properties"])

    assert order.index("rationale") < order.index("decision")
    assert order.index("candidate_id") < order.index("decision")
    assert order.index("decisive_pair_ids") < order.index("decision")


def test_differential_attack_schema_decodes_reasoning_before_the_verdict():
    schema = differential_attack_schema(("D1", "D2"), ("P1",))
    order = list(schema["properties"])

    assert order.index("explanation") < order.index("attack")
    assert order.index("attack_type") < order.index("attack")
    assert order.index("evidence_pair_ids") < order.index("attack")


def test_attack_validation_schema_decodes_reasoning_before_the_critical_questions():
    schema = attack_validation_schema()
    order = list(schema["properties"])

    for question in (
        "citations_support_stated_attack", "targets_proposed_diagnosis",
        "not_merely_coexisting_or_alternative", "directly_falsifies_target",
        "falsified_condition_is_necessary", "independently_establishes_alternative",
    ):
        assert order.index("explanation") < order.index(question)


def test_direct_parser_rejects_schema_invalid_array_type_before_resolution():
    payload = proposal_payload(decisive_pair_ids="P1")
    with pytest.raises(ValueError, match="array of strings"):
        parse_proposal(payload)


def test_attack_parser_rejects_extra_fields_before_resolution():
    payload = {
        "attack": False,
        "attack_type": "none",
        "target_candidate_id": "",
        "alternative_candidate_id": "",
        "evidence_pair_ids": [],
        "explanation": "No grounded defect.",
        "unexpected": "not allowed",
    }
    with pytest.raises(ValueError, match="frozen schema"):
        parse_differential_attack(
            payload, ("D1", "D2"), {"P1": "D1", "P2": "D2"}, "D1"
        )


def test_evidence_aware_activation_can_promote_lower_retrieval_rank():
    owners = {f"P{index}": f"D{index}" for index in range(1, 9)}
    payload = {
        "selections": [
            {"candidate_id": "D5", "evidence_pair_ids": ["P5"]},
            {"candidate_id": "D2", "evidence_pair_ids": ["P2"]},
            {"candidate_id": "D7", "evidence_pair_ids": ["P7"]},
            {"candidate_id": "D1", "evidence_pair_ids": ["P1"]},
        ],
        "rationale": "Current diagnostic evidence outranks retrieval position.",
    }

    activation = parse_evidence_aware_activation(
        payload, tuple(owners.values()), owners
    )

    assert activation.candidate_ids == ("D5", "D2", "D7", "D1")


def test_evidence_aware_activation_rejects_cross_family_evidence():
    owners = {"P1": "D1", "P2": "D2", "P3": "D3", "P4": "D4"}
    payload = {
        "selections": [
            {"candidate_id": "D1", "evidence_pair_ids": ["P2"]},
            {"candidate_id": "D2", "evidence_pair_ids": ["P2"]},
            {"candidate_id": "D3", "evidence_pair_ids": ["P3"]},
            {"candidate_id": "D4", "evidence_pair_ids": ["P4"]},
        ],
        "rationale": "Invalid ownership.",
    }
    with pytest.raises(ValueError, match="candidate-owned evidence"):
        parse_evidence_aware_activation(payload, tuple(owners.values()), owners)


def test_activation_schema_remains_bounded_after_unique_items_removal():
    owners = {f"P{index}": f"D{index}" for index in range(1, 9)}
    schema = evidence_aware_activation_schema(tuple(owners.values()), owners)
    selections = schema["properties"]["selections"]
    citations = selections["items"]["properties"]["evidence_pair_ids"]
    assert selections["minItems"] == 0
    assert selections["maxItems"] == 4
    assert citations["minItems"] == 1
    assert citations["maxItems"] == 2


def test_non_strict_activation_rebuilds_sparse_payload():
    owners = {"P1": "D1", "P2": "D2"}
    payload = {"selections": [{"candidate_id": "D1"}], "rationale": 123}
    activation = parse_evidence_aware_activation(
        payload, tuple(owners.values()), owners, limit=2, strict=False
    )
    assert activation.candidate_ids == ("D1", "D2")
