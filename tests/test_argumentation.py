from clinical_cds.argumentation import (
    ArgumentScheme,
    RelationType,
    build_patient_argument_graph,
    parse_reasoner_proposal,
    parse_verifier_report,
    resolve_argument_graph,
    verifier_output_schema,
)
from clinical_cds.direct import load_direct_dataset
from clinical_cds.retrieval import KnowledgeRetriever, section_evidence


def _proposal_payload():
    return {
        "candidates": [
            {
                "diagnosis": "Hypertension",
                "arguments": [
                    {
                        "scheme": "argument_from_diagnostic_criterion",
                        "premise": "Repeated blood pressure satisfies the criterion.",
                        "evidence_ids": ["S-PE", "K1"],
                    }
                ],
            },
            {
                "diagnosis": "Suspected Hypertension",
                "arguments": [
                    {
                        "scheme": "argument_from_risk_factor",
                        "premise": "Family history increases prior plausibility.",
                        "evidence_ids": ["S-FH", "K2"],
                    }
                ],
            },
        ],
        "preferred_diagnosis": "Suspected Hypertension",
        "abstain": False,
    }


def _verifier_payload(first_verdict="supported"):
    return {
        "reviews": [
            {
                "argument_id": "A1",
                "verdict": first_verdict,
                "failed_critical_questions": (
                    [] if first_verdict == "supported" else ["criterion_satisfied"]
                ),
                "explanation": "Criterion checked.",
                "evidence_ids": ["S-PE", "K1"],
            },
            {
                "argument_id": "A2",
                "verdict": "supported",
                "failed_critical_questions": [],
                "explanation": "Risk factor is treated as prior evidence.",
                "evidence_ids": ["S-FH", "K2"],
            },
        ],
        "counterarguments": [],
        "abstain": False,
    }


def _knowledge_support(bundle):
    return {
        fact.evidence_id: (
            fact.diagnosis_label,
            *fact.diagnostic_path,
        )
        for fact in bundle.facts
    }


def _build(direct_root, first_verdict="supported"):
    dataset = load_direct_dataset(direct_root)
    case = dataset.cases[0]
    bundle = KnowledgeRetriever(dataset.graphs).retrieve(case, top_k=3)
    proposal = parse_reasoner_proposal(_proposal_payload(), case)
    verifier = parse_verifier_report(
        _verifier_payload(first_verdict),
        proposal,
    )
    valid_ids = tuple(item[0] for item in section_evidence(case)) + bundle.evidence_ids
    graph = build_patient_argument_graph(
        case_id=case.case_id,
        proposal=proposal,
        verifier=verifier,
        valid_evidence_ids=valid_ids,
        knowledge_support=_knowledge_support(bundle),
    )
    return proposal, verifier, graph


def test_argument_graph_contains_best_explanation_support_and_rebuttal(
    direct_root,
):
    proposal, verifier, graph = _build(direct_root)

    assert graph.quality.argument_schema_validity == 1.0
    assert graph.quality.argument_evidence_validity == 1.0
    assert graph.quality.verifier_review_coverage == 1.0
    assert {
        node.scheme
        for node in graph.nodes
        if node.node_type == "diagnosis"
    } == {ArgumentScheme.BEST_EXPLANATION}
    assert any(
        relation.relation == RelationType.SUPPORTS
        for relation in graph.relations
    )
    assert any(
        relation.relation == RelationType.REBUTS
        for relation in graph.relations
    )

    resolution = resolve_argument_graph(graph, proposal, verifier)
    assert resolution.selected_diagnosis == "Hypertension"
    assert resolution.abstained is False
    assert "D1" in resolution.accepted_argument_ids
    assert "D2" in resolution.rejected_argument_ids


def test_verifier_schema_is_restricted_to_proposed_identifiers(direct_root):
    dataset = load_direct_dataset(direct_root)
    proposal = parse_reasoner_proposal(
        _proposal_payload(),
        dataset.cases[0],
    )

    schema = verifier_output_schema(proposal)
    properties = schema["properties"]
    reviews = properties["reviews"]
    review_ids = reviews["items"]["properties"]["argument_id"]["enum"]
    target_ids = (
        properties["counterarguments"]["items"]["properties"]
        ["target_argument_id"]["enum"]
    )

    assert reviews["minItems"] == 2
    assert reviews["maxItems"] == 2
    assert review_ids == ["A1", "A2"]
    assert target_ids == ["D1", "D2", "A1", "A2"]


def test_risk_factor_alone_cannot_survive_failed_diagnostic_support(
    direct_root,
):
    proposal, verifier, graph = _build(
        direct_root,
        first_verdict="undercut",
    )

    resolution = resolve_argument_graph(graph, proposal, verifier)

    assert any(
        relation.source_id == "Q-A1"
        and relation.target_id == "A1"
        and relation.relation == RelationType.UNDERCUTS
        for relation in graph.relations
    )
    assert "A1" in resolution.rejected_argument_ids
    assert resolution.selected_diagnosis == ""
    assert resolution.abstained is True


def test_invalid_evidence_identifier_is_rejected(direct_root):
    payload = _proposal_payload()
    payload["candidates"][0]["arguments"][0]["evidence_ids"] = [
        "S-PE",
        "K999",
    ]
    dataset = load_direct_dataset(direct_root)
    case = dataset.cases[0]
    bundle = KnowledgeRetriever(dataset.graphs).retrieve(case, top_k=3)
    proposal = parse_reasoner_proposal(payload, case)
    verifier = parse_verifier_report(_verifier_payload(), proposal)
    graph = build_patient_argument_graph(
        case_id=case.case_id,
        proposal=proposal,
        verifier=verifier,
        valid_evidence_ids=tuple(
            item[0] for item in section_evidence(case)
        )
        + bundle.evidence_ids,
        knowledge_support=_knowledge_support(bundle),
    )
    resolution = resolve_argument_graph(graph, proposal, verifier)

    assert graph.quality.argument_evidence_validity == 0.5
    assert "A1" in resolution.rejected_argument_ids
    assert resolution.selected_diagnosis == ""
    assert resolution.abstained is True


def test_verifier_counterargument_blocks_candidate_diagnosis(direct_root):
    dataset = load_direct_dataset(direct_root)
    case = dataset.cases[0]
    bundle = KnowledgeRetriever(dataset.graphs).retrieve(case, top_k=3)
    proposal = parse_reasoner_proposal(_proposal_payload(), case)
    verifier_payload = _verifier_payload()
    verifier_payload["counterarguments"] = [
        {
            "target_argument_id": "D1",
            "scheme": "argument_from_negative_evidence",
            "premise": "The submitted history contains evidence against the conclusion.",
            "conclusion": "Hypertension is not established.",
            "evidence_ids": ["S-PMH", "K1"],
            "relation": "undercuts",
        }
    ]
    verifier = parse_verifier_report(verifier_payload, proposal)
    graph = build_patient_argument_graph(
        case_id=case.case_id,
        proposal=proposal,
        verifier=verifier,
        valid_evidence_ids=tuple(
            item[0] for item in section_evidence(case)
        )
        + bundle.evidence_ids,
        knowledge_support=_knowledge_support(bundle),
    )

    resolution = resolve_argument_graph(graph, proposal, verifier)

    assert "C1" in resolution.accepted_argument_ids
    assert "D1" in resolution.rejected_argument_ids
    assert resolution.abstained is True


def test_duplicate_evidence_does_not_inflate_scheme_score(direct_root):
    payload = _proposal_payload()
    payload["candidates"][0]["arguments"].append(
        {
            "scheme": "argument_from_diagnostic_criterion",
            "premise": "The same repeated blood pressure satisfies the criterion.",
            "evidence_ids": ["S-PE", "K1"],
        }
    )
    verifier_payload = _verifier_payload()
    verifier_payload["reviews"].insert(
        1,
        {
            "argument_id": "A2",
            "verdict": "supported",
            "failed_critical_questions": [],
            "explanation": "Duplicate criterion support.",
            "evidence_ids": ["S-PE", "K1"],
        },
    )
    verifier_payload["reviews"][2]["argument_id"] = "A3"

    dataset = load_direct_dataset(direct_root)
    case = dataset.cases[0]
    bundle = KnowledgeRetriever(dataset.graphs).retrieve(case, top_k=3)
    proposal = parse_reasoner_proposal(payload, case)
    verifier = parse_verifier_report(verifier_payload, proposal)
    graph = build_patient_argument_graph(
        case_id=case.case_id,
        proposal=proposal,
        verifier=verifier,
        valid_evidence_ids=tuple(
            item[0] for item in section_evidence(case)
        )
        + bundle.evidence_ids,
        knowledge_support=_knowledge_support(bundle),
    )

    resolution = resolve_argument_graph(graph, proposal, verifier)

    assert resolution.candidate_scores["D1"] == 4


def test_valid_but_diagnostically_unrelated_knowledge_is_rejected(direct_root):
    payload = {
        "candidates": [
            {
                "diagnosis": "Asthma",
                "arguments": [
                    {
                        "scheme": "argument_from_diagnostic_criterion",
                        "premise": "The finding is presented as an asthma criterion.",
                        "evidence_ids": ["S-PE", "K1"],
                    }
                ],
            }
        ],
        "preferred_diagnosis": "Asthma",
        "abstain": False,
    }
    verifier_payload = {
        "reviews": [
            {
                "argument_id": "A1",
                "verdict": "supported",
                "failed_critical_questions": [],
                "explanation": "The supplied identifiers exist.",
                "evidence_ids": ["S-PE", "K1"],
            }
        ],
        "counterarguments": [],
        "abstain": False,
    }
    dataset = load_direct_dataset(direct_root)
    case = dataset.cases[0]
    bundle = KnowledgeRetriever(dataset.graphs).retrieve(case, top_k=3)
    proposal = parse_reasoner_proposal(payload, case)
    verifier = parse_verifier_report(verifier_payload, proposal)
    graph = build_patient_argument_graph(
        case_id=case.case_id,
        proposal=proposal,
        verifier=verifier,
        valid_evidence_ids=tuple(
            item[0] for item in section_evidence(case)
        )
        + bundle.evidence_ids,
        knowledge_support=_knowledge_support(bundle),
    )

    resolution = resolve_argument_graph(graph, proposal, verifier)

    assert graph.quality.valid_evidence_reference_fraction == 1.0
    assert graph.quality.argument_evidence_validity == 0.0
    assert "A1" in resolution.rejected_argument_ids
    assert resolution.abstained is True
