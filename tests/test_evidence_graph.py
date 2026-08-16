from clinical_cds.evidence_graph import (
    DECISIVE_ANCHOR_COMPILER_ID,
    EVIDENCE_GRAPH_COMPILER_ID,
    EvidenceRelation,
    PropositionKind,
    audit_evidence_graph,
    audit_decisive_anchors,
    atomize_retrieved_family_facts,
    compile_evidence_graph,
    compile_evidence_graphs,
    compile_decisive_anchors,
    compile_decisive_anchor_graphs,
    qualifies_decisive_anchor,
)
from clinical_cds.graph_extensions import load_project_graph_extensions
from clinical_cds.schema import FamilyChildFact, RetrievedFact


def test_compiler_atomizes_sourced_graph_without_losing_provenance():
    graph = load_project_graph_extensions()[0].graph

    propositions = compile_evidence_graph(graph)
    audit = audit_evidence_graph(graph, propositions)

    assert audit.compiler_id == EVIDENCE_GRAPH_COMPILER_ID
    assert audit.proposition_count >= audit.source_premise_count
    assert audit.provenance_complete is True
    assert audit.atomic_text_complete is True
    assert all(item.parent_node_id and item.parent_text for item in propositions)


def test_compiler_types_tests_measurements_risk_and_conjunctions():
    graph = load_project_graph_extensions()[0].graph
    propositions = compile_evidence_graph(graph)

    assert any(item.kind == PropositionKind.DIAGNOSTIC_TEST for item in propositions)
    assert any(item.relation == EvidenceRelation.WEAK_SUPPORT for item in propositions)
    assert any(item.conjunctive for item in propositions)


def test_compiler_is_case_independent_and_contains_no_patient_or_gold_fields():
    graph = load_project_graph_extensions()[0].graph
    payload = [item.to_dict() for item in compile_evidence_graph(graph)]

    assert payload
    assert all("patient" not in item and "gold" not in item for item in payload)


def test_same_compiler_applies_to_every_graph_without_category_rules():
    graph = load_project_graph_extensions()[0].graph

    compiled = compile_evidence_graphs((graph,))

    assert set(compiled) == {"Gastritis"}
    assert compiled["Gastritis"] == compile_evidence_graph(graph)


def test_decisive_anchor_compiler_derives_only_sourced_test_or_threshold_claims():
    graph = load_project_graph_extensions()[0].graph

    anchors = compile_decisive_anchors(graph)
    audit = audit_decisive_anchors(graph, anchors)

    assert audit.compiler_id == DECISIVE_ANCHOR_COMPILER_ID
    assert audit.anchor_count == len(anchors)
    assert audit.provenance_complete is True
    assert anchors
    assert all(item.modalities or item.threshold_bearing for item in anchors)
    assert all(item.knowledge_source_ids and item.parent_node_id for item in anchors)
    assert all("Risk Factors" != item.premise_type for item in anchors)


def test_decisive_anchor_compiler_is_category_independent():
    graph = load_project_graph_extensions()[0].graph

    compiled = compile_decisive_anchor_graphs((graph,))

    assert compiled == {"Gastritis": compile_decisive_anchors(graph)}


def test_decisive_anchor_qualification_excludes_negative_and_noncriteria_claims():
    from clinical_cds.argumentation import KnowledgeRole

    assert qualifies_decisive_anchor(
        "CTPA directly displays thrombus in the pulmonary artery.",
        KnowledgeRole.DIAGNOSTIC_CRITERION,
    )
    assert not qualifies_decisive_anchor(
        "CTPA is negative for thrombus.", KnowledgeRole.DIAGNOSTIC_CRITERION
    )
    assert not qualifies_decisive_anchor(
        "CTPA is useful in selected patients.", KnowledgeRole.GUIDELINE
    )


def test_comparative_definition_negation_is_not_an_exclusion():
    from clinical_cds.argumentation import KnowledgeRole
    from clinical_cds.evidence_graph import EvidenceRelation, _relation

    # From clinical_cds/knowledge/Gastritis.provenance.json: "without" here
    # names what a *different*, contrasting diagnosis is defined by -- the
    # sourced claim is a positive defining criterion for its own diagnosis,
    # not an exclusion of it.
    text = (
        "Gastric histology shows chronic inflammation without the gland "
        "loss or intestinal metaplasia that defines atrophic gastritis"
    )
    # This is a genuine, sourced histology-based defining criterion -- it
    # must qualify as a decisive anchor, not be excluded by the embedded
    # "without" clause naming a contrasting diagnosis's own definition.
    assert qualifies_decisive_anchor(text, KnowledgeRole.DIAGNOSTIC_CRITERION) is True
    assert _relation(text, KnowledgeRole.DIAGNOSTIC_CRITERION) != (
        EvidenceRelation.CONTRADICTION_OR_EXCLUSION
    )


def test_with_or_without_is_not_an_exclusion():
    from clinical_cds.argumentation import KnowledgeRole
    from clinical_cds.evidence_graph import EvidenceRelation, _relation

    text = (
        "Histopathology confirms loss of gastric glands, with or without "
        "intestinal metaplasia, in chronic inflammation"
    )
    assert _relation(text, KnowledgeRole.DIAGNOSTIC_CRITERION) != (
        EvidenceRelation.CONTRADICTION_OR_EXCLUSION
    )


def test_genuine_negative_finding_still_excludes():
    from clinical_cds.argumentation import KnowledgeRole
    from clinical_cds.evidence_graph import EvidenceRelation, _relation

    assert _relation(
        "CTPA is negative for thrombus.", KnowledgeRole.DIAGNOSTIC_CRITERION
    ) == EvidenceRelation.CONTRADICTION_OR_EXCLUSION


def test_retrieved_fact_atomization_preserves_parent_and_source_identity():
    original = FamilyChildFact("graph:test", RetrievedFact(
        evidence_id="KF1",
        node_id="node-1",
        category="Example",
        diagnosis_label="Example Child",
        premise_type="Diagnostic Criteria",
        text="Imaging shows opacity; Culture confirms an organism.",
        score=0.0,
        diagnostic_path=("Example Family", "Example Child"),
        knowledge_source_ids=("source-1",),
        source_chunk_id="chunk-1",
    ))

    atoms = atomize_retrieved_family_facts((original,))

    assert len(atoms) == 2
    assert len({item.fact.evidence_id for item in atoms}) == 2
    assert {item.fact.node_id for item in atoms} == {"node-1"}
    assert {item.fact.source_chunk_id for item in atoms} == {"chunk-1"}
    assert all(item.fact.knowledge_source_ids == ("source-1",) for item in atoms)
