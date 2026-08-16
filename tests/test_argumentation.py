from dataclasses import replace

from clinical_cds.argumentation import (
    KnowledgeRole,
    family_candidate_inventory_entries,
    family_entailment_shortlist,
    knowledge_claims,
    knowledge_role,
)
from clinical_cds.direct import load_direct_dataset
from clinical_cds.schema import (
    FamilyChildFact,
    FamilyDiagnosisAlternative,
    RetrievedFact,
    RetrievedFamilyRoute,
)


def test_knowledge_roles_preserve_diagnostic_authority():
    assert knowledge_role("Diagnostic Criteria") == KnowledgeRole.DIAGNOSTIC_CRITERION
    assert knowledge_role("Clinical Features") == KnowledgeRole.CLINICAL_FEATURE
    assert knowledge_role("Risk Factors") == KnowledgeRole.RISK_FACTOR
    assert knowledge_role("Guideline") == KnowledgeRole.GUIDELINE


def test_knowledge_claims_retain_graph_provenance():
    fact = RetrievedFact(
        evidence_id="K1",
        node_id="node-1",
        category="Pneumonia",
        diagnosis_label="Bacterial Pneumonia",
        premise_type="Diagnostic Criteria",
        text="A pulmonary infiltrate with fever supports pneumonia.",
        score=1.0,
        diagnostic_path=("Pneumonia", "Bacterial Pneumonia"),
        source_chunk_id="source-chunk:" + "1" * 20,
    )

    claim = knowledge_claims((fact,))[0]

    assert claim.evidence_id == "K1"
    assert claim.node_id == "node-1"
    assert claim.role == KnowledgeRole.DIAGNOSTIC_CRITERION
    assert claim.diagnostic_path == ("Pneumonia", "Bacterial Pneumonia")


def test_family_inventory_uses_one_slot_and_keeps_child_warrants(direct_root):
    case = replace(load_direct_dataset(direct_root).cases[0], options={})
    facts = (
        RetrievedFact(
            "K1", "node-nstemi", "ACS", "NSTEMI", "Diagnostic Criteria",
            "Troponin rise without ST elevation supports NSTEMI.", 1.0,
            ("Suspected ACS", "NSTE-ACS", "NSTEMI"),
            source_chunk_id="source-chunk:" + "1" * 20,
        ),
        RetrievedFact(
            "K2", "node-ua", "ACS", "UA", "Diagnostic Criteria",
            "Ischemia without biomarker rise supports UA.", 0.8,
            ("Suspected ACS", "NSTE-ACS", "UA"),
            source_chunk_id="source-chunk:" + "2" * 20,
        ),
    )
    route = RetrievedFamilyRoute(
        family_rank=1,
        graph_id="direct:acs",
        family_key="graph:direct:acs",
        representative_diagnosis="NSTEMI",
        alternatives=tuple(
            FamilyDiagnosisAlternative(
                candidate_id=f"candidate:{label.casefold()}",
                diagnosis_label=label,
                graph_id="direct:acs",
                diagnostic_path=fact.diagnostic_path,
                source_chunk_ids=(fact.source_chunk_id,),
                original_candidate_rank=index,
                representative=index == 1,
            )
            for index, (label, fact) in enumerate(
                (("NSTEMI", facts[0]), ("UA", facts[1])), 1
            )
        ),
    )

    inventory = family_candidate_inventory_entries(
        case, facts, family_routes=(route,)
    )

    assert len(inventory) == 1
    assert inventory[0].diagnosis == "ACS"
    assert inventory[0].evidence_ids == ("K1", "K2")
    assert inventory[0].graph_id == "direct:acs"


def test_entailment_shortlist_is_family_local_and_safety_filtered():
    facts = (
        FamilyChildFact("graph:pneumonia", RetrievedFact(
            "KF1", "n1", "Pneumonia", "Bacterial Pneumonia",
            "Diagnostic Criteria", "Fever with a pulmonary infiltrate.",
            0.0, ("Pneumonia", "Bacterial Pneumonia"),
        )),
        FamilyChildFact("graph:pneumonia", RetrievedFact(
            "KF2", "n2", "Pneumonia", "Bacterial Pneumonia",
            "Diagnostic Criteria", "Family history increases probability.",
            0.0, ("Pneumonia", "Bacterial Pneumonia"),
        )),
        FamilyChildFact("graph:pneumonia", RetrievedFact(
            "KF3", "n3", "Pneumonia", "Viral Pneumonia",
            "Diagnostic Criteria", "No focal pulmonary infiltrate.",
            0.0, ("Pneumonia", "Viral Pneumonia"),
        )),
        FamilyChildFact("graph:pe", RetrievedFact(
            "KF4", "n4", "Pulmonary Embolism", "Low-risk PE",
            "Diagnostic Criteria", "CT pulmonary angiography shows embolus.",
            0.0, ("Pulmonary Embolism", "Low-risk PE"),
        )),
    )

    selected = family_entailment_shortlist(
        (("S-HPI", "Current fever with a new pulmonary infiltrate."),),
        facts,
        maximum_per_family=2,
    )

    assert [item.fact.evidence_id for item in selected] == ["KF1", "KF4"]
