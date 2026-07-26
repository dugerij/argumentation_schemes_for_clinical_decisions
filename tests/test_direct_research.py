import json

from clinical_cds.direct import load_direct_dataset
from clinical_cds.medqa import load_medqa_cases
from clinical_cds.retrieval import (
    KnowledgeRetriever,
    render_flat_retrieval,
    render_graph_retrieval,
    section_evidence,
)


def test_direct_loader_compiles_annotations_and_guideline_graph(direct_root):
    dataset = load_direct_dataset(direct_root)

    assert dataset.audit.case_count == 1
    assert dataset.audit.graph_count == 1
    assert dataset.audit.missing_graph_categories == ()
    assert dataset.cases[0].gold_label == "Hypertension"
    assert [node.text for node in dataset.cases[0].gold_observations] == [
        "Blood pressure is 170/100 mmHg.",
        "Headache",
    ]
    assert len(dataset.graphs[0].leaf_labels) == 1


def test_flat_and_graph_conditions_use_the_same_retrieved_facts(direct_root):
    dataset = load_direct_dataset(direct_root)
    case = dataset.cases[0]
    bundle = KnowledgeRetriever(dataset.graphs).retrieve(case, top_k=3)

    assert bundle.facts
    assert [fact.evidence_id for fact in bundle.facts] == ["K1", "K2", "K3"]
    assert len(render_flat_retrieval(bundle).splitlines()) == 3
    assert len(render_graph_retrieval(bundle).splitlines()) == 6

def test_section_evidence_ids_are_stable_when_a_section_is_absent(direct_root):
    case = load_direct_dataset(direct_root).cases[0]
    full_ids = dict(
        (section_name, evidence_id)
        for evidence_id, section_name, _ in section_evidence(case)
    )
    sections = dict(case.sections)
    sections.pop("chief_complaint")
    reduced_case = type(case)(
        **{
            **case.__dict__,
            "sections": sections,
        }
    )
    reduced_ids = dict(
        (section_name, evidence_id)
        for evidence_id, section_name, _ in section_evidence(reduced_case)
    )

    assert full_ids["physical_exam"] == "S-PE"
    assert reduced_ids["physical_exam"] == "S-PE"


def test_medqa_loader_keeps_diagnostic_questions_only(tmp_path, direct_root):
    medqa_dir = tmp_path / "medqa"
    medqa_dir.mkdir()
    rows = [
        {
            "question": "Which of the following is the most likely diagnosis?",
            "answer": "Hypertension",
            "options": {"A": "Asthma", "B": "Hypertension"},
            "answer_idx": "B",
            "meta_info": "step2",
        },
        {
            "question": "Which imaging study best evaluates this finding?",
            "answer": "Example scan",
            "options": {"A": "Example scan", "B": "Other scan"},
            "answer_idx": "A",
            "meta_info": "step1",
        },
    ]
    (medqa_dir / "test.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    graphs = load_direct_dataset(direct_root).graphs

    cases = load_medqa_cases(medqa_dir, split="test", graphs=graphs)

    assert len(cases) == 1
    assert cases[0].gold_label == "Hypertension"
    assert cases[0].metadata["direct_graph_covered"] is True
