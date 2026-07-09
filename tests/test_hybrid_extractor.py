from llama_index.core import Document
from llama_index.core.schema import MetadataMode

from retrieval.concepts.hybrid_extractor import HybridClinicalPathExtractor
from retrieval.concepts.schema import UMLSConcept
from retrieval.index import build_document


class _FakeUMLSClient:
    def __init__(self) -> None:
        self.mapping = {
            "acute kidney injury": UMLSConcept(
                cui="C0001",
                preferred_term="Acute kidney injury",
                semantic_type="Disease or Syndrome",
                source_vocabulary="SNOMEDCT_US",
                category="diagnosis",
            ),
            "brain natriuretic peptide": UMLSConcept(
                cui="C0002",
                preferred_term="Brain natriuretic peptide",
                semantic_type="Laboratory Procedure",
                source_vocabulary="LNC",
                category="lab_or_measurement",
            ),
            "coronary artery disease": UMLSConcept(
                cui="C0003",
                preferred_term="Coronary artery disease",
                semantic_type="Disease or Syndrome",
                source_vocabulary="SNOMEDCT_US",
                category="diagnosis",
            ),
            "creatinine": UMLSConcept(
                cui="C0004",
                preferred_term="Creatinine",
                semantic_type="Laboratory Procedure",
                source_vocabulary="LNC",
                category="lab_or_measurement",
            ),
            "furosemide": UMLSConcept(
                cui="C0005",
                preferred_term="Furosemide",
                semantic_type="Pharmacologic Substance",
                source_vocabulary="RXNORM",
                category="medication",
            ),
            "lasix": UMLSConcept(
                cui="C0005",
                preferred_term="Furosemide",
                semantic_type="Pharmacologic Substance",
                source_vocabulary="RXNORM",
                category="medication",
            ),
            "losartan": UMLSConcept(
                cui="C0006",
                preferred_term="Losartan",
                semantic_type="Pharmacologic Substance",
                source_vocabulary="RXNORM",
                category="medication",
            ),
        }

    def best_match(self, term: str):
        return self.mapping.get(" ".join(term.lower().split()))


def test_hybrid_extractor_builds_entities_and_relations():
    extractor = HybridClinicalPathExtractor(umls_client=_FakeUMLSClient(), candidate_limit=12)
    node = Document(
        text=(
            "Brief Hospital Course:\n"
            "#) Acute kidney injury: Resolved. Patient with elevated creatinine to 1.7 on admission. "
            "Her losartan was initially held but was restarted by discharge.\n"
            "Patient was diuresed with IV lasix with good response.\n"
            "ADMISSION LABS: BNP-3199 Creat-1.7\n"
        ),
        id_="chunk-1",
    )

    [transformed] = extractor([node])
    kg_nodes = transformed.metadata["nodes"]
    kg_relations = transformed.metadata["relations"]

    labels_by_name = {(kg_node.label, kg_node.name) for kg_node in kg_nodes}
    relation_labels = {relation.label for relation in kg_relations}

    assert ("DISEASE", "Acute kidney injury") in labels_by_name
    assert ("MEDICATION", "Losartan") in labels_by_name
    assert ("MEDICATION", "Furosemide") in labels_by_name
    assert ("LAB_TEST", "Creatinine") in labels_by_name
    assert ("LAB_RESULT", "Creatinine = 1.7") in labels_by_name
    assert "MENTIONS" in relation_labels
    assert "HAS_RESULT" in relation_labels
    assert "HELD" in relation_labels
    assert "RESTARTED" in relation_labels
    assert "ADMINISTERED" in relation_labels
    losartan = next(kg_node for kg_node in kg_nodes if kg_node.name == "Losartan")
    assert getattr(losartan, "canonical_id", "").startswith("medication:")
    assert losartan.properties["mention_count"] >= 1


def test_build_document_keeps_umls_hints_out_of_source_text(tmp_path):
    input_dir = tmp_path / "evidence"
    input_dir.mkdir()
    note = input_dir / "note.txt"
    note.write_text("SOURCE TEXT:\nCreat-1.7", encoding="utf-8")

    document = build_document(note, input_dir, hint_block="UMLS concept hints:\n- DISEASE: Example")

    assert document.metadata["umls_hint_block"].startswith("UMLS concept hints:")
    assert document.get_content(metadata_mode=MetadataMode.NONE) == "SOURCE TEXT:\nCreat-1.7"
    assert "umls_hint_block" in document.excluded_embed_metadata_keys
    assert "umls_hint_block" in document.excluded_llm_metadata_keys
