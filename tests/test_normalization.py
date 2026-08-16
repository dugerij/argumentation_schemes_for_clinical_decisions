import sqlite3

from clinical_cds.normalization import (
    UMLSNormalizer,
    diagnosis_path_compatibility,
    diagnosis_family_key,
    graph_label_umls_query,
    lexical_diagnosis_key,
)
from clinical_cds.terminology.schema import UMLSConcept
from clinical_cds.terminology.local_umls import (
    LocalUMLSClient,
    build_local_umls_subset,
)
from clinical_cds.evaluation import (
    diagnosis_family_match,
    exact_label_match,
    hierarchy_score,
)


def test_diagnosis_path_compatibility_distinguishes_hierarchy_levels():
    path = (
        "Suspected Pulmonary Embolism",
        "Pulmonary Embolism",
        "Low-risk PE",
    )

    assert diagnosis_path_compatibility("Low-risk PE", path) == 1.0
    assert diagnosis_path_compatibility("Pulmonary Embolism", path) == 0.75
    assert diagnosis_path_compatibility("Suspected Pulmonary Embolism", path) == 0.5


def test_diagnosis_path_compatibility_rejects_sibling_diagnoses():
    bacterial_path = (
        "Suspected Pneumonia",
        "Pneumonia",
        "Bacterial Pneumonia",
    )

    assert diagnosis_path_compatibility("Viral Pneumonia", bacterial_path) == 0.0


def test_evaluation_reports_parent_credit_without_exact_credit():
    relations = {
        "pulmonaryembolism": {"lowriskpe": 1},
        "lowriskpe": {"pulmonaryembolism": 1},
        "pneumonia": {"bacterialpneumonia": 1, "viralpneumonia": 1},
        "bacterialpneumonia": {"pneumonia": 1},
        "viralpneumonia": {"pneumonia": 1},
    }

    assert hierarchy_score(
        "Pulmonary Embolism",
        "Low-risk PE",
        relations,
    ) == 0.75
    assert hierarchy_score(
        "Viral Pneumonia",
        "Bacterial Pneumonia",
        relations,
    ) == 0.0
    assert hierarchy_score(
        "Pneumonia",
        "Bacterial Pneumonia",
        relations,
    ) == 0.75


def test_expanded_nstemi_is_an_exact_semantic_match():
    assert exact_label_match(
        "Non-ST Elevation Myocardial Infarction (NSTEMI)",
        "NSTEMI",
    ) == 1.0


def test_lexical_equivalence_takes_precedence_over_noisy_concept_mapping():
    class NoisyNormalizer:
        def diagnosis_key(self, value: str) -> str:
            return f"incorrect:{value}"

    assert exact_label_match(
        "Non-ST Elevation Myocardial Infarction (NSTEMI)",
        "NSTEMI",
        NoisyNormalizer(),
    ) == 1.0


def test_missing_subtype_or_aetiology_receives_parent_credit():
    assert hierarchy_score(
        "Multiple Sclerosis",
        "Relapsing-Remitting Multiple Sclerosis",
        {},
    ) == 0.75
    assert hierarchy_score(
        "Pneumonia",
        "Bacterial Pneumonia",
        {},
    ) == 0.75
    assert hierarchy_score(
        "Pulmonary Embolism",
        "Low-risk PE",
        {},
    ) == 0.75


def test_orthogonal_copd_modifiers_receive_family_not_exact_credit():
    assert exact_label_match("COPD Exacerbation", "Severe COPD") == 0.0
    assert hierarchy_score("COPD Exacerbation", "Severe COPD", {}) == 0.5
    assert diagnosis_family_match("COPD Exacerbation", "Severe COPD") == 1.0


def test_conflicting_pneumonia_aetiologies_only_match_at_family_level():
    assert hierarchy_score("Viral Pneumonia", "Bacterial Pneumonia", {}) == 0.0
    assert diagnosis_family_match(
        "Viral Pneumonia",
        "Bacterial Pneumonia",
    ) == 0.0


def test_reviewed_graph_label_queries_expand_ambiguous_terms():
    assert graph_label_umls_query("UA") == "unstable angina"
    assert graph_label_umls_query("HFpEF") == (
        "heart failure with preserved ejection fraction"
    )
    assert graph_label_umls_query("Suspected Alzheimer") == "Alzheimer disease"
    assert graph_label_umls_query("Thyroid Nodules") == "thyroid nodule"


def test_shared_umls_concept_preserves_specificity_modifiers():
    class SharedPulmonaryEmbolismConcept:
        def diagnosis_key(self, value: str) -> str:
            del value
            return "cui:c0034065"

    normalizer = SharedPulmonaryEmbolismConcept()

    assert exact_label_match(
        "Pulmonary Embolism",
        "Low-risk PE",
        normalizer,
    ) == 0.0
    assert hierarchy_score(
        "Pulmonary Embolism",
        "Low-risk PE",
        {},
        normalizer,
    ) == 0.75


class _SharedBroadConceptClient:
    source_vocabularies = ("SNOMEDCT_US",)
    supports_full_alias_lookup = True
    database_id = "sha256:test"

    def best_match(self, term):
        preferred = (
            "Hypertensive disorder"
            if "hypertension" in term.casefold()
            else "Heart failure"
        )
        cui = "C0020538" if "hypertension" in term.casefold() else "C0018801"
        return UMLSConcept(
            cui=cui,
            preferred_term=preferred,
            semantic_type="Disease or Syndrome",
            source_vocabulary="SNOMEDCT_US",
            category="diagnosis",
        )

    def concept_terms(self, cui, limit=3):
        del limit
        if cui == "C0020538":
            return ("Hypertension", "Severe hypertension")
        return ("Heart failure", "Heart failure with reduced ejection fraction")


def test_umls_canonical_key_retains_subtype_and_severity_qualifiers():
    normalizer = UMLSNormalizer(client=_SharedBroadConceptClient())

    assert normalizer.diagnosis_key("HFrEF") != normalizer.diagnosis_key(
        "Heart Failure"
    )
    assert normalizer.diagnosis_key("Severe hypertension") != (
        normalizer.diagnosis_key("Hypertension")
    )
    assert diagnosis_family_key("Severe hypertension") == "hypertension"


class _NonDiagnosisConceptClient:
    """A UMLS concept lookup that only ever returns a non-diagnosis match."""

    source_vocabularies = ("SNOMEDCT_US",)
    supports_full_alias_lookup = True
    database_id = "sha256:test"

    def best_match(self, term):
        del term
        return UMLSConcept(
            cui="C9999999",
            preferred_term="Chest pain",
            semantic_type="Sign or Symptom",
            source_vocabulary="SNOMEDCT_US",
            category="finding",
        )

    def concept_terms(self, cui, limit=3):
        del cui, limit
        return ()


def test_crosswalk_query_still_requires_a_diagnosis_category_match():
    # "UA" is a reviewed crosswalk label (graph_label_umls_query expands it
    # to "unstable angina"), so diagnosis_key takes the crosswalk branch.
    # UMLS must still only ever establish terminology identity for an actual
    # diagnosis concept, not any matched concept regardless of category --
    # a "finding" match must be rejected the same way a non-crosswalk term
    # rejects it, falling back to the lexical key.
    normalizer = UMLSNormalizer(client=_NonDiagnosisConceptClient())

    assert graph_label_umls_query("UA") == "unstable angina"
    assert normalizer.diagnosis_key("UA") == lexical_diagnosis_key("UA")
    assert not normalizer.diagnosis_key("UA").startswith("cui:")


def test_umls_expansion_does_not_drop_source_specificity():
    normalizer = UMLSNormalizer(client=_SharedBroadConceptClient())

    heart_failure_expansions = normalizer.expand_text("Known HFrEF")
    hypertension_expansions = normalizer.expand_text("Severe hypertension")

    assert "Heart failure" not in heart_failure_expansions
    assert "Heart failure with reduced ejection fraction" in heart_failure_expansions
    assert "Hypertension" not in hypertension_expansions
    assert "Severe hypertension" in hypertension_expansions


def test_reordered_low_risk_pe_is_an_exact_match():
    assert exact_label_match(
        "Pulmonary Embolism, low-risk",
        "Low-risk PE",
    ) == 1.0


def test_asthma_copd_overlap_is_not_severe_copd():
    assert diagnosis_family_match("Asthma-COPD", "Severe COPD") == 1.0
    assert hierarchy_score("Asthma-COPD", "Severe COPD", {}) == 0.5


def test_acs_subtypes_share_family_but_conflict_as_diagnoses():
    assert diagnosis_family_match("STEMI-ACS", "NSTE-ACS") == 0.0
    assert hierarchy_score("STEMI-ACS", "NSTE-ACS", {}) == 0.0


def test_runtime_umls_subset_keeps_all_aliases_for_matched_cui(tmp_path):
    source = tmp_path / "full.sqlite3"
    with sqlite3.connect(source) as conn:
        conn.executescript(
            """
            CREATE TABLE umls_semantic_types (
                cui TEXT PRIMARY KEY, semantic_type TEXT, category TEXT
            );
            CREATE TABLE umls_preferred_terms (
                cui TEXT, source_vocabulary TEXT, preferred_term TEXT
            );
            CREATE TABLE umls_terms (
                cui TEXT, normalized_term TEXT, term TEXT, preferred_term TEXT,
                semantic_type TEXT, source_vocabulary TEXT, source_code TEXT,
                category TEXT, is_preferred INTEGER, source_rank INTEGER
            );
            """
        )
        conn.execute(
            "INSERT INTO umls_semantic_types VALUES (?, ?, ?)",
            ("C1", "Disease or Syndrome", "diagnosis"),
        )
        conn.execute(
            "INSERT INTO umls_preferred_terms VALUES (?, ?, ?)",
            ("C1", "SNOMEDCT_US", "Pulmonary embolism"),
        )
        conn.executemany(
            "INSERT INTO umls_terms VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ("C1", "pulmonary embolism", "Pulmonary embolism", "Pulmonary embolism", "Disease or Syndrome", "SNOMEDCT_US", "59282003", "diagnosis", 1, 0),
                ("C1", "pe", "PE", "Pulmonary embolism", "Disease or Syndrome", "SNOMEDCT_US", "59282003", "diagnosis", 0, 0),
            ),
        )

    subset = build_local_umls_subset(source, tmp_path / "subset.sqlite3", ["PE"])
    client = LocalUMLSClient(subset, lookup_cache_db_path=tmp_path / "cache.sqlite3")

    assert client.best_match("pulmonary embolism").cui == "C1"
    assert "PE" in client.concept_terms("C1")
