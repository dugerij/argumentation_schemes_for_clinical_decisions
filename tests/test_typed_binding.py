from types import SimpleNamespace

from clinical_cds.typed_binding import (
    Polarity,
    assess_family_diagnostic_test_result,
    assess_typed_binding,
    clinical_atom,
    controlled_diagnosis_assertion,
    deterministic_binding_authority,
    patient_atomic_spans,
)


def test_normal_description_does_not_manufacture_negative_polarity():
    atom = clinical_atom(
        "Global systolic function is normal (LVEF >55%)."
    )

    assert atom.polarity == Polarity.PRESENT


def test_preserved_ef_executes_positive_numeric_threshold():
    assessment = assess_typed_binding(
        "Global systolic function is normal (LVEF >55%).",
        "LVEF >=50%",
        role="diagnostic_criterion",
    )

    assert assessment.admissible is True


class FakeUMLSNormalizer:
    def concept(self, term, categories=None):
        del categories
        key = " ".join(term.casefold().split())
        concepts = {
            "pulmonary embolism": "C0034065",
            "pe": "C0034065",
            "troponin": "C0077404",
            "cardiac troponin": "C0077404",
            "pulmonary artery": "C0034052",
            "pulmonary artery thrombus": "C9990001",
            "stenosis": "C1261287",
            "non-st-elevation": "C5205888",
            "nste-acs": "C5205888",
        }
        cui = concepts.get(key)
        return SimpleNamespace(cui=cui) if cui else None


def test_umls_synonyms_support_concept_compatibility():
    result = assess_typed_binding(
        "Current PE confirmed on imaging",
        "Pulmonary embolism confirmed on imaging",
        role="diagnostic_criterion",
        normalizer=FakeUMLSNormalizer(),
    )

    assert result.admissible
    assert result.patient_atom.cuis == ("C0034065",)


def test_completed_positive_diagnostic_test_can_establish_family_only():
    result = assess_family_diagnostic_test_result(
        "Current CTPA showed pulmonary embolism.",
        "Pulmonary Embolism",
        normalizer=FakeUMLSNormalizer(),
    )

    assert result.admissible
    assert next(
        item for item in result.critical_questions
        if item.question_id == "family_only_scope"
    ).passed
    assert assess_family_diagnostic_test_result(
        "CT chest showed bilateral pulmonary emboli.",
        "Pulmonary Embolism",
        normalizer=FakeUMLSNormalizer(),
    ).admissible


def test_ordered_negative_historical_or_equivocal_tests_never_certify_family():
    findings = (
        "CTPA was ordered to rule out pulmonary embolism.",
        "CTPA was negative for pulmonary embolism.",
        "History of CT-confirmed pulmonary embolism five years ago.",
        "CTPA was suspicious for possible pulmonary embolism.",
    )
    assert all(
        not assess_family_diagnostic_test_result(
            finding,
            "Pulmonary Embolism",
            normalizer=FakeUMLSNormalizer(),
        ).admissible
        for finding in findings
    )


def test_completed_test_result_can_bind_matching_test_criterion_but_not_a_plan():
    criterion = "Endoscopy shows gastric erosions in the antrum."

    assert assess_typed_binding(
        "EGD showed several gastric erosions in the antrum.",
        criterion,
        role="diagnostic_criterion",
    ).admissible
    assert not assess_typed_binding(
        "EGD was ordered to evaluate possible gastric erosions.",
        criterion,
        role="diagnostic_criterion",
    ).admissible


def test_deterministic_authority_rejects_mismatched_test_modalities():
    authority = deterministic_binding_authority(
        "Coronary angiography showed 90% RCA stenosis.",
        "Endoscopy shows erosive changes supporting GERD.",
        candidate_label="GERD",
        role="diagnostic_criterion",
    )

    assert not authority.authorized
    assert authority.authority_type == "semantic_review"


def test_controlled_child_assertion_may_use_umls_but_not_related_manifestation():
    assert controlled_diagnosis_assertion(
        "non-ST-elevation",
        "non-ST-elevation",
        "NSTE-ACS",
        normalizer=FakeUMLSNormalizer(),
    )
    assert not controlled_diagnosis_assertion(
        "Endoscopy showed a gastric ulcer with bleeding",
        "Bleeding is observed",
        "Upper Gastrointestinal Bleeding",
        normalizer=FakeUMLSNormalizer(),
    )


def test_deterministic_authority_rejects_historical_diagnosis():
    authority = deterministic_binding_authority(
        "History of pulmonary embolism five years ago.",
        "Pulmonary embolism is present.",
        candidate_label="Pulmonary Embolism",
        role="diagnostic_criterion",
        normalizer=FakeUMLSNormalizer(),
    )

    assert not authority.authorized


def test_deterministic_authority_accepts_executable_numeric_threshold():
    authority = deterministic_binding_authority(
        "Current LVEF = 35%.",
        "LVEF <=40%.",
        candidate_label="Heart Failure",
        role="diagnostic_criterion",
    )

    assert authority.authorized
    assert authority.authority_type == "exact_measurement"


def test_executable_child_binding_may_compose_to_its_graph_parent():
    authority = deterministic_binding_authority(
        "Current LVEF = 35%.",
        "LVEF <=40%.",
        candidate_label="Heart Failure",
        role="diagnostic_criterion",
    )

    assert authority.authorized
    assert authority.authority_type == "exact_measurement"

def test_historical_diagnosis_does_not_prove_current_disease():
    result = assess_typed_binding(
        "History of pulmonary embolism five years ago",
        "Pulmonary embolism is present",
        role="diagnostic_criterion",
        normalizer=FakeUMLSNormalizer(),
    )

    assert not result.admissible
    assert not next(
        item for item in result.critical_questions
        if item.question_id == "temporality_compatible"
    ).passed


def test_negative_observation_cannot_be_positive_support():
    result = assess_typed_binding(
        "No pulmonary embolism on CT",
        "Pulmonary embolism is present on CT",
        role="diagnostic_criterion",
        normalizer=FakeUMLSNormalizer(),
    )

    assert not result.admissible


def test_negative_observation_can_satisfy_an_absence_based_criterion():
    result = assess_typed_binding(
        "Troponin negative x2",
        "No troponin elevation",
        role="diagnostic_criterion",
    )

    assert result.admissible


def test_shared_normality_word_cannot_bind_unrelated_absent_findings():
    result = assess_typed_binding(
        "Aortic diameters are normal",
        "Troponin levels are normal; normal ECG",
        role="diagnostic_criterion",
    )

    assert not result.admissible


def test_generic_levels_word_cannot_bind_unrelated_absent_findings():
    result = assess_typed_binding(
        "The diameters of aorta at the sinus, ascending and arch levels are normal.",
        "hs-cTn levels are normal, Normal ECG",
        role="diagnostic_criterion",
    )

    assert not result.admissible


def test_serial_abnormal_compact_troponin_rise_supports_elevated_biomarker():
    result = assess_typed_binding(
        (
            "cTropnT-0.02* CK-MB-50* cTropnT-0.35* "
            "cTropnT-0.49* cTropnT-0.80*"
        ),
        "Elevated levels of cardiac biomarkers, especially troponin T or I",
        role="diagnostic_criterion",
    )

    assert result.admissible


def test_single_abnormal_compact_troponin_does_not_imply_elevation():
    result = assess_typed_binding(
        "cTropnT-0.02*",
        "Elevated levels of cardiac biomarkers, especially troponin T or I",
        role="diagnostic_criterion",
    )

    assert not result.admissible


def test_serial_troponin_cannot_complete_multi_part_pe_criterion():
    result = assess_typed_binding(
        "cTropnT-0.02* cTropnT-0.35* cTropnT-0.49* cTropnT-0.80*",
        (
            "The patient is hemodynamically stable, but there is evidence of "
            "RV functional impairment, such as RV enlargement or ventricular "
            "septal deviation on echocardiography, and elevated blood "
            "biomarkers such as cardiac troponin."
        ),
        role="diagnostic_criterion",
    )

    assert not result.admissible


def test_timestamp_delimited_serial_labs_remain_one_exact_atomic_span():
    source = (
        "Cardiac Enzymes: 01:44PM BLOOD cTropnT-0.02* "
        "12:50AM BLOOD CK-MB-50* cTropnT-0.35* "
        "02:00AM BLOOD cTropnT-0.49* 07:00AM BLOOD cTropnT-0.80*"
    )

    spans = patient_atomic_spans(source)

    assert any(
        span.startswith("cTropnT-0.02*") and span.endswith("cTropnT-0.80*")
        for span in spans
    )


def test_st_segment_depression_satisfies_non_st_elevation_criterion():
    result = assess_typed_binding(
        "Since the previous tracing ST segment depression in V2 is more prominent.",
        "ECG non-ST-elevation",
        role="diagnostic_criterion",
    )

    assert result.admissible
    assert result.score == 1.0


def test_ecg_std_abbreviation_satisfies_non_st_elevation_criterion():
    for finding in (
        "ETT showed 0.5 mm horizontal STD V4-V6.",
        "New isolated concordant STD in V1.",
    ):
        result = assess_typed_binding(
            finding,
            "ECG non-ST-elevation",
            role="diagnostic_criterion",
        )
        assert result.admissible


def test_numeric_threshold_is_executable():
    passing = assess_typed_binding(
        "Blood pressure is 170/100 mmHg",
        "Repeated blood pressure above 140/90 mmHg",
        role="diagnostic_criterion",
    )
    failing = assess_typed_binding(
        "Blood pressure is 120/70 mmHg",
        "Repeated blood pressure above 140/90 mmHg",
        role="diagnostic_criterion",
    )

    assert passing.admissible
    assert not failing.admissible


def test_unicode_numeric_threshold_operators_are_executable():
    assert not assess_typed_binding(
        "LVEF = 41%",
        "LVEF≥50%",
        role="diagnostic_criterion",
    ).admissible


def test_numeric_range_is_executable():
    assert assess_typed_binding(
        "Left ventriculogram LVEF = 45%",
        "LVEF 41–49%",
        role="diagnostic_criterion",
    ).admissible
    assert not assess_typed_binding(
        "Left ventriculogram LVEF = 25%",
        "LVEF 41–49%",
        role="diagnostic_criterion",
    ).admissible


def test_unitless_context_number_cannot_satisfy_percentage_threshold():
    result = assess_typed_binding(
        "TRACING #3: ECHO: LVEF = 41%",
        "LVEF≤40%",
        role="diagnostic_criterion",
    )

    assert not result.admissible
    assert not next(
        item for item in result.critical_questions
        if item.question_id == "quantity_compatible"
    ).passed


def test_worded_duration_threshold_is_executable():
    assert not assess_typed_binding(
        "A mild episode of chest pain earlier last week has resolved.",
        "AF episodes that last more than 7 days.",
        role="diagnostic_criterion",
    ).admissible
    assert assess_typed_binding(
        "Current AF episodes last 9 days.",
        "AF episodes that last more than 7 days.",
        role="diagnostic_criterion",
    ).admissible
    assert assess_typed_binding(
        "LVEF = 55%",
        "LVEF≥50%",
        role="diagnostic_criterion",
    ).admissible
    assert not assess_typed_binding(
        "LVEF = 41%",
        "LVEF≤40%",
        role="diagnostic_criterion",
    ).admissible


def test_related_anatomy_does_not_satisfy_a_different_atomic_criterion():
    result = assess_typed_binding(
        "Mild pulmonary artery systolic hypertension",
        "CT pulmonary angiography directly displays pulmonary artery thrombus",
        role="diagnostic_criterion",
    )

    assert not result.admissible


def test_umls_overlap_does_not_make_a_long_criterion_true():
    result = assess_typed_binding(
        "Mild pulmonary artery systolic hypertension",
        (
            "CT pulmonary angiography directly displays pulmonary artery "
            "thrombus and confirms acute embolism"
        ),
        role="diagnostic_criterion",
        normalizer=FakeUMLSNormalizer(),
    )

    assert not result.admissible
    criterion_question = next(
        item for item in result.critical_questions
        if item.question_id == "criterion_satisfied"
    )
    assert not criterion_question.passed


def test_narrative_peak_troponin_statement_satisfies_verbose_lab_criterion():
    # A prose peak-value statement ("Peak troponin greater than 8") shares a
    # comparison bigram ("peak troponin") and a parseable quantity with the
    # KG's verbose hs-cTn threshold sentence, even though most of that
    # sentence's boilerplate ("...of the normal control value") never
    # lexically overlaps. The full-sentence coverage score alone used to
    # block this (score=0.25, below the old 0.6/0.4 thresholds).
    result = assess_typed_binding(
        "Peak troponin greater than 8. Treated ICU with a heparin gtt.",
        "The peak hs-cTn exceeded the 99th percentile of the normal control value",
        role="diagnostic_criterion",
    )

    assert result.admissible
    authority = deterministic_binding_authority(
        "Peak troponin greater than 8. Treated ICU with a heparin gtt.",
        "The peak hs-cTn exceeded the 99th percentile of the normal control value",
        candidate_label="NSTEMI",
        role="diagnostic_criterion",
    )
    assert authority.authorized
    assert authority.authority_type == "exact_measurement"


def test_shared_quantity_and_phrase_still_require_quantity_agreement():
    # The relaxed criterion_satisfied branch must not bypass the separate,
    # independently-required quantity_compatible check: a shared "peak
    # troponin" phrase with a value that does not clear a real, parseable
    # threshold must remain inadmissible.
    result = assess_typed_binding(
        "Peak troponin 0.02, well within the reference range.",
        "The peak troponin is above 50 ng/mL",
        role="diagnostic_criterion",
    )

    assert not result.admissible


def test_bare_patient_value_cannot_satisfy_an_unverifiable_criterion_threshold():
    # "The peak hs-cTn exceeded the 99th percentile of the normal control
    # value" has no threshold quantity_compatible can actually check --
    # "exceeded" sits before "the 99th percentile", not adjacent to a real
    # number, so it never becomes a comparable operator. When neither side
    # states an explicit, checkable direction, a bare patient value ("0.02,
    # well within normal limits") must not be treated as satisfying it --
    # unlike test_narrative_peak_troponin_statement_satisfies_verbose_lab_
    # criterion below, where the patient text itself states "greater than".
    result = assess_typed_binding(
        "Peak troponin 0.02, well within normal limits.",
        "The peak hs-cTn exceeded the 99th percentile of the normal control value",
        role="diagnostic_criterion",
    )

    assert not result.admissible


def test_narrative_comparator_synonyms_are_recognized_operators():
    # "greater than" and "exceeded"/"exceeds" are common clinical narrative
    # phrasing for numeric thresholds but were not in NUMBER_RE's operator
    # list (only "more than"/"above"/"over" were), silently defaulting to
    # equality and disabling quantity_compatible's threshold check whenever
    # a criterion or patient statement used them instead.
    assert assess_typed_binding(
        "Troponin exceeded 80 ng/mL on repeat draw.",
        "Troponin above 50 ng/mL",
        role="diagnostic_criterion",
    ).admissible
    assert not assess_typed_binding(
        "Troponin exceeded 10 ng/mL on repeat draw.",
        "Troponin above 50 ng/mL",
        role="diagnostic_criterion",
    ).admissible


def test_long_guideline_is_evaluated_as_atomic_sentences():
    result = assess_typed_binding(
        "ECG shows non-ST-elevation",
        (
            "CT angiography displays pulmonary artery thrombus. "
            "ECG shows non-ST-elevation. "
            "D-dimer below threshold can exclude embolism."
        ),
        role="diagnostic_criterion",
    )

    assert result.admissible
    assert result.matched_clause == "ECG shows non-ST-elevation."


def test_umls_generic_finding_cannot_erase_quantified_anatomic_context():
    result = assess_typed_binding(
        "90% LCX and RCA stenosis",
        "CTA/MRA demonstrates carotid or intracranial arterial stenosis >=50%",
        role="diagnostic_criterion",
        normalizer=FakeUMLSNormalizer(),
    )

    assert not result.admissible
    assert not next(
        item for item in result.critical_questions
        if item.question_id == "criterion_satisfied"
    ).passed
