from clinical_cds.argumentation import KnowledgeClaim, KnowledgeRole
from clinical_cds.runner import _rank_dossier_binding_options


def _claim(evidence_id, role, text="LVEF >=50%"):
    return KnowledgeClaim(
        evidence_id=evidence_id,
        node_id=evidence_id,
        diagnosis_label="",
        diagnostic_path=(),
        premise_type="",
        text=text,
        role=role,
        counterevidence_eligible=False,
    )


def test_admissible_pair_is_marked_typed_binding_admissible():
    patient_pairs = (("S1", "Global systolic function is normal (LVEF >55%)."),)
    claims = (_claim("K1", KnowledgeRole.DIAGNOSTIC_CRITERION),)

    options = _rank_dossier_binding_options(patient_pairs, ("K1",), claims, None)

    assert len(options) == 1
    assert options[0]["typed_binding_admissible"] is True


def test_non_admissible_pair_is_marked_not_typed_binding_admissible():
    patient_pairs = (("S1", "Patient denies any cardiac history."),)
    claims = (_claim("K1", KnowledgeRole.DIAGNOSTIC_CRITERION),)

    options = _rank_dossier_binding_options(patient_pairs, ("K1",), claims, None)

    assert len(options) == 1
    assert options[0]["typed_binding_admissible"] is False


def test_dossier_ranking_prefers_admissible_pairs_first():
    # An admissible diagnostic-criterion pairing and an inadmissible one
    # competing for the same knowledge id. Ranking must put the admissible
    # pairing first -- callers rely on this order when deduplicating by
    # knowledge node -- even though the pool-padding step (which fills the
    # dossier up to the bounded option count when few pairs are available)
    # still surfaces the inadmissible one as a second, lower-priority entry.
    patient_pairs = (
        ("S1", "Global systolic function is normal (LVEF >55%)."),
        ("S2", "Patient denies any cardiac history."),
    )
    claims = (_claim("K1", KnowledgeRole.DIAGNOSTIC_CRITERION),)

    options = _rank_dossier_binding_options(patient_pairs, ("K1",), claims, None)

    assert options[0]["patient_evidence_id"] == "S1"
    assert options[0]["typed_binding_admissible"] is True
