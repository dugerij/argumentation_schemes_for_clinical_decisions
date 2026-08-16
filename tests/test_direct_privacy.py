import pytest

from clinical_cds.direct import _parse_annotation, _parse_annotation_key


RAW_NARRATIVE = "Blood pressure is 170/100 mmHg and patient reports chest pain"


def test_malformed_annotation_key_error_omits_raw_clinical_text():
    with pytest.raises(ValueError) as excinfo:
        _parse_annotation_key(RAW_NARRATIVE)

    assert RAW_NARRATIVE not in str(excinfo.value)
    assert "sha256=" in str(excinfo.value)


def test_non_diagnosis_annotation_root_error_omits_raw_clinical_text():
    with pytest.raises(ValueError) as excinfo:
        _parse_annotation("case-1", f"{RAW_NARRATIVE}$Cause_1", {})

    assert RAW_NARRATIVE not in str(excinfo.value)
    assert "sha256=" in str(excinfo.value)
