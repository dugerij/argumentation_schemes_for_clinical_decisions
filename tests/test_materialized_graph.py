import io
import zipfile

from retrieval.cds_graph import CdsMaterializedGraphStore, build_cds_materialized_graph


def _write_member(archive: zipfile.ZipFile, name: str, text: str) -> None:
    archive.writestr(name, text.encode("utf-8"))


def _build_sample_cds_zip(path) -> None:
    clinical_csv = (
        "stay_id,text,HPI,tests,past_medication,diagnosis,primary_diagnosis,secondary_diagnosis\n"
        '101,"note","fever and cough","cxr infiltrate","none","pneumonia","['"'Pneumonia'"']","[]"\n'
        '102,"note","chest pain and dyspnea","troponin elevated","aspirin","myocardial infarction","['"'Myocardial infarction'"']","[]"\n'
    )
    diagnosis_csv = (
        "stay_id,HPI,patient_info,initial_vitals,diagnosis,primary_diagnosis,secondary_diagnosis\n"
        '101,"fever and cough","male age 70","hr 110 temp 38.5","pneumonia","['"'Pneumonia'"']","[]"\n'
        '102,"chest pain and dyspnea","female age 64","hr 95 bp 150/90","myocardial infarction","['"'Myocardial infarction'"']","[]"\n'
    )
    initial_csv = (
        "stay_id,triage,pain,chiefcomplaint,arrival_transport,disposition,icd_code,icd_title,icd_version\n"
        '101,"3","2","cough","walk-in","home","J18","pneumonia","10"\n'
        '102,"2","8","chest pain","ambulance","admit","I21","mi","10"\n'
    )
    specialty_csv = (
        "stay_id,HPI,patient_info,initial_vitals,specialty\n"
        '101,"fever and cough","male age 70","hr 110 temp 38.5","['"'Pulmonology'"']"\n'
        '102,"chest pain and dyspnea","female age 64","hr 95 bp 150/90","['"'Cardiology'"']"\n'
    )

    nested_buffer = io.BytesIO()
    with zipfile.ZipFile(nested_buffer, "w") as nested_archive:
        _write_member(nested_archive, "clinical_data.csv", clinical_csv)

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("clinical_data.csv.zip", nested_buffer.getvalue())
        _write_member(archive, "diagnosis.csv", diagnosis_csv)
        _write_member(archive, "initial_assessment_info.csv", initial_csv)
        _write_member(archive, "specialty_referral.csv", specialty_csv)


def test_build_materialized_graph_from_cds_archive(tmp_path):
    zip_path = tmp_path / "cases.zip"
    output_dir = tmp_path / "graph"
    _build_sample_cds_zip(zip_path)

    stats = build_cds_materialized_graph(zip_path=zip_path, output_dir=output_dir)

    assert stats.case_count == 2
    assert stats.diagnosis_case_count == 2
    assert stats.triage_case_count == 2
    assert stats.specialty_case_count == 2

    store = CdsMaterializedGraphStore.from_persist_dir(output_dir)
    case = store.get_case(101)

    assert case is not None
    assert case.primary_diagnosis == ("Pneumonia",)
    assert 101 in store.search_stay_ids("fever cough")
