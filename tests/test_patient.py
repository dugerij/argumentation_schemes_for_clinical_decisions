from pathlib import Path

from clinical_cds.cli import build_parser
from clinical_cds.patient import load_patient_case


def test_patient_example_loads_as_a_diagnostic_case():
    repository_root = Path(__file__).resolve().parents[1]

    case = load_patient_case(repository_root / "examples/patient.example.json")

    assert case.dataset == "submitted_patient"
    assert case.task == "diagnosis"
    assert case.gold_label == ""
    assert case.sections["chief_complaint"] == "Chest pain"


def test_diagnose_command_accepts_local_umls_index():
    args = build_parser().parse_args(
        [
            "diagnose",
            "--patient",
            "examples/patient.example.json",
            "--umls-db",
            "output/cache/umls_local.sqlite3",
        ]
    )

    assert args.command == "diagnose"
    assert args.umls_db == Path("output/cache/umls_local.sqlite3")
    assert args.mode == "symbolic_argument"
