import csv
import json
from pathlib import Path
from unittest.mock import patch

import make_domain_subset


def test_make_domain_subset_main_writes_matching_notes_and_questions(tmp_path):
    notes_csv = tmp_path / "discharge.csv"
    questions_path = tmp_path / "questions.jsonl"
    notes_output_dir = tmp_path / "notes"
    questions_output_path = tmp_path / "renal_questions.jsonl"

    with notes_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "note_id",
                "subject_id",
                "hadm_id",
                "note_type",
                "note_seq",
                "charttime",
                "storetime",
                "text",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "note_id": "n1",
                "subject_id": "s1",
                "hadm_id": "h1",
                "note_type": "DS",
                "note_seq": 1,
                "charttime": "",
                "storetime": "",
                "text": "Chronic kidney disease with hyperkalemia and dialysis planning.",
            }
        )
        writer.writerow(
            {
                "note_id": "n2",
                "subject_id": "s2",
                "hadm_id": "h2",
                "note_type": "DS",
                "note_seq": 1,
                "charttime": "",
                "storetime": "",
                "text": "Ankle sprain after a fall.",
            }
        )

    questions = [
        {
            "question": "What is the best next step for chronic kidney disease with hyperkalemia?",
            "options": {"A": "Dialysis", "B": "Bandage"},
        },
        {
            "question": "What is the best treatment for ankle sprain?",
            "options": {"A": "Rest", "B": "Dialysis"},
        },
    ]
    with questions_path.open("w", encoding="utf-8") as file:
        for item in questions:
            file.write(json.dumps(item) + "\n")

    argv = [
        "make_domain_subset.py",
        "--notes-csv-path",
        str(notes_csv),
        "--notes-output-dir",
        str(notes_output_dir),
        "--questions-input-path",
        str(questions_path),
        "--questions-output-path",
        str(questions_output_path),
        "--domain",
        "renal_metabolic",
        "--seed-term",
        "chronic kidney disease",
        "--seed-term",
        "hyperkalemia",
        "--min-domain-hits",
        "1",
    ]

    class _FakeMatcher:
        def match_details(self, text: str):
            lowered = text.lower()
            if "kidney" in lowered or "hyperkalemia" in lowered or "dialysis" in lowered:
                return 2, ["chronic kidney disease", "hyperkalemia"]
            return 0, []

    with patch("sys.argv", argv), patch("make_domain_subset.build_domain_matcher", return_value=_FakeMatcher()):
        make_domain_subset.main()

    written_questions = [json.loads(line) for line in questions_output_path.read_text(encoding="utf-8").splitlines()]
    assert len(written_questions) == 1
    assert "chronic kidney disease" in written_questions[0]["question"].lower()
    manifest = json.loads(questions_output_path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert manifest["note_count"] == 1
    assert manifest["question_count"] == 1
    assert manifest["seed_terms"] == ["chronic kidney disease", "hyperkalemia"]
    assert manifest["notes_matcher"] == "vocab"
    assert manifest["questions_matcher"] == "vocab"
