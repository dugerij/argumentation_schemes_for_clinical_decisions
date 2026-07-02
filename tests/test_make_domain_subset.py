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
    assert manifest["notes_matcher"] == "keyword"
    assert manifest["questions_matcher"] == "keyword"


def test_make_domain_subset_can_refine_with_umls(tmp_path):
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
                "text": "Kidney disease note.",
            }
        )

    with questions_path.open("w", encoding="utf-8") as file:
        file.write(json.dumps({"question": "Kidney disease question?", "options": {"A": "Dialysis"}}) + "\n")

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
        "--min-domain-hits",
        "1",
        "--refine-notes-with-umls",
        "--refine-questions-with-umls",
    ]

    class _KeywordMatcher:
        def match_details(self, text: str):
            return (1, ["kidney"]) if "kidney" in text.lower() else (0, [])

    class _UMLSMatcher:
        def match_details(self, text: str):
            return (1, ["chronic kidney disease"]) if "kidney" in text.lower() else (0, [])

    def _fake_build_domain_matcher(_domain: str, matcher: str, *, prefilter_min_hits: int = 1):
        del prefilter_min_hits
        if matcher == "umls":
            return _UMLSMatcher()
        return _KeywordMatcher()

    with patch("sys.argv", argv), patch("make_domain_subset.build_domain_matcher", side_effect=_fake_build_domain_matcher):
        make_domain_subset.main()

    manifest = json.loads(questions_output_path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    selection = json.loads(questions_output_path.with_suffix(".selection.json").read_text(encoding="utf-8"))
    assert manifest["refine_notes_with_umls"] is True
    assert manifest["refine_questions_with_umls"] is True
    assert selection["notes"]["refined_with_umls"] is True
    assert selection["questions"]["refined_with_umls"] is True


def test_make_domain_subset_resumes_pending_umls_refinement(tmp_path):
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
                "text": "Chronic kidney disease with dialysis.",
            }
        )

    with questions_path.open("w", encoding="utf-8") as file:
        file.write(json.dumps({"question": "Kidney disease question?", "options": {"A": "Dialysis"}}) + "\n")

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
        "--min-domain-hits",
        "1",
        "--refine-notes-with-umls",
        "--refine-questions-with-umls",
    ]

    class _KeywordMatcher:
        def match_details(self, text: str):
            return (1, ["kidney"]) if "kidney" in text.lower() else (0, [])

    class _PendingUMLSMatcher:
        def __init__(self):
            class _Client:
                failed_request_count = 0

            class _Extractor:
                client = _Client()

            self.extractor = _Extractor()

        def match_details(self, text: str):
            self.extractor.client.failed_request_count += 1
            return 0, []

    class _GoodUMLSMatcher:
        def __init__(self):
            class _Client:
                failed_request_count = 0

            class _Extractor:
                client = _Client()

            self.extractor = _Extractor()

        def match_details(self, text: str):
            return (1, ["chronic kidney disease"]) if "kidney" in text.lower() else (0, [])

    def _first_build_domain_matcher(_domain: str, matcher: str, *, prefilter_min_hits: int = 1):
        del prefilter_min_hits
        if matcher == "umls":
            return _PendingUMLSMatcher()
        return _KeywordMatcher()

    def _second_build_domain_matcher(_domain: str, matcher: str, *, prefilter_min_hits: int = 1):
        del prefilter_min_hits
        if matcher == "umls":
            return _GoodUMLSMatcher()
        return _KeywordMatcher()

    with patch("sys.argv", argv), patch("make_domain_subset.build_domain_matcher", side_effect=_first_build_domain_matcher):
        make_domain_subset.main()

    first_manifest = json.loads(questions_output_path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert first_manifest["note_refinement_pending_count"] == 1
    assert first_manifest["question_refinement_pending_count"] == 1

    with patch("sys.argv", argv), patch("make_domain_subset.build_domain_matcher", side_effect=_second_build_domain_matcher):
        make_domain_subset.main()

    final_manifest = json.loads(questions_output_path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert final_manifest["resumed_notes_subset"] is True
    assert final_manifest["note_refinement_pending_count"] == 0
    assert final_manifest["question_refinement_pending_count"] == 0
