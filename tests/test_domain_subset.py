import csv
import json
from pathlib import Path

from eval.question_subset import filter_questions_for_domain_and_notes
from helpers.clinical_domains import HybridDomainMatcher
from ingest.mimic import MimicDischargeDomainSubsetConfig, extract_mimic_discharge_domain_subset


def test_extract_mimic_discharge_domain_subset_keeps_only_matching_notes(tmp_path):
    csv_path = tmp_path / "discharge.csv"
    output_dir = tmp_path / "notes"

    with csv_path.open("w", encoding="utf-8", newline="") as file:
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
                "text": "Patient with chronic kidney disease, hyperkalemia, and dialysis history.",
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
                "text": "Patient with isolated wrist fracture and no renal issues.",
            }
        )

    written = extract_mimic_discharge_domain_subset(
        MimicDischargeDomainSubsetConfig(
            csv_path=csv_path,
            output_dir=output_dir,
            domain="renal_metabolic",
            limit=None,
        )
    )

    assert len(written) == 1
    assert written[0].name == "s1_h1_n1.txt"
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["document_count"] == 1


def test_filter_questions_for_domain_and_notes_keeps_all_relevant_questions(tmp_path):
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    (notes_dir / "renal.txt").write_text(
        "chronic kidney disease dialysis hyperkalemia creatinine renal protection",
        encoding="utf-8",
    )

    questions = [
        {
            "question": "What is the best next step for chronic kidney disease with hyperkalemia?",
            "options": {"A": "Dialysis", "B": "Splinting"},
        },
        {
            "question": "What is the best treatment for ankle sprain?",
            "options": {"A": "Rest", "B": "Dialysis"},
        },
        {
            "question": "Which measure improves renal protection in chronic kidney disease?",
            "options": {"A": "Blood pressure control", "B": "Casting"},
        },
    ]

    result = filter_questions_for_domain_and_notes(
        questions=questions,
        domain="renal_metabolic",
        notes_dir=notes_dir,
        min_overlap_terms=2,
        limit=None,
    )

    kept_questions = [item["question"] for item in result.kept_questions]
    assert kept_questions == [
        "What is the best next step for chronic kidney disease with hyperkalemia?",
        "Which measure improves renal protection in chronic kidney disease?",
    ]
    included = [row for row in result.metadata if row["included"]]
    assert len(included) == 2


def test_hybrid_domain_matcher_skips_umls_when_keyword_prefilter_misses():
    class _StubMatcher:
        def __init__(self) -> None:
            self.calls = 0

        def match_details(self, text: str):
            self.calls += 1
            return 1, ["ckd"]

    stub = _StubMatcher()
    matcher = HybridDomainMatcher("renal_metabolic", stub, prefilter_min_hits=1)

    misses = matcher.match_details("Ankle fracture after a fall.")
    hits = matcher.match_details("Chronic kidney disease with dialysis planning.")

    assert misses == (0, [])
    assert hits == (1, ["ckd"])
    assert stub.calls == 1
