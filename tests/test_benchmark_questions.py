import tempfile
import unittest
from pathlib import Path

from eval.question_subset import filter_questions_by_note_overlap


class BenchmarkQuestionFilterTests(unittest.TestCase):
    def test_filters_questions_to_note_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp)
            (input_dir / "case1.txt").write_text(
                "Patient with atrial fibrillation, hypertension, and chest pain received aspirin.",
                encoding="utf-8",
            )
            (input_dir / "case2.txt").write_text(
                "Myocardial infarction with troponin elevation and coronary disease.",
                encoding="utf-8",
            )

            questions = [
                {
                    "question": "What is the best next step for atrial fibrillation with hypertension?",
                    "options": {"A": "Aspirin", "B": "Insulin"},
                },
                {
                    "question": "Which antibiotic is best for cellulitis?",
                    "options": {"A": "Cephalexin", "B": "Vancomycin"},
                },
                {
                    "question": "Which finding is expected after myocardial infarction in coronary disease?",
                    "options": {"A": "Troponin elevation", "B": "Low glucose"},
                },
            ]

            retained, details = filter_questions_by_note_overlap(
                questions=questions,
                input_dir=input_dir,
                max_questions=2,
                min_overlap_terms=2,
            )

            self.assertEqual(len(retained), 2)
            retained_questions = [item["question"] for item in retained]
            self.assertIn("What is the best next step for atrial fibrillation with hypertension?", retained_questions)
            self.assertIn("Which finding is expected after myocardial infarction in coronary disease?", retained_questions)
            self.assertNotIn("Which antibiotic is best for cellulitis?", retained_questions)
            self.assertTrue(any(item["overlap_count"] == 0 for item in details))


if __name__ == "__main__":
    unittest.main()
