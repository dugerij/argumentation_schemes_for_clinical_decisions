import csv
import tempfile
import unittest
from pathlib import Path

from ingest.mimic import (
    MimicDischargeSubsetConfig,
    extract_mimic_discharge_subset,
)


class MimicDischargeExtractionTests(unittest.TestCase):
    def test_none_limit_extracts_all_matching_notes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "discharge.csv"
            output_dir = root / "notes"

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
                for index in range(3):
                    writer.writerow(
                        {
                            "note_id": f"n{index}",
                            "subject_id": f"s{index}",
                            "hadm_id": f"h{index}",
                            "note_type": "DS",
                            "note_seq": index,
                            "charttime": "",
                            "storetime": "",
                            "text": f"note {index}",
                        }
                    )

            written = extract_mimic_discharge_subset(
                MimicDischargeSubsetConfig(csv_path=csv_path, output_dir=output_dir, limit=None)
            )

            self.assertEqual(len(written), 3)

    def test_none_max_chars_keeps_full_note_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "discharge.csv"
            output_dir = root / "notes"
            text = "alpha " * 20

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
                        "text": text,
                    }
                )

            written = extract_mimic_discharge_subset(
                MimicDischargeSubsetConfig(csv_path=csv_path, output_dir=output_dir, limit=None, max_chars=None)
            )

            content = written[0].read_text(encoding="utf-8")
            self.assertIn(text, content)
            self.assertIn("truncated: false", content)
if __name__ == "__main__":
    unittest.main()
