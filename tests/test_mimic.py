import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from ingest.mimic import (
    MimicDischargeSubsetConfig,
    MimicExtCardiovascularSubsetConfig,
    extract_mimic_discharge_subset,
    extract_mimic_ext_cardiovascular_subset,
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


class MimicExtCardiovascularExtractionTests(unittest.TestCase):
    def test_extracts_only_cardiovascular_icd_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "mimic_ext"
            dataset_dir.mkdir()
            output_dir = root / "cases"

            assessment_path = dataset_dir / "initial_assessment_info.csv"
            with assessment_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "stay_id",
                        "triage",
                        "pain",
                        "chiefcomplaint",
                        "arrival_transport",
                        "disposition",
                        "icd_code",
                        "icd_title",
                        "icd_version",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "stay_id": "1",
                        "triage": "2",
                        "pain": "5",
                        "chiefcomplaint": "CHEST PAIN",
                        "arrival_transport": "WALK IN",
                        "disposition": "ADMITTED",
                        "icd_code": "I214",
                        "icd_title": "NSTEMI",
                        "icd_version": "10",
                    }
                )
                writer.writerow(
                    {
                        "stay_id": "2",
                        "triage": "3",
                        "pain": "3",
                        "chiefcomplaint": "ABDOMINAL PAIN",
                        "arrival_transport": "WALK IN",
                        "disposition": "ADMITTED",
                        "icd_code": "K260",
                        "icd_title": "Duodenal ulcer",
                        "icd_version": "10",
                    }
                )
                writer.writerow(
                    {
                        "stay_id": "3",
                        "triage": "1",
                        "pain": "0",
                        "chiefcomplaint": "PALPITATIONS",
                        "arrival_transport": "AMBULANCE",
                        "disposition": "ADMITTED",
                        "icd_code": "42731",
                        "icd_title": "ATRIAL FIBRILLATION",
                        "icd_version": "9",
                    }
                )

            clinical_csv = root / "clinical_data.csv"
            with clinical_csv.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "stay_id",
                        "text",
                        "HPI",
                        "tests",
                        "past_medication",
                        "diagnosis",
                        "primary_diagnosis",
                        "secondary_diagnosis",
                    ],
                )
                writer.writeheader()
                for stay_id, diagnosis in [("1", "Chest pain"), ("2", "Ulcer"), ("3", "Atrial fibrillation")]:
                    writer.writerow(
                        {
                            "stay_id": stay_id,
                            "text": f"<br>Clinical narrative for {stay_id}<comma> {diagnosis}</br>",
                            "HPI": f"HPI for {stay_id}",
                            "tests": "Troponin",
                            "past_medication": "Aspirin",
                            "diagnosis": diagnosis,
                            "primary_diagnosis": f"['{diagnosis}']",
                            "secondary_diagnosis": "[]",
                        }
                    )
            with zipfile.ZipFile(dataset_dir / "clinical_data.csv.zip", "w") as archive:
                archive.write(clinical_csv, arcname="clinical_data.csv")

            written = extract_mimic_ext_cardiovascular_subset(
                MimicExtCardiovascularSubsetConfig(
                    dataset_dir=dataset_dir,
                    output_dir=output_dir,
                    limit=None,
                    max_chars=None,
                )
            )

            self.assertEqual(len(written), 2)
            names = sorted(path.name for path in written)
            self.assertEqual(names, ["1_I214.txt", "3_42731.txt"])
            content = written[0].read_text(encoding="utf-8")
            self.assertIn("MIMIC-IV-Ext cardiovascular case", content)
            self.assertIn("icd_title: NSTEMI", content)
            self.assertIn("Clinical narrative for 1, Chest pain", content)

            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["document_count"], 2)
            self.assertEqual(manifest["cardiovascular_case_count"], 2)


if __name__ == "__main__":
    unittest.main()
