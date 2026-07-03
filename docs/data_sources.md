# Local Data Sources

Do not commit clinical data, MIMIC-IV notes, or generated index outputs.

For note-backed indexing, place extracted MIMIC discharge note `.txt` files under:

```text
data/evidence/mimic_discharge_subset/
```

The `data/` directory is ignored by git. Keep only source code, configuration, and documentation in the repository.

Recommended workflow:

1. Download `MIMIC-IV-Note` from PhysioNet and ensure the discharge-note table is available locally, either as `data/mimic_iv_note/discharge.csv` or in the extracted PhysioNet folder at `data/mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note/discharge.csv.gz`.
2. Run `python make_index.py extract-mimic-discharge --limit 25 --max-chars all`.
3. Run `python make_index.py build`.
