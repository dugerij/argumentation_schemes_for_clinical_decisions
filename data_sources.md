# Local Data Sources

Do not commit clinical data, MIMIC-IV notes, or generated index outputs.

For note-backed indexing, place extracted MIMIC discharge note `.txt` files under:

```text
data/evidence/mimic_discharge_subset/
```

The `data/` directory is ignored by git. Keep only source code, configuration, and documentation in the repository.

Recommended workflow:

1. Run `python make_index.py extract-mimic-discharge --csv-path data/mimic_iv_note/discharge.csv --limit 25 --max-chars 6000`.
2. Run `python make_index.py build`.
