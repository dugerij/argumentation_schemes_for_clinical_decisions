import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from helpers.config import env_bool, env_optional_int, parse_optional_int, startup_check
from retrieval.index import ensure_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Indexing and evidence ingestion utilities.")
    parser.add_argument(
        "command",
        choices=("build", "build-schema", "extract-mimic-discharge"),
        help="Build the index, build the schema-guided index, or extract a MIMIC discharge subset.",
    )
    parser.add_argument("--input-dir", default=None, help="Directory containing source .txt files.")
    parser.add_argument("--output-dir", default=None, help="Directory used to persist the index.")
    parser.add_argument("--limit", default="25", help="Maximum number of MIMIC notes to extract, or 'all'.")
    parser.add_argument("--csv-path", default=None, help="MIMIC discharge CSV to subset.")
    parser.add_argument("--max-chars", default="6000", help="Maximum characters per extracted note, or 'all'.")
    parser.add_argument("--note-type", default="DS", help="MIMIC note type to keep.")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    startup_check()
    args = parse_args()

    input_dir = Path(args.input_dir or os.environ.get("INPUT_BASE_DIR", "data/evidence/mimic_discharge_subset"))
    output_dir = Path(args.output_dir or os.environ.get("OUTPUT_BASE_DIR", "output"))

    if args.command == "extract-mimic-discharge":
        from ingest.mimic import MimicDischargeSubsetConfig, extract_mimic_discharge_subset

        csv_path = Path(args.csv_path or os.environ.get("MIMIC_DISCHARGE_CSV", "data/mimic_iv_note/discharge.csv"))
        manifest = extract_mimic_discharge_subset(
            MimicDischargeSubsetConfig(
                csv_path=csv_path,
                output_dir=input_dir,
                limit=env_optional_int("MIMIC_DISCHARGE_LIMIT", parse_optional_int(args.limit, default=25)),
                note_type=os.environ.get("MIMIC_DISCHARGE_NOTE_TYPE", args.note_type),
                max_chars=env_optional_int("MIMIC_DISCHARGE_MAX_CHARS", parse_optional_int(args.max_chars, default=6000)),
                overwrite=True,
            )
        )
        print(f"Extracted {len(manifest)} MIMIC discharge notes into {input_dir}")
        return

    ensure_index(
        input_dir=input_dir,
        output_dir=output_dir,
        use_umls=env_bool("UMLS_ENABLED", True),
        schema_guided=(args.command == "build-schema"),
    )


if __name__ == "__main__":
    main()
