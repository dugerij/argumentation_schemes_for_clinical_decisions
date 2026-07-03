import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from helpers.config import env_bool, env_optional_int, parse_optional_int, startup_check
from helpers.paths import resolve_mimic_discharge_csv
from retrieval.index import ensure_index


BUILD_COMMANDS = {"build", "build-schema"}


def parse_args() -> argparse.Namespace:
    """Parse the command surface used after the source CSV has been downloaded."""

    parser = argparse.ArgumentParser(description="Indexing and evidence ingestion utilities.")
    parser.add_argument(
        "command",
        choices=(
            "build",
            "build-schema",
            "extract-mimic-discharge",
        ),
        help="Build the index, build the schema-guided index, or extract a MIMIC discharge subset.",
    )
    parser.add_argument("--input-dir", default=None, help="Directory containing source .txt files.")
    parser.add_argument("--output-dir", default=None, help="Directory used to persist the index.")
    parser.add_argument("--limit", default="all", help="Maximum number of MIMIC notes to extract, or 'all'.")
    parser.add_argument("--csv-path", default=None, help="MIMIC discharge CSV to subset.")
    parser.add_argument("--max-chars", default="all", help="Maximum characters per extracted note, or 'all'.")
    parser.add_argument("--note-type", default="DS", help="MIMIC note type to keep.")
    return parser.parse_args()


def main() -> None:
    """Run the main notes-first CLI.

    `extract-mimic-discharge` turns the source CSV into one note-per-file
    evidence documents. `build` and `build-schema` operate on those extracted
    text files and therefore remain the only commands that require model setup.
    """

    load_dotenv()
    args = parse_args()
    startup_check(require_models=(args.command in BUILD_COMMANDS), require_paths=(args.command in BUILD_COMMANDS))

    input_dir = Path(args.input_dir or os.environ.get("INPUT_BASE_DIR", "data/evidence/mimic_discharge_subset"))
    output_dir = Path(args.output_dir or os.environ.get("OUTPUT_BASE_DIR", "output"))
    csv_path = resolve_mimic_discharge_csv(args.csv_path)

    if args.command == "extract-mimic-discharge":
        from ingest.mimic import MimicDischargeSubsetConfig, extract_mimic_discharge_subset

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing MIMIC discharge CSV: {csv_path}. "
                "Download the MIMIC-IV-Note discharge file from PhysioNet and place it there first."
            )

        manifest = extract_mimic_discharge_subset(
            MimicDischargeSubsetConfig(
                csv_path=csv_path,
                output_dir=input_dir,
                limit=env_optional_int("MIMIC_DISCHARGE_LIMIT", parse_optional_int(args.limit)),
                note_type=os.environ.get("MIMIC_DISCHARGE_NOTE_TYPE", args.note_type),
                max_chars=env_optional_int("MIMIC_DISCHARGE_MAX_CHARS", parse_optional_int(args.max_chars, default=None)),
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
