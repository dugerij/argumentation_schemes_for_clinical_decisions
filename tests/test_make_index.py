from argparse import Namespace
import os
from pathlib import Path
from unittest.mock import patch

import make_index
from helpers.paths import resolve_mimic_discharge_csv


def test_build_command_uses_non_schema_mode(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    args = Namespace(
        command="build",
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        limit="25",
        csv_path=None,
        max_chars="6000",
        note_type="DS",
    )

    with patch("make_index.load_dotenv"), patch("make_index.startup_check"), patch(
        "make_index.parse_args", return_value=args
    ), patch("make_index.ensure_index") as ensure_index:
        make_index.main()

    ensure_index.assert_called_once_with(
        input_dir=input_dir,
        output_dir=output_dir,
        use_umls=True,
        schema_guided=False,
    )


def test_build_schema_command_uses_schema_mode_even_if_env_default_differs(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    args = Namespace(
        command="build-schema",
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        limit="25",
        csv_path=None,
        max_chars="6000",
        note_type="DS",
    )

    with patch("make_index.load_dotenv"), patch("make_index.startup_check"), patch(
        "make_index.parse_args", return_value=args
    ), patch("make_index.ensure_index") as ensure_index, patch.dict(
        "os.environ",
        {
            "INPUT_BASE_DIR": str(Path("unused_input")),
            "OUTPUT_BASE_DIR": str(Path("unused_output")),
            "UMLS_ENABLED": "true",
            "INDEX_SCHEMA_GUIDED": "false",
        },
        clear=False,
    ):
        make_index.main()

    ensure_index.assert_called_once_with(
        input_dir=input_dir,
        output_dir=output_dir,
        use_umls=True,
        schema_guided=True,
    )

def test_extract_mimic_discharge_requires_existing_csv(tmp_path):
    input_dir = tmp_path / "input"
    csv_path = tmp_path / "mimic_iv_note" / "discharge.csv"
    args = Namespace(
        command="extract-mimic-discharge",
        input_dir=str(input_dir),
        output_dir=str(tmp_path / "output"),
        limit="25",
        csv_path=str(csv_path),
        max_chars="6000",
        note_type="DS",
    )

    with patch("make_index.load_dotenv"), patch("make_index.startup_check"), patch(
        "make_index.parse_args", return_value=args
    ):
        try:
            make_index.main()
        except FileNotFoundError as exc:
            assert str(csv_path) in str(exc)
        else:
            raise AssertionError("Expected FileNotFoundError")


def test_resolve_mimic_discharge_csv_finds_physionet_extract_layout(tmp_path):
    discharge_path = tmp_path / "data" / "mimic-iv-note-deidentified-free-text-clinical-notes-2.2" / "note" / "discharge.csv.gz"
    discharge_path.parent.mkdir(parents=True)
    discharge_path.write_bytes(b"placeholder")

    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        resolved = resolve_mimic_discharge_csv(None)
    finally:
        os.chdir(cwd)

    assert resolved == Path("data/mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note/discharge.csv.gz")


def test_resolve_mimic_discharge_csv_falls_back_when_explicit_legacy_path_is_missing(tmp_path):
    discharge_path = tmp_path / "data" / "mimic-iv-note-deidentified-free-text-clinical-notes-2.2" / "note" / "discharge.csv.gz"
    discharge_path.parent.mkdir(parents=True)
    discharge_path.write_bytes(b"placeholder")

    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        resolved = resolve_mimic_discharge_csv("data/mimic_iv_note/discharge.csv")
    finally:
        os.chdir(cwd)

    assert resolved == Path("data/mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note/discharge.csv.gz")
