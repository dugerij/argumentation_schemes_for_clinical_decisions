from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import make_index


def test_build_command_uses_non_schema_mode(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    args = Namespace(
        command="build",
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        limit="25",
        csv_path=None,
        dataset_dir=None,
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
        dataset_dir=None,
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
