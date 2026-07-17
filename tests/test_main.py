from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import main


def test_materialize_graph_command_uses_requested_input_and_output(tmp_path, capsys):
    input_path = tmp_path / "cases.zip"
    output_dir = tmp_path / "output"
    args = Namespace(
        command="materialize-graph",
        input_path=str(input_path),
        dataset_format="mimic_ext_cds",
        output_dir=str(output_dir),
        no_overwrite=False,
    )

    fake_stats = type(
        "Stats",
        (),
        {
            "artifact_path": str(output_dir / "cds_case_graph.pkl.gz"),
            "manifest_path": str(output_dir / "cds_case_graph_manifest.json"),
            "case_count": 10,
            "diagnosis_case_count": 9,
            "triage_case_count": 10,
            "specialty_case_count": 4,
            "token_count": 99,
            "build_seconds": 1.25,
            "artifact_bytes": 2048,
        },
    )()

    with patch("main.load_dotenv"), patch("main.startup_check"), patch(
        "main.parse_args", return_value=args
    ), patch("main.build_cds_materialized_graph", return_value=fake_stats) as build_graph:
        main.main()

    build_graph.assert_called_once_with(
        zip_path=Path(input_path),
        output_dir=Path(output_dir),
        overwrite=True,
    )
    assert "cds_case_graph.pkl.gz" in capsys.readouterr().out


def test_answer_question_command_passes_case_request(tmp_path):
    output_dir = tmp_path / "output"
    args = Namespace(
        command="answer-question",
        task="diagnosis",
        case_id=101,
        output_dir=str(output_dir),
        question=None,
        clinical_goal=None,
        max_rounds=2,
        top_k_cases=3,
        dry_run=True,
    )

    fake_response = type("Response", (), {"model_dump_json": lambda self, indent=2: '{"ok": true}'})()

    with patch("main.load_dotenv"), patch("main.startup_check"), patch(
        "main.parse_args", return_value=args
    ), patch("main.generate_recommendation", return_value=fake_response) as generate_recommendation:
        main.main()

    request = generate_recommendation.call_args.args[0]
    assert request.case_id == 101
    assert request.task == "diagnosis"
    assert request.top_k_cases == 3
