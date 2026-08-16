from pathlib import Path

from clinical_cds.reporting import _matplotlib_config_dir, write_report_plots


def _summary_row(mode: str, **overrides) -> dict:
    row = {
        "dataset": "direct",
        "subset": "all",
        "mode": mode,
        "n": 12,
        "accuracy": 0.5,
        "hierarchy_score": 0.5,
        "coverage": 1.0,
        "observation_f1": 0.5,
        "argument_schema_validity": None,
        "argument_evidence_validity": None,
        "verifier_review_coverage": None,
        "symbolic_trace_fidelity": None,
    }
    row.update(overrides)
    return row


def test_argument_quality_plot_reads_the_live_argumentation_mode(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))

    # No row matches the argumentation mode: every panel renders the
    # "Not available" placeholder.
    empty_paths = write_report_plots(
        [_summary_row("direct")], [], tmp_path / "empty"
    )

    # A row carries real argument-quality metrics under the mode name the
    # live pipeline actually emits (evidence_grounded_argumentation).
    # Before the fix, reporting.py filtered for the retired
    # "structured_argument"/"symbolic_argument" mode names, so this row
    # would never match and the chart would render identically to the
    # empty case above regardless of the data supplied here.
    populated_paths = write_report_plots(
        [
            _summary_row("direct"),
            _summary_row(
                "evidence_grounded_argumentation",
                argument_schema_validity=0.9,
                argument_evidence_validity=0.8,
                verifier_review_coverage=0.7,
                symbolic_trace_fidelity=0.6,
            ),
        ],
        [],
        tmp_path / "populated",
    )

    empty_size = empty_paths["argument_quality_plot"].stat().st_size
    populated_size = populated_paths["argument_quality_plot"].stat().st_size
    assert populated_size > 0
    assert populated_size != empty_size


def test_matplotlib_config_defaults_to_output_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "writable-output"
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.setenv("OUTPUT_ROOT", str(output_root))

    assert _matplotlib_config_dir() == output_root / ".matplotlib"


def test_explicit_matplotlib_config_takes_precedence(
    monkeypatch,
    tmp_path: Path,
) -> None:
    explicit_config = tmp_path / "explicit-config"
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "ignored-output"))
    monkeypatch.setenv("MPLCONFIGDIR", str(explicit_config))

    assert _matplotlib_config_dir() == explicit_config
