from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any


MODE_ORDER = [
    "direct",
    "flat_rag",
    "graph_rag",
    "evidence_grounded_argumentation",
    "structured_argument",
    "symbolic_argument",
]
MODE_LABELS = {
    "direct": "Direct",
    "flat_rag": "Flat RAG",
    "graph_rag": "Graph RAG",
    "evidence_grounded_argumentation": "Evidence-Grounded LLM Argumentation",
    "structured_argument": "Structured argument",
    "symbolic_argument": "Symbolic argument",
}
MODE_COLORS = {
    "direct": "#687078",
    "flat_rag": "#3478A5",
    "graph_rag": "#D17A22",
    "evidence_grounded_argumentation": "#2A7F62",
    "structured_argument": "#2A7F62",
    "symbolic_argument": "#8B3F36",
}


def _matplotlib_config_dir() -> Path:
    configured = os.environ.get("MPLCONFIGDIR")
    if configured:
        return Path(configured).expanduser().resolve()
    output_root = Path(os.environ.get("OUTPUT_ROOT", "output"))
    return (output_root / ".matplotlib").expanduser().resolve()


def _matplotlib():
    config_dir = _matplotlib_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(config_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def write_report_plots(
    summary_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = _matplotlib()
    from matplotlib.ticker import PercentFormatter

    primary_rows = [
        row
        for row in summary_rows
        if row["subset"] == "all" and row["mode"] in MODE_ORDER
    ]
    primary_rows.sort(
        key=lambda row: (
            str(row["dataset"]),
            MODE_ORDER.index(str(row["mode"])),
        )
    )
    datasets = sorted({str(row["dataset"]) for row in primary_rows})
    multiple_datasets = len(datasets) > 1

    def row_label(row: dict[str, Any]) -> str:
        mode_label = MODE_LABELS.get(str(row["mode"]), str(row["mode"]))
        return (
            f"{row['dataset']} · {mode_label}"
            if multiple_datasets
            else mode_label
        )

    def draw_metric(
        axis: Any,
        rows: list[dict[str, Any]],
        metric: str,
        title: str,
    ) -> None:
        observed = [row for row in rows if row.get(metric) is not None]
        axis.set_title(title, fontsize=11, fontweight="bold", loc="left")
        if not observed:
            axis.text(
                0.5,
                0.5,
                "Not available",
                ha="center",
                va="center",
                color="#687078",
                transform=axis.transAxes,
            )
            axis.set_axis_off()
            return
        values = [float(row[metric]) for row in observed]
        positions = list(range(len(observed)))
        bars = axis.barh(
            positions,
            values,
            height=0.62,
            color=[MODE_COLORS[str(row["mode"])] for row in observed],
            edgecolor="#202428",
            linewidth=0.4,
        )
        axis.set_yticks(positions)
        axis.set_yticklabels([row_label(row) for row in observed])
        axis.invert_yaxis()
        axis.set_xlim(0.0, 1.0)
        axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        axis.grid(axis="x", color="#D8D8D8", linewidth=0.6)
        axis.set_axisbelow(True)
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                min(value + 0.018, 0.98),
                bar.get_y() + bar.get_height() / 2,
                f"{value:.0%}",
                ha="right" if value > 0.9 else "left",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

    sample_sizes = sorted(
        {
            int(row.get("n") or 0)
            for row in primary_rows
            if int(row.get("n") or 0) > 0
        }
    )
    sample_note = (
        f"n={sample_sizes[0]}; one case equals "
        f"{100 / sample_sizes[0]:.1f} percentage points."
        if len(sample_sizes) == 1
        else "Sample size varies by condition."
    )
    clinical_metrics = [
        ("accuracy", "Exact diagnosis accuracy"),
        ("hierarchy_score", "Hierarchy-aware score"),
        ("coverage", "Answer coverage"),
        ("observation_f1", "Observation grounding F1"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
    for axis, (metric, title) in zip(
        axes.flat,
        clinical_metrics,
        strict=True,
    ):
        draw_metric(axis, primary_rows, metric, title)
    figure.suptitle(
        "Clinical method comparison profile",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.015,
        sample_note,
        ha="center",
        fontsize=9,
        color="#4B5560",
    )
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.95))
    metrics_path = output_dir / "method_comparison_metrics.png"
    figure.savefig(metrics_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    argument_metrics = [
        ("argument_schema_validity", "Schema validity"),
        ("argument_evidence_validity", "Diagnosis-aligned evidence"),
        ("verifier_review_coverage", "Verifier coverage"),
        ("symbolic_trace_fidelity", "Decision-trace consistency"),
    ]
    argument_rows = [
        row
        for row in primary_rows
        if row["mode"] == "evidence_grounded_argumentation"
    ]
    argument_quality_path = output_dir / "argument_quality.png"
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.0))
    for axis, (metric, title) in zip(
        axes.flat,
        argument_metrics,
        strict=True,
    ):
        draw_metric(axis, argument_rows, metric, title)
    figure.suptitle(
        "Argument and explanation quality",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    figure.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
    figure.savefig(
        argument_quality_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)

    def wilson_interval(
        proportion: float,
        sample_size: int,
    ) -> tuple[float, float]:
        if sample_size <= 0:
            return proportion, proportion
        z = 1.959963984540054
        denominator = 1.0 + z**2 / sample_size
        centre = (
            proportion + z**2 / (2.0 * sample_size)
        ) / denominator
        margin = (
            z
            * math.sqrt(
                proportion * (1.0 - proportion) / sample_size
                + z**2 / (4.0 * sample_size**2)
            )
            / denominator
        )
        return max(0.0, centre - margin), min(1.0, centre + margin)

    progression_path = output_dir / "accuracy_progression.png"
    if datasets:
        figure, axes = plt.subplots(
            len(datasets),
            1,
            figsize=(10.5, max(5.2, 4.3 * len(datasets))),
            squeeze=False,
        )
        for axis, dataset in zip(axes.flat, datasets, strict=True):
            rows = [
                row
                for row in primary_rows
                if str(row["dataset"]) == dataset
                and row.get("accuracy") is not None
            ]
            values = [float(row["accuracy"]) for row in rows]
            positions = list(range(len(rows)))
            intervals = [
                wilson_interval(value, int(row.get("n") or 0))
                for value, row in zip(values, rows, strict=True)
            ]
            lower = [
                value - interval[0]
                for value, interval in zip(values, intervals, strict=True)
            ]
            upper = [
                interval[1] - value
                for value, interval in zip(values, intervals, strict=True)
            ]
            axis.plot(
                positions,
                values,
                color="#384047",
                linewidth=1.8,
                zorder=2,
            )
            axis.errorbar(
                positions,
                values,
                yerr=[lower, upper],
                fmt="none",
                ecolor="#8C959D",
                elinewidth=1.2,
                capsize=4,
                zorder=1,
            )
            axis.scatter(
                positions,
                values,
                s=85,
                color=[MODE_COLORS[str(row["mode"])] for row in rows],
                edgecolor="#202428",
                linewidth=0.7,
                zorder=3,
            )
            for position, value in zip(positions, values, strict=True):
                axis.annotate(
                    f"{value:.0%}",
                    (position, value),
                    xytext=(0, 11),
                    textcoords="offset points",
                    ha="center",
                    fontweight="bold",
                )
            for index in range(1, len(values)):
                difference = (values[index] - values[index - 1]) * 100
                difference_label = (
                    "0 pp"
                    if abs(difference) < 0.05
                    else f"{difference:+.0f} pp"
                )
                axis.text(
                    index - 0.5,
                    min(
                        0.88,
                        max(values[index - 1], values[index]) + 0.13,
                    ),
                    difference_label,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#384047",
                    bbox={
                        "boxstyle": "round,pad=0.25",
                        "facecolor": "#F2EFE7",
                        "edgecolor": "none",
                    },
                )
            axis.set_xticks(positions)
            axis.set_xticklabels(
                [
                    MODE_LABELS.get(str(row["mode"]), str(row["mode"]))
                    for row in rows
                ]
            )
            axis.set_ylim(0.0, 1.0)
            axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            axis.set_ylabel("Exact-match accuracy")
            axis.grid(axis="y", color="#D8D8D8", linewidth=0.6)
            axis.set_axisbelow(True)
            axis.set_title(
                (
                    f"{dataset}: accuracy across the method comparison"
                    if multiple_datasets
                    else "Accuracy across the method comparison"
                ),
                fontsize=13,
                fontweight="bold",
                loc="left",
            )
        figure.text(
            0.5,
            0.015,
            (
                "Adjacent labels show change from the preceding condition; "
                "whiskers are 95% Wilson intervals. "
                + sample_note
            ),
            ha="center",
            fontsize=9,
            color="#4B5560",
        )
        figure.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
        figure.savefig(
            progression_path,
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(figure)
    else:
        figure, axis = plt.subplots(figsize=(8.5, 3.2))
        axis.text(
            0.5,
            0.5,
            "No accuracy summaries were available.",
            ha="center",
            va="center",
        )
        axis.set_axis_off()
        figure.savefig(
            progression_path,
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(figure)
    return {
        "method_comparison_metrics_plot": metrics_path,
        "argument_quality_plot": argument_quality_path,
        "accuracy_progression_plot": progression_path,
    }
