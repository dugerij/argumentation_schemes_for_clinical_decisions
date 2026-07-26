from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Mapping

from clinical_cds.reporting import _matplotlib


SCHEME_LABELS = {
    "argument_from_clinical_sign": "Clinical sign",
    "argument_from_diagnostic_criterion": "Diagnostic criterion",
    "argument_from_risk_factor": "Risk factor",
    "argument_from_guideline_authority": "Guideline authority",
    "argument_from_negative_evidence": "Negative evidence",
    "argument_from_alternative_explanation": "Alternative explanation",
    "critical_question_challenge": "Critical question",
}
STATUS_COLORS = {
    "accepted": ("#DDECE5", "#2A7F62"),
    "rejected": ("#F1DFDC", "#9A463B"),
    "undecided": ("#F3E9CE", "#B57B21"),
    "unlabelled": ("#E8EBED", "#687078"),
}


def load_argument_trace(
    path: Path,
    *,
    case_id: str | None = None,
    trace_id: str | None = None,
) -> dict[str, Any]:
    if case_id and trace_id:
        raise ValueError("Select a trace by case_id or trace_id, not both.")
    first_trace: dict[str, Any] | None = None
    with Path(path).open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                continue
            if first_trace is None:
                first_trace = payload
            if case_id and str(payload.get("case_id") or "") == case_id:
                return payload
            if trace_id and str(payload.get("trace_id") or "") == trace_id:
                return payload
            if not case_id and not trace_id:
                return payload
    selector = f"case_id={case_id!r}" if case_id else f"trace_id={trace_id!r}"
    if case_id or trace_id:
        raise KeyError(f"No argument trace matched {selector}.")
    if first_trace is None:
        raise ValueError(f"No argument traces were found in {path}.")
    return first_trace


def _shorten(value: object, width: int) -> str:
    normalized = " ".join(str(value or "").split())
    return textwrap.shorten(
        normalized,
        width=width,
        placeholder="...",
    )


def _status(
    argument_id: str,
    accepted: set[str],
    rejected: set[str],
    undecided: set[str],
) -> str:
    if argument_id in accepted:
        return "accepted"
    if argument_id in rejected:
        return "rejected"
    if argument_id in undecided:
        return "undecided"
    return "unlabelled"


def plot_argument_trace(
    trace: Mapping[str, Any],
    output_path: Path,
) -> Path:
    proposal = trace.get("proposal")
    verifier = trace.get("verifier")
    graph = trace.get("argument_graph")
    resolution = trace.get("symbolic_resolution")
    if not all(isinstance(value, Mapping) for value in (
        proposal,
        verifier,
        graph,
        resolution,
    )):
        raise ValueError("A complete proposal, verifier, graph, and resolution are required.")

    candidates = [
        candidate
        for candidate in proposal.get("candidates", [])
        if isinstance(candidate, Mapping)
    ]
    if not candidates:
        raise ValueError("The argument trace contains no diagnosis candidates.")

    reviews = {
        str(review.get("argument_id") or ""): review
        for review in verifier.get("reviews", [])
        if isinstance(review, Mapping)
    }
    counterarguments = [
        counter
        for counter in verifier.get("counterarguments", [])
        if isinstance(counter, Mapping)
    ]
    graph_nodes = [
        node
        for node in graph.get("nodes", [])
        if isinstance(node, Mapping)
    ]
    accepted = {
        str(value) for value in resolution.get("accepted_argument_ids", [])
    }
    rejected = {
        str(value) for value in resolution.get("rejected_argument_ids", [])
    }
    undecided = {
        str(value) for value in resolution.get("undecided_argument_ids", [])
    }
    candidate_scores = {
        str(key): int(value)
        for key, value in dict(
            resolution.get("candidate_scores", {})
        ).items()
    }
    argument_to_candidate = {
        str(argument.get("argument_id") or ""): str(
            candidate.get("candidate_id") or ""
        )
        for candidate in candidates
        for argument in candidate.get("arguments", [])
        if isinstance(argument, Mapping)
    }

    plt = _matplotlib()
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    figure_height = 3.0 + 2.15 * len(candidates)
    figure, axis = plt.subplots(figsize=(15.0, figure_height))
    figure.patch.set_facecolor("#F5F2EA")
    axis.set_facecolor("#F5F2EA")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_axis_off()

    case_id = str(trace.get("case_id") or "unknown case")
    quality = dict(graph.get("quality", {}))
    quality_text = (
        f"Schema {float(quality.get('argument_schema_validity') or 0):.0%}"
        f"  ·  Evidence {float(quality.get('argument_evidence_validity') or 0):.0%}"
        f"  ·  Verifier {float(quality.get('verifier_review_coverage') or 0):.0%}"
    )
    axis.text(
        0.02,
        0.975,
        f"Argument trace · {case_id}",
        ha="left",
        va="top",
        fontsize=17,
        fontweight="bold",
        color="#202428",
    )
    axis.text(
        0.98,
        0.972,
        quality_text,
        ha="right",
        va="top",
        fontsize=9.5,
        color="#4B5560",
    )

    columns = {
        "reasoner": (0.025, 0.39),
        "verifier": (0.44, 0.265),
        "resolver": (0.75, 0.225),
    }
    for label, (left, width) in (
        ("1  Reasoner proposal", columns["reasoner"]),
        ("2  Verification and attacks", columns["verifier"]),
        ("3  Symbolic resolution", columns["resolver"]),
    ):
        axis.text(
            left,
            0.905,
            label,
            ha="left",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="#384047",
        )

    lane_top = 0.865
    lane_bottom = 0.145
    lane_gap = 0.018
    lane_height = (
        lane_top
        - lane_bottom
        - lane_gap * (len(candidates) - 1)
    ) / len(candidates)
    preferred = str(proposal.get("preferred_diagnosis") or "")

    for index, candidate in enumerate(candidates):
        candidate_id = str(candidate.get("candidate_id") or "")
        diagnosis = str(candidate.get("diagnosis") or "")
        top = lane_top - index * (lane_height + lane_gap)
        bottom = top - lane_height
        centre_y = (top + bottom) / 2

        lane = FancyBboxPatch(
            (0.012, bottom),
            0.976,
            lane_height,
            boxstyle="round,pad=0.004,rounding_size=0.008",
            facecolor="#FBFAF6" if index % 2 == 0 else "#F7F5EF",
            edgecolor="#D8D3C8",
            linewidth=0.8,
        )
        axis.add_patch(lane)

        reasoner_left, reasoner_width = columns["reasoner"]
        reasoner_box = FancyBboxPatch(
            (reasoner_left, bottom + 0.012),
            reasoner_width,
            lane_height - 0.024,
            boxstyle="round,pad=0.006,rounding_size=0.008",
            facecolor="#DDEAF2" if diagnosis == preferred else "#E9EFF2",
            edgecolor="#3478A5",
            linewidth=1.1,
        )
        axis.add_patch(reasoner_box)
        preferred_label = "  ·  preferred" if diagnosis == preferred else ""
        axis.text(
            reasoner_left + 0.012,
            top - 0.027,
            f"{candidate_id}  {_shorten(diagnosis, 44)}{preferred_label}",
            ha="left",
            va="top",
            fontsize=10.5,
            fontweight="bold",
            color="#203746",
        )
        argument_lines: list[str] = []
        arguments = [
            argument
            for argument in candidate.get("arguments", [])
            if isinstance(argument, Mapping)
        ]
        for argument in arguments:
            argument_id = str(argument.get("argument_id") or "")
            scheme = SCHEME_LABELS.get(
                str(argument.get("scheme") or ""),
                str(argument.get("scheme") or ""),
            )
            evidence = ", ".join(
                str(value) for value in argument.get("evidence_ids", [])
            )
            argument_lines.append(
                f"{argument_id} · {scheme} · [{evidence}]\n"
                f"{_shorten(argument.get('premise'), 70)}"
            )
        axis.text(
            reasoner_left + 0.012,
            top - 0.069,
            "\n\n".join(argument_lines) or "No valid arguments",
            ha="left",
            va="top",
            fontsize=8.2,
            linespacing=1.25,
            color="#27343C",
        )

        verifier_left, verifier_width = columns["verifier"]
        verifier_box = FancyBboxPatch(
            (verifier_left, bottom + 0.012),
            verifier_width,
            lane_height - 0.024,
            boxstyle="round,pad=0.006,rounding_size=0.008",
            facecolor="#F2EFE7",
            edgecolor="#8C8474",
            linewidth=1.0,
        )
        axis.add_patch(verifier_box)
        verification_lines: list[str] = []
        for argument in arguments:
            argument_id = str(argument.get("argument_id") or "")
            review = reviews.get(argument_id, {})
            verdict = str(review.get("verdict") or "not reviewed")
            final_status = _status(
                argument_id,
                accepted,
                rejected,
                undecided,
            )
            verification_lines.append(
                f"{argument_id}  {verdict}  →  {final_status}"
            )
        candidate_counters = [
            counter
            for counter in counterarguments
            if (
                str(counter.get("target_argument_id") or "") == candidate_id
                or argument_to_candidate.get(
                    str(counter.get("target_argument_id") or "")
                )
                == candidate_id
            )
        ]
        candidate_challenges = [
            node
            for node in graph_nodes
            if node.get("node_type") == "challenge"
            and str(node.get("candidate_id") or "") == candidate_id
        ]
        accepted_attacks = sum(
            str(node.get("argument_id") or "") in accepted
            for node in [*candidate_counters, *candidate_challenges]
        )
        verification_lines.append(
            f"Attacks: {len(candidate_counters) + len(candidate_challenges)}"
            f"  ·  accepted: {accepted_attacks}"
        )
        axis.text(
            verifier_left + 0.012,
            top - 0.027,
            "\n\n".join(verification_lines),
            ha="left",
            va="top",
            fontsize=8.5,
            linespacing=1.25,
            color="#343A3F",
        )

        resolver_left, resolver_width = columns["resolver"]
        candidate_status = _status(
            candidate_id,
            accepted,
            rejected,
            undecided,
        )
        fill_color, edge_color = STATUS_COLORS[candidate_status]
        resolver_box = FancyBboxPatch(
            (resolver_left, bottom + 0.012),
            resolver_width,
            lane_height - 0.024,
            boxstyle="round,pad=0.006,rounding_size=0.008",
            facecolor=fill_color,
            edgecolor=edge_color,
            linewidth=1.2,
        )
        axis.add_patch(resolver_box)
        accepted_support = [
            str(argument.get("argument_id") or "")
            for argument in arguments
            if str(argument.get("argument_id") or "") in accepted
        ]
        axis.text(
            resolver_left + 0.012,
            top - 0.027,
            candidate_status.upper(),
            ha="left",
            va="top",
            fontsize=10.5,
            fontweight="bold",
            color=edge_color,
        )
        axis.text(
            resolver_left + 0.012,
            top - 0.071,
            (
                f"Scheme score: {candidate_scores.get(candidate_id, 0)}\n"
                f"Accepted support: "
                f"{', '.join(accepted_support) or 'none'}"
            ),
            ha="left",
            va="top",
            fontsize=8.8,
            linespacing=1.45,
            color="#343A3F",
        )

        for start, end in (
            (
                reasoner_left + reasoner_width + 0.006,
                verifier_left - 0.008,
            ),
            (
                verifier_left + verifier_width + 0.006,
                resolver_left - 0.008,
            ),
        ):
            arrow = FancyArrowPatch(
                (start, centre_y),
                (end, centre_y),
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.1,
                color="#7B858C",
            )
            axis.add_patch(arrow)

    selected = str(resolution.get("selected_diagnosis") or "")
    abstained = bool(resolution.get("abstained"))
    outcome_fill, outcome_edge = (
        STATUS_COLORS["undecided"]
        if abstained
        else STATUS_COLORS["accepted"]
    )
    outcome_box = FancyBboxPatch(
        (0.18, 0.035),
        0.64,
        0.072,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=outcome_fill,
        edgecolor=outcome_edge,
        linewidth=1.3,
    )
    axis.add_patch(outcome_box)
    outcome = "ABSTAINED" if abstained else f"SELECTED  ·  {selected}"
    changed = bool(preferred and preferred != selected)
    axis.text(
        0.5,
        0.077,
        outcome,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=outcome_edge,
    )
    axis.text(
        0.5,
        0.049,
        (
            f"Reasoner preferred: {preferred or 'none'}"
            f"  ·  Resolution changed: {'yes' if changed else 'no'}"
        ),
        ha="center",
        va="center",
        fontsize=8.8,
        color="#4B5560",
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path
