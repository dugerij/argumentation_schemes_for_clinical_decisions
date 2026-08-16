from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from clinical_cds.direct import label_key
from clinical_cds.io import load_jsonl


MODE = "evidence_grounded_argumentation"
GRAPH_MODE = "graph_rag"
_VALID_ID_RE = re.compile(r"^[SK][A-Za-z0-9_-]+$")


def _safe_str(value: object, default: str = "") -> str:
    return default if value is None else str(value)


def _extract_identifiers(values: Iterable[object]) -> list[str]:
    identifiers: list[str] = []
    seen: set[str] = set()
    for value in values:
        for piece in str(value).split(","):
            text = " ".join(piece.replace("\r", " ").replace("\n", " ").split()).strip()
            text = text.strip(",")
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            identifiers.append(text)
    return identifiers


def _evidence_namespace(id_value: str) -> str:
    if not id_value or len(id_value) < 2:
        return "empty_or_short"
    if id_value.startswith("S-"):
        return "section_or_patient"
    if id_value.startswith("K"):
        return "knowledge"
    if id_value.startswith("KFA-"):
        return "knowledge_atomized"
    return "other"


def _find_graph_candidates(trace: dict[str, Any], graph_label: str) -> list[str]:
    if not graph_label:
        return []
    target = label_key(graph_label)
    matches: list[str] = []
    for candidate in trace.get("candidate_inventory", ()):
        candidate_id = _safe_str(candidate.get("candidate_id"))
        if not candidate_id:
            continue
        if label_key(_safe_str(candidate.get("diagnosis"))) == target:
            matches.append(candidate_id)
            continue
        for path in candidate.get("diagnostic_paths", ()) or ():
            if isinstance(path, str):
                if label_key(path) == target:
                    matches.append(candidate_id)
                    break
            else:
                for node in path:
                    if label_key(_safe_str(node)) == target:
                        matches.append(candidate_id)
                        break
                else:
                    continue
                break
    return matches


def _candidate_map(trace: dict[str, Any]) -> dict[str, str]:
    return {
        _safe_str(item.get("candidate_id")): _safe_str(item.get("diagnosis"))
        for item in trace.get("candidate_inventory", ())
        if item.get("candidate_id")
    }


def _graph_shape_signal(
    trace: dict[str, Any],
    graph_label: str,
    graph_citations: list[str],
    active_ids: tuple[str, ...],
) -> dict[str, Any]:
    candidate_inventory = trace.get("candidate_inventory", ())
    candidate_by_id = {
        _safe_str(entry.get("candidate_id")): _safe_str(entry.get("diagnosis"))
        for entry in candidate_inventory
    }
    graph_citations_set = set(graph_citations)
    graph_candidates = _find_graph_candidates(trace, graph_label)
    active_set = set(active_ids)
    active_graph_candidates = [cid for cid in graph_candidates if cid in active_set]
    graph_candidates_sorted = sorted(set(graph_candidates))

    active_candidate_evidence = {
        _safe_str(entry.get("candidate_id")): set(_extract_identifiers(entry.get("evidence_ids", ()) or ()))
        for entry in candidate_inventory
        if _safe_str(entry.get("candidate_id")) in active_set
    }
    all_candidate_evidence = {
        _safe_str(entry.get("candidate_id")): set(_extract_identifiers(entry.get("evidence_ids", ()) or ()))
        for entry in candidate_inventory
        if _safe_str(entry.get("candidate_id"))
    }
    overlap_by_active_candidate = {
        candidate_id: sorted(active_candidate_evidence.get(candidate_id, set()).intersection(graph_citations))
        for candidate_id in sorted(active_candidate_evidence)
    }
    overlap_by_all_graph_candidate = {
        candidate_id: sorted(all_candidate_evidence.get(candidate_id, set()).intersection(graph_citations))
        for candidate_id in graph_candidates
    }
    non_active_graph_candidates = [
        cid for cid in graph_candidates if cid not in active_set
    ]
    namespace_counts = Counter(_evidence_namespace(value) for value in graph_citations)

    return {
        "graph_candidates": graph_candidates_sorted,
        "active_graph_candidates": active_graph_candidates,
        "non_active_graph_candidates": non_active_graph_candidates,
        "active_count": len(active_set),
        "active_candidate_evidence_overlap": overlap_by_active_candidate,
        "graph_candidate_evidence_overlap": overlap_by_all_graph_candidate,
        "graph_citation_namespaces": dict(sorted(namespace_counts.items())),
        "graph_candidate_diagnoses": {
            candidate_id: candidate_by_id.get(candidate_id, "")
            for candidate_id in graph_candidates_sorted
        },
    }


def _classify_case(
    *,
    trace: dict[str, Any],
    prediction: dict[str, Any],
    graph_row: dict[str, Any] | None,
) -> tuple[str, list[str], dict[str, Any]]:
    metadata = prediction.get("metadata", {})
    if not bool(prediction.get("abstained")):
        return "resolved", ["No abstention."], {}

    if _safe_str(prediction.get("error")) or _safe_str(trace.get("error")):
        return "execution_failure", [
            "Stage error prevented a complete argument trace."
        ], {
            "error": _safe_str(trace.get("error") or prediction.get("error")),
        }

    abstention_category = _safe_str(metadata.get("abstention_category"))
    if abstention_category != "insufficient_evidence":
        return abstention_category or "other_abstention", [
            "Abstention category was not marked as insufficient evidence.",
        ], {"abstention_category": abstention_category}

    dialectical = trace.get("dialectical_trace") or {}
    resolution = trace.get("dialectical_resolution") or {}
    direct = dialectical.get("direct_differential") or {}
    attack = dialectical.get("attack") or {}
    attack_validation = dialectical.get("attack_validation") or {}
    protection = dialectical.get("protected_graph_incumbent")

    direct_decision = _safe_str(direct.get("decision"))
    active_ids = tuple(
        dialectical.get("active_candidate_ids")
        or dialectical.get("activation", {}).get("candidate_ids")
        or ()
    )

    if _safe_str(resolution.get("action")) != "abstain":
        return _safe_str(resolution.get("action"), "non_abstain"), [
            f"Resolver used { _safe_str(resolution.get('action'), 'non-abstain') } "
            "instead of abstain."
        ], {
            "resolver_action": _safe_str(resolution.get("action")),
        }

    graph_label = _safe_str(graph_row.get("predicted_label")) if graph_row else ""
    graph_citations = _extract_identifiers(graph_row.get("citations", ()) or ()) if graph_row else []
    graph_shape = _graph_shape_signal(
        trace, graph_label, graph_citations, active_ids
    )
    graph_invalid_ids = [item for item in graph_citations if not _VALID_ID_RE.fullmatch(item)]

    graph_candidates = _find_graph_candidates(trace, graph_label)
    active_graph_candidates = bool(
        graph_candidates and set(graph_candidates).intersection(set(active_ids))
    )

    if direct_decision in {"none_of_supplied_candidates", "insufficient"}:
        if not direct.get("candidate_id"):
            if graph_invalid_ids:
                return "protected_graph_missing_bad_graph_citations", [
                    "Graph citation tokens are malformed (often comma-prefixed).",
                    "Normalize and split citation strings before downstream matching."
                ], {
                    "graph_citations": graph_citations,
                    "graph_invalid_citations": graph_invalid_ids,
                    "active_count": len(active_ids),
                    **graph_shape,
                }
            if graph_candidates and not active_graph_candidates:
                return "protected_graph_missing_graph_candidate_inactive", [
                    "Graph-mode family was not in the top active candidate set.",
                    "Increase direct differential activation budget or adjust ranking rules."
                ], {
                    "graph_candidates": graph_candidates,
                    "active_candidates": list(active_ids),
                    **graph_shape,
                }
            if not graph_label:
                return "direct_declined_no_graph_anchor", [
                    "No graph-mode anchor diagnosis available for this case.",
                    "Check run integrity for graph method outputs."
                ], {
                    "direct_decision": direct_decision,
                    **graph_shape,
                }
            return "direct_declined_no_supported_family", [
                "Direct stage did not nominate a supported family candidate.",
                "Review VERSION_III_DIRECT_SYSTEM_PROMPT constraints and rationale-to-pair mapping."
            ], {
                "direct_decision": direct_decision,
                "active_count": len(active_ids),
                **graph_shape,
            }
        if not direct.get("decisive_pair_ids"):
            return "direct_declined_no_decisive_pairs", [
                "Direct differential returned candidate without decisive pair evidence.",
                "Allow softer fallback when rationale is strong and pairs are implied."
            ], {
                "direct_candidate": _safe_str(direct.get("candidate_id")),
                "direct_decision": direct_decision,
            }

        if bool(attack.get("attack")) and not bool(attack_validation.get("valid")):
            return "attack_validation_failed", [
                "Attack was proposed but validation was not accepted.",
                "Review attack-validation instructions and evidence overlap thresholds."
            ], {
                "attack_type": _safe_str(attack.get("attack_type")),
                "attack_scope": _safe_str(attack_validation.get("scope")),
            }

        if bool(attack.get("attack")) and attack.get("attack_type") and protection is None:
            return "no_protected_incumbent_after_attack", [
                "Direct decision was challenged and no protected incumbent met match criteria.",
                "Widen protected incumbent matching conditions or increase citation overlap tolerance."
            ], {
                "attack_type": _safe_str(attack.get("attack_type")),
                "attack_valid": False,
            }

        return "direct_declined_no_protected", [
            "No protected incumbent was available after direct-stage resolution.",
            "Inspect protected incumbent requirements and graph citation quality."
        ], {
            "direct_decision": direct_decision,
            "direct_candidate": _safe_str(direct.get("candidate_id")),
            "active_count": len(active_ids),
            **graph_shape,
        }

    return "other_abstention_reason", [
        "Abstention occurred without a direct-stage insufficiency flag.",
        "Inspect trace fields for parser or verifier edge-case conditions."
    ], {
        "direct_decision": direct_decision,
        "resolver_action": _safe_str(resolution.get("action")),
    }


def dialectical_to_trace_parent(trace: dict[str, Any]) -> dict[str, Any]:
    # helper kept for readability in this module only
    return trace.get("dialectical_trace") or {}


def _case_record(
    case_id: str,
    trace: dict[str, Any],
    prediction: dict[str, Any],
    adjudication: dict[str, Any],
    graph_row: dict[str, Any] | None,
) -> dict[str, Any]:
    dialectical = trace.get("dialectical_trace") or {}
    direct = dialectical.get("direct_differential") or {}
    resolution = trace.get("dialectical_resolution") or {}
    resolver = trace.get("resolver_decision") or {}
    candidate_assessments = resolver.get("candidate_assessments") or ()
    verdicts = Counter(item.get("verdict") for item in candidate_assessments)
    active_ids = tuple(
        dialectical.get("active_candidate_ids")
        or dialectical.get("activation", {}).get("candidate_ids")
        or ()
    )
    candidate_names = _candidate_map(trace)
    graph_label = _safe_str(graph_row.get("predicted_label")) if graph_row else ""

    bucket, recommendations, details = _classify_case(
        trace=trace,
        prediction=prediction,
        graph_row=graph_row,
    )
    graph_shape = _graph_shape_signal(
        trace, graph_label, _extract_identifiers(graph_row.get("citations", ()) or ()),
        tuple(active_ids),
    )

    return {
        "case_id": case_id,
        "status": "abstained" if bool(prediction.get("abstained")) else "resolved",
        "failure_bucket": bucket,
        "failure_recommendations": recommendations,
        "failure_details": details,
        "direct_decision": _safe_str(direct.get("decision")),
        "direct_candidate_id": _safe_str(direct.get("candidate_id")),
        "direct_child_label": _safe_str(direct.get("child_label")),
        "direct_alternative_id": _safe_str(direct.get("strongest_alternative_id")),
        "direct_decisive_pairs": list(direct.get("decisive_pair_ids") or ()),
        "resolver_action": _safe_str(resolution.get("action")),
        "resolution_reason": _safe_str(resolution.get("reason")),
        "leading_diagnoses": list(resolution.get("leading_diagnoses") or ()),
        "possible_diagnoses": list(resolution.get("possible_diagnoses") or ()),
        "active_candidate_count": len(active_ids),
        "active_candidate_ids": list(active_ids),
        "active_candidate_verdicts": {
            _safe_str(item.get("candidate_id")): _safe_str(item.get("verdict"))
            for item in candidate_assessments
        },
        "active_candidate_diagnoses": [
            {
                "candidate_id": cid,
                "diagnosis": candidate_names.get(cid, ""),
                "verdict": next(
                    (item.get("verdict") for item in candidate_assessments
                     if item.get("candidate_id") == cid),
                    "",
                ),
            }
            for cid in active_ids
        ],
        "verdict_counts": dict(verdicts),
        "budget_skips": int(verdicts.get("not_evaluated_due_to_budget", 0)),
        "graph_predicted_label": graph_label,
        "graph_citations": _extract_identifiers(graph_row.get("citations", ()) or ()) if graph_row else [],
        "protected_graph_incumbent": dialectical.get("protected_graph_incumbent"),
        "adjudication_abstained": bool(adjudication.get("abstained")),
        "adjudication_selected": _safe_str(adjudication.get("selected_diagnosis")),
        "evidence_ids": list(prediction.get("citations") or ()),
        "resolver_input_hash": _safe_str(prediction.get("metadata", {}).get("resolver_input_sha256")),
        "abstention_category": _safe_str(prediction.get("metadata", {}).get("abstention_category")),
        "trace_error": _safe_str(trace.get("error")),
        "graph_shape": graph_shape,
    }


def build_failure_profile(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    predictions = load_jsonl(run_dir / "predictions.jsonl")
    traces = load_jsonl(run_dir / "argument_traces.jsonl")
    adjudications = load_jsonl(run_dir / "adjudications.jsonl")

    prediction_by_case: dict[str, dict[str, Any]] = {}
    graph_by_case: dict[str, dict[str, Any]] = {}
    for row in predictions:
        case_id = _safe_str(row.get("case_id"))
        if str(row.get("mode") or "") == MODE:
            prediction_by_case[case_id] = row
        if str(row.get("mode") or "") == GRAPH_MODE:
            graph_by_case[case_id] = row

    trace_by_case = {str(row.get("case_id") or ""): row for row in traces}
    adjudication_by_case = {
        str(row.get("case_id") or ""): row for row in adjudications
    }

    case_ids = sorted(prediction_by_case)
    rows: list[dict[str, Any]] = []
    bucket_counts: Counter[str] = Counter()
    for case_id in case_ids:
        pred = prediction_by_case[case_id]
        trace = trace_by_case.get(case_id)
        if not trace:
            continue
        row = _case_record(
            case_id,
            trace=trace,
            prediction=pred,
            adjudication=adjudication_by_case.get(case_id, {}),
            graph_row=graph_by_case.get(case_id),
        )
        bucket_counts[row["failure_bucket"]] += 1
        rows.append(row)

    total_cases = len(case_ids)
    breakdown = [
        {
            "bucket": bucket,
            "count": count,
            "share": count / total_cases if total_cases else 0.0,
        }
        for bucket, count in sorted(bucket_counts.items(), key=lambda item: (-item[1], item[0]))
    ]

    summary = {
        "run_dir": str(run_dir),
        "mode": MODE,
        "case_count": total_cases,
        "abstention_count": sum(1 for row in rows if row["status"] == "abstained"),
        "bucket_counts": dict(bucket_counts),
    }
    return summary, rows, breakdown


def _build_markdown(run_dir: Path, summary: dict[str, Any], rows: list[dict[str, Any]], breakdown: list[dict[str, Any]]) -> str:
    lines = [
        "# Argumentation Failure Profile",
        f"- Run: `{summary['run_dir']}`",
        f"- Method: `{summary['mode']}`",
        f"- Cases: `{summary['case_count']}`",
        f"- Abstentions: `{summary['abstention_count']}`",
        "",
        "## Failure buckets",
    ]
    for item in breakdown:
        lines.append(f"- `{item['bucket']}`: {item['count']} ({item['share']:.1%})")

    lines.extend([
        "",
        "## Priority cases",
        "| case_id | failure_bucket | direct_decision | failure_recommendations |",
        "|---|---|---|---|",
    ])
    for row in [r for r in rows if r["status"] == "abstained"]:
        lines.append(
            "| "
            f"{row['case_id']} | {row['failure_bucket']} | {row['direct_decision']} | "
            f"{'; '.join(row['failure_recommendations'])[:240]} |"
        )

    lines.extend([
        "",
        "## Common fixes by bucket",
    ])
    for item in breakdown:
        bucket = item["bucket"]
        examples = [
            r["failure_recommendations"][0]
            for r in rows
            if r["failure_bucket"] == bucket and r["failure_recommendations"]
        ]
        if examples:
            lines.append(f"- **{bucket}**: {examples[0]}")

    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile argumentation abstentions and classify likely root causes."
    )
    parser.add_argument("run_dir", help="Run directory path or parent containing comparison run.")
    parser.add_argument(
        "--output",
        default="",
        help="Output path for JSON profile (defaults to argumentation_failure_profile.json in the run dir).",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Also write argumentation_failure_profile.md",
    )
    return parser.parse_args()


def _resolve_run_dir(root: Path) -> Path:
    if root.is_file():
        raise ValueError(f"Expected directory, got file: {root}")
    if (root / "argument_traces.jsonl").exists():
        return root
    if (root / "comparison-development" / "argument_traces.jsonl").exists():
        return root / "comparison-development"
    if (root / "comparison" / "argument_traces.jsonl").exists():
        return root / "comparison"
    raise FileNotFoundError(f"Could not find argument_traces.jsonl under: {root}")


def main() -> int:
    args = _parse_args()
    run_dir = _resolve_run_dir(Path(args.run_dir).expanduser().resolve())
    summary, rows, breakdown = build_failure_profile(run_dir)
    output = {
        "summary": summary,
        "bucket_breakdown": breakdown,
        "cases": rows,
    }
    output_path = Path(args.output) if args.output else run_dir / "argumentation_failure_profile.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.markdown:
        (run_dir / "argumentation_failure_profile.md").write_text(
            _build_markdown(run_dir, summary, rows, breakdown),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
