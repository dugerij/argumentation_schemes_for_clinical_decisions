from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from eval.embedding_benchmark import discover_questions_path
from eval.question_subset import (
    filter_questions_for_domain_and_notes,
    load_questions_jsonl,
    write_questions_jsonl,
)
from helpers.clinical_domains import (
    HybridDomainMatcher,
    KeywordDomainMatcher,
    UMLSDomainMatcher,
    normalize_domain_name,
)
from helpers.config import parse_optional_int
from ingest.mimic import MimicDischargeDomainSubsetConfig, extract_mimic_discharge_domain_subset
from retrieval.concepts.umls import UMLSClient, UMLSConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create matched clinical-note and MedQA domain subsets.")
    parser.add_argument("--domain", default="renal_metabolic", help="Clinical domain key.")
    parser.add_argument("--notes-csv-path", default=None, help="MIMIC discharge CSV path.")
    parser.add_argument("--notes-output-dir", default="data/evidence/renal_metabolic_discharge_subset")
    parser.add_argument("--questions-input-path", default=None, help="MedQA questions JSONL path.")
    parser.add_argument("--questions-output-path", default="data/eval/renal_metabolic_medqa.jsonl")
    parser.add_argument("--note-limit", default="all", help="Maximum number of matching notes, or 'all'.")
    parser.add_argument("--question-limit", default="all", help="Maximum number of matching questions, or 'all'.")
    parser.add_argument("--note-max-chars", default="6000", help="Maximum characters per note, or 'all'.")
    parser.add_argument("--note-type", default=os.environ.get("MIMIC_DISCHARGE_NOTE_TYPE", "DS"))
    parser.add_argument("--min-domain-hits", type=int, default=2)
    parser.add_argument("--min-question-overlap", type=int, default=2)
    parser.add_argument("--notes-matcher", choices=("hybrid", "umls", "keyword"), default="keyword")
    parser.add_argument("--questions-matcher", choices=("hybrid", "umls", "keyword"), default="keyword")
    parser.add_argument("--notes-prefilter-min-hits", type=int, default=2)
    parser.add_argument("--questions-prefilter-min-hits", type=int, default=1)
    parser.add_argument("--refine-notes-with-umls", action="store_true")
    parser.add_argument("--refine-questions-with-umls", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing subset artifacts and rebuild from scratch.")
    return parser.parse_args()


def build_domain_matcher(domain: str, matcher: str, *, prefilter_min_hits: int = 1):
    if matcher == "keyword":
        return KeywordDomainMatcher(domain)
    client = UMLSClient(UMLSConfig.from_env())
    umls_matcher = UMLSDomainMatcher(client, domain)
    if matcher == "umls":
        return umls_matcher
    return HybridDomainMatcher(domain, umls_matcher, prefilter_min_hits=prefilter_min_hits)


def matcher_failed_request_count(matcher: Any) -> int:
    if isinstance(matcher, HybridDomainMatcher):
        return matcher_failed_request_count(matcher.umls_matcher)
    extractor = getattr(matcher, "extractor", None)
    client = getattr(extractor, "client", None)
    failed_request_count = getattr(client, "failed_request_count", None)
    if isinstance(failed_request_count, int):
        return failed_request_count
    return 0


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_checkpoint(path: Path, state: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def existing_subset_notes(output_dir: Path) -> list[Path]:
    return sorted(output_dir.glob("*.txt"))


def should_resume_notes(output_dir: Path, *, fresh: bool) -> bool:
    return not fresh and output_dir.exists() and any(output_dir.glob("*.txt"))


def refine_note_subset_with_matcher(
    note_paths: list[Path],
    *,
    matcher: Any,
    min_domain_hits: int,
    checkpoint_path: Path,
) -> tuple[list[Path], list[dict[str, Any]], int]:
    state = load_checkpoint(checkpoint_path)
    kept: list[Path] = []
    metadata: list[dict[str, Any]] = []
    pending_count = 0

    for path in note_paths:
        key = path.name
        cached = state.get(key)
        if cached and cached.get("status") in {"included", "excluded"}:
            metadata.append(cached)
            if cached["status"] == "included" and path.exists():
                kept.append(path)
            elif cached["status"] == "excluded" and path.exists():
                path.unlink(missing_ok=True)
            continue

        failures_before = matcher_failed_request_count(matcher)
        hit_count, matched_terms = matcher.match_details(path.read_text(encoding="utf-8"))
        failures_after = matcher_failed_request_count(matcher)

        if failures_after > failures_before:
            row = {
                "file": key,
                "status": "pending_retry",
                "domain_hit_count": None,
                "matched_terms": [],
                "included": None,
            }
            state[key] = row
            metadata.append(row)
            pending_count += 1
            save_checkpoint(checkpoint_path, state)
            continue

        included = hit_count >= min_domain_hits
        row = {
            "file": key,
            "status": "included" if included else "excluded",
            "domain_hit_count": hit_count,
            "matched_terms": matched_terms[:20],
            "included": included,
        }
        state[key] = row
        metadata.append(row)
        if included:
            kept.append(path)
        else:
            path.unlink(missing_ok=True)
        save_checkpoint(checkpoint_path, state)

    return kept, metadata, pending_count


def question_key(item: dict[str, Any]) -> str:
    return str(item.get("question", "")).strip()


def refine_questions_with_matcher(
    questions: list[dict[str, Any]],
    *,
    matcher: Any,
    limit: int | None,
    checkpoint_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    state = load_checkpoint(checkpoint_path)
    kept: list[tuple[int, dict[str, Any]]] = []
    metadata: list[dict[str, Any]] = []
    pending_count = 0

    for item in questions:
        key = question_key(item)
        cached = state.get(key)
        if cached and cached.get("status") in {"included", "excluded"}:
            metadata.append(cached)
            if cached["status"] == "included":
                kept.append((int(cached.get("domain_hit_count") or 0), item))
            continue

        parts = [str(item.get("question", ""))]
        options = item.get("options") or {}
        if isinstance(options, dict):
            parts.extend(str(value) for value in options.values())
        phrases = item.get("metamap_phrases") or []
        if isinstance(phrases, list):
            parts.extend(str(value) for value in phrases)

        failures_before = matcher_failed_request_count(matcher)
        hit_count, matched_terms = matcher.match_details("\n".join(parts))
        failures_after = matcher_failed_request_count(matcher)

        if failures_after > failures_before:
            row = {
                "question": item.get("question"),
                "status": "pending_retry",
                "domain_hit_count": None,
                "matched_terms": [],
                "included": None,
            }
            state[key] = row
            metadata.append(row)
            pending_count += 1
            save_checkpoint(checkpoint_path, state)
            continue

        included = hit_count > 0
        row = {
            "question": item.get("question"),
            "status": "included" if included else "excluded",
            "domain_hit_count": hit_count,
            "matched_terms": matched_terms[:20],
            "included": included,
        }
        state[key] = row
        metadata.append(row)
        if included:
            kept.append((hit_count, item))
        save_checkpoint(checkpoint_path, state)

    kept.sort(key=lambda row: (-row[0], str(row[1].get("question", ""))))
    selected_rows = kept if limit is None else kept[:limit]
    return [item for _, item in selected_rows], metadata, pending_count


def main() -> None:
    load_dotenv()
    args = parse_args()

    domain = normalize_domain_name(args.domain)
    notes_csv_path = Path(args.notes_csv_path or os.environ.get("MIMIC_DISCHARGE_CSV", "data/mimic_iv_note/discharge.csv"))
    notes_output_dir = Path(args.notes_output_dir)
    questions_input_path = discover_questions_path(Path(args.questions_input_path) if args.questions_input_path else None)
    if questions_input_path is None:
        raise FileNotFoundError("No MedQA questions file found in default locations.")
    questions_output_path = Path(args.questions_output_path)

    note_limit = parse_optional_int(args.note_limit, default=None)
    question_limit = parse_optional_int(args.question_limit, default=None)
    note_max_chars = parse_optional_int(args.note_max_chars, default=6000)

    notes_matcher = build_domain_matcher(
        domain,
        args.notes_matcher,
        prefilter_min_hits=args.notes_prefilter_min_hits,
    )
    questions_matcher = build_domain_matcher(
        domain,
        args.questions_matcher,
        prefilter_min_hits=args.questions_prefilter_min_hits,
    )

    resumed_notes = should_resume_notes(notes_output_dir, fresh=args.fresh)
    if resumed_notes:
        written_notes = existing_subset_notes(notes_output_dir)
    else:
        written_notes = extract_mimic_discharge_domain_subset(
            MimicDischargeDomainSubsetConfig(
                csv_path=notes_csv_path,
                output_dir=notes_output_dir,
                domain=domain,
                limit=note_limit,
                note_type=args.note_type,
                max_chars=note_max_chars,
                min_domain_hits=args.min_domain_hits,
                overwrite=True,
            ),
            matcher=notes_matcher,
        )

    note_refinement_metadata: list[dict[str, Any]] = []
    note_pending_count = 0
    note_checkpoint_path = notes_output_dir / ".umls_refinement_notes.json"
    if args.refine_notes_with_umls:
        written_notes, note_refinement_metadata, note_pending_count = refine_note_subset_with_matcher(
            written_notes,
            matcher=build_domain_matcher(domain, "umls"),
            min_domain_hits=args.min_domain_hits,
            checkpoint_path=note_checkpoint_path,
        )

    questions = load_questions_jsonl(questions_input_path)
    result = filter_questions_for_domain_and_notes(
        questions=questions,
        domain=domain,
        notes_dir=notes_output_dir,
        min_overlap_terms=args.min_question_overlap,
        limit=question_limit,
        matcher=questions_matcher,
    )

    kept_questions = result.kept_questions
    question_refinement_metadata: list[dict[str, Any]] = []
    question_pending_count = 0
    question_checkpoint_path = questions_output_path.with_suffix(".umls_refinement_questions.json")
    if args.refine_questions_with_umls:
        kept_questions, question_refinement_metadata, question_pending_count = refine_questions_with_matcher(
            kept_questions,
            matcher=build_domain_matcher(domain, "umls"),
            limit=question_limit,
            checkpoint_path=question_checkpoint_path,
        )
    write_questions_jsonl(questions_output_path, kept_questions)

    manifest = {
        "domain": domain,
        "notes_csv_path": notes_csv_path.as_posix(),
        "notes_output_dir": notes_output_dir.as_posix(),
        "questions_input_path": questions_input_path.as_posix(),
        "questions_output_path": questions_output_path.as_posix(),
        "note_count": len(written_notes),
        "question_count": len(kept_questions),
        "note_limit": note_limit,
        "question_limit": question_limit,
        "min_domain_hits": args.min_domain_hits,
        "min_question_overlap": args.min_question_overlap,
        "notes_matcher": args.notes_matcher,
        "questions_matcher": args.questions_matcher,
        "notes_prefilter_min_hits": args.notes_prefilter_min_hits,
        "questions_prefilter_min_hits": args.questions_prefilter_min_hits,
        "refine_notes_with_umls": args.refine_notes_with_umls,
        "refine_questions_with_umls": args.refine_questions_with_umls,
        "resumed_notes_subset": resumed_notes,
        "note_refinement_pending_count": note_pending_count,
        "question_refinement_pending_count": question_pending_count,
    }
    questions_output_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    selection = {
        "notes": {
            "matcher": args.notes_matcher,
            "refined_with_umls": args.refine_notes_with_umls,
            "checkpoint_path": note_checkpoint_path.as_posix(),
            "pending_count": note_pending_count,
            "refinement_metadata": note_refinement_metadata,
        },
        "questions": {
            "matcher": args.questions_matcher,
            "refined_with_umls": args.refine_questions_with_umls,
            "checkpoint_path": question_checkpoint_path.as_posix(),
            "pending_count": question_pending_count,
            "initial_selection_metadata": result.metadata,
            "refinement_metadata": question_refinement_metadata,
        },
    }
    questions_output_path.with_suffix(".selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
