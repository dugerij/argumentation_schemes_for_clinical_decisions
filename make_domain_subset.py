from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from eval.embedding_benchmark import discover_questions_path
from eval.question_subset import (
    filter_questions_for_domain_and_notes,
    load_questions_jsonl,
    write_questions_jsonl,
)
from helpers.config import parse_optional_int
from helpers.paths import resolve_mimic_discharge_csv
from helpers.term_matching import KeywordSeedMatcher, UMLSSeedVocabularyMatcher
from ingest.mimic import MimicDischargeDomainSubsetConfig, extract_mimic_discharge_domain_subset
from retrieval.concepts.umls import UMLSClient, UMLSConfig


DEFAULT_SPECIALTY_SEEDS: dict[str, tuple[str, ...]] = {
    "renal_metabolic": (
        "acute kidney injury",
        "chronic kidney disease",
        "end stage renal disease",
        "end-stage renal disease",
        "renal failure",
        "kidney failure",
        "glomerulonephritis",
        "nephrotic syndrome",
        "nephritic syndrome",
        "pyelonephritis",
        "hyperkalemia",
        "hypokalemia",
        "hyponatremia",
        "hypernatremia",
        "metabolic acidosis",
        "metabolic alkalosis",
        "diabetic ketoacidosis",
        "uremia",
        "proteinuria",
        "albuminuria",
        "hematuria",
    ),
}


def normalize_specialty_name(specialty: str) -> str:
    """Normalize a specialty label into a stable key."""

    return specialty.strip().lower().replace("-", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create matched clinical-note and MedQA domain subsets.")
    parser.add_argument("--domain", default="renal_metabolic", help="Clinical domain key.")
    parser.add_argument("--notes-csv-path", default=None, help="MIMIC discharge CSV path.")
    parser.add_argument("--notes-output-dir", default="data/evidence/renal_metabolic_discharge_subset")
    parser.add_argument("--questions-input-path", default=None, help="MedQA questions JSONL path.")
    parser.add_argument("--questions-output-path", default="data/eval/renal_metabolic_medqa.jsonl")
    parser.add_argument("--note-limit", default="all", help="Maximum number of matching notes, or 'all'.")
    parser.add_argument("--question-limit", default="all", help="Maximum number of matching questions, or 'all'.")
    parser.add_argument("--note-max-chars", default="all", help="Maximum characters per note, or 'all'.")
    parser.add_argument("--note-type", default=os.environ.get("MIMIC_DISCHARGE_NOTE_TYPE", "DS"))
    parser.add_argument("--min-domain-hits", type=int, default=3)
    parser.add_argument("--min-question-overlap", type=int, default=2)
    parser.add_argument("--notes-matcher", choices=("keyword", "vocab"), default="vocab")
    parser.add_argument("--questions-matcher", choices=("keyword", "vocab"), default="vocab")
    parser.add_argument(
        "--seed-term",
        action="append",
        default=[],
        help="Specialty seed term. Repeat to provide multiple terms. If omitted, built-in seeds are used when available.",
    )
    parser.add_argument("--fresh", action="store_true", help="Ignore existing subset artifacts and rebuild from scratch.")
    return parser.parse_args()


def resolve_seed_terms(domain: str, configured_seed_terms: list[str]) -> tuple[str, ...]:
    """Resolve the specialty seed terms used to build the matching vocabulary."""

    cleaned = tuple(term.strip() for term in configured_seed_terms if term and term.strip())
    if cleaned:
        return cleaned
    builtin = DEFAULT_SPECIALTY_SEEDS.get(domain)
    if builtin:
        return builtin
    raise ValueError(
        f"No built-in seed terms found for specialty '{domain}'. "
        "Pass one or more --seed-term values."
    )


def build_domain_matcher(domain: str, matcher: str, *, seed_terms: tuple[str, ...]):
    """Build a domain matcher for note or question filtering.

    `keyword` uses only the provided seed terms. `vocab` expands the seed terms
    through UMLS once, then filters notes and questions with plain term
    matching against that generated vocabulary.
    """

    if matcher == "keyword":
        return KeywordSeedMatcher(seed_terms)
    client = UMLSClient(UMLSConfig.from_env())
    return UMLSSeedVocabularyMatcher(client, seed_terms)


def existing_subset_notes(output_dir: Path) -> list[Path]:
    return sorted(output_dir.glob("*.txt"))


def should_resume_notes(output_dir: Path, *, fresh: bool) -> bool:
    return not fresh and output_dir.exists() and any(output_dir.glob("*.txt"))


def main() -> None:
    load_dotenv()
    args = parse_args()

    domain = normalize_specialty_name(args.domain)
    notes_csv_path = resolve_mimic_discharge_csv(args.notes_csv_path)
    notes_output_dir = Path(args.notes_output_dir)
    questions_input_path = discover_questions_path(Path(args.questions_input_path) if args.questions_input_path else None)
    if questions_input_path is None:
        raise FileNotFoundError("No MedQA questions file found in default locations.")
    questions_output_path = Path(args.questions_output_path)

    note_limit = parse_optional_int(args.note_limit, default=None)
    question_limit = parse_optional_int(args.question_limit, default=None)
    note_max_chars = parse_optional_int(args.note_max_chars, default=None)
    seed_terms = resolve_seed_terms(domain, args.seed_term)

    notes_matcher = build_domain_matcher(domain, args.notes_matcher, seed_terms=seed_terms)
    questions_matcher = build_domain_matcher(domain, args.questions_matcher, seed_terms=seed_terms)

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
    write_questions_jsonl(questions_output_path, kept_questions)

    manifest = {
        "domain": domain,
        "notes_csv_path": notes_csv_path.as_posix(),
        "notes_output_dir": notes_output_dir.as_posix(),
        "questions_input_path": questions_input_path.as_posix(),
        "questions_output_path": questions_output_path.as_posix(),
        "note_count": len(written_notes),
        "question_count": len(kept_questions),
        "seed_terms": list(seed_terms),
        "note_limit": note_limit,
        "question_limit": question_limit,
        "min_domain_hits": args.min_domain_hits,
        "min_question_overlap": args.min_question_overlap,
        "notes_matcher": args.notes_matcher,
        "questions_matcher": args.questions_matcher,
        "resumed_notes_subset": resumed_notes,
    }
    questions_output_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    selection = {
        "notes": {
            "matcher": args.notes_matcher,
        },
        "questions": {
            "matcher": args.questions_matcher,
            "initial_selection_metadata": result.metadata,
        },
    }
    questions_output_path.with_suffix(".selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
