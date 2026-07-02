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
from helpers.clinical_domains import HybridDomainMatcher, UMLSDomainMatcher, normalize_domain_name
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
    parser.add_argument("--min-domain-hits", type=int, default=1)
    parser.add_argument("--min-question-overlap", type=int, default=2)
    parser.add_argument("--matcher", choices=("hybrid", "umls", "keyword"), default="hybrid")
    parser.add_argument("--prefilter-min-hits", type=int, default=1)
    return parser.parse_args()


def build_domain_matcher(domain: str, matcher: str, *, prefilter_min_hits: int = 1):
    if matcher == "keyword":
        return None
    client = UMLSClient(UMLSConfig.from_env())
    umls_matcher = UMLSDomainMatcher(client, domain)
    if matcher == "umls":
        return umls_matcher
    return HybridDomainMatcher(domain, umls_matcher, prefilter_min_hits=prefilter_min_hits)


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
    domain_matcher = build_domain_matcher(domain, args.matcher, prefilter_min_hits=args.prefilter_min_hits)

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
        matcher=domain_matcher,
    )

    questions = load_questions_jsonl(questions_input_path)
    result = filter_questions_for_domain_and_notes(
        questions=questions,
        domain=domain,
        notes_dir=notes_output_dir,
        min_overlap_terms=args.min_question_overlap,
        limit=question_limit,
        matcher=domain_matcher,
    )
    write_questions_jsonl(questions_output_path, result.kept_questions)

    manifest = {
        "domain": domain,
        "notes_csv_path": notes_csv_path.as_posix(),
        "notes_output_dir": notes_output_dir.as_posix(),
        "questions_input_path": questions_input_path.as_posix(),
        "questions_output_path": questions_output_path.as_posix(),
        "note_count": len(written_notes),
        "question_count": len(result.kept_questions),
        "note_limit": note_limit,
        "question_limit": question_limit,
        "min_domain_hits": args.min_domain_hits,
        "min_question_overlap": args.min_question_overlap,
        "matcher": args.matcher,
        "prefilter_min_hits": args.prefilter_min_hits,
    }
    questions_output_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    questions_output_path.with_suffix(".selection.json").write_text(json.dumps(result.metadata, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
