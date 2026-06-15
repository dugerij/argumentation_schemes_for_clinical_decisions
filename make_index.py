import argparse
import asyncio
import subprocess
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_DOMAIN = """
clinical medicine, including basic medical sciences such as anatomy, biochemistry,
physiology, pathophysiology, pathology and pharmacology; epidemiology; clinical
symptomatology; examination findings; investigations and interpretation; disease
progression and prognosis; treatment mechanisms, side effects, interactions, and
patient-centered care.
""".strip()

CLINICAL_ARGUMENTATION_DOMAIN = """
clinical medicine and evidence-based guideline reasoning over MIMIC-IV notes,
NICE guidelines, WHO guidelines, and medical textbooks. Extract clinically
relevant entities and relationships for auditable recommendation generation.
Prefer UMLS/MEDCIN-aligned concepts where available, including diseases,
symptoms, signs, medications, procedures, laboratory tests, findings,
contraindications, adverse effects, complications, risk factors, clinical goals,
guideline recommendations, and treatment relationships. The resulting graph will
support formal argumentation schemes, medical critical questions, and Abstract
Argumentation Framework adjudication of clinical conflicts.
""".strip()


def tune_prompts(
    domain: str,
    output: str,
    selection_method: str,
    limit: int,
    max_tokens: int,
    min_examples_required: int,
    discover_entity_types: bool,
    language: str,
) -> None:
    command = [
        "graphrag",
        "prompt-tune",
        "--domain",
        domain,
        "--output",
        output,
        "--selection-method",
        selection_method,
        "--limit",
        str(limit),
        "--max-tokens",
        str(max_tokens),
        "--min-examples-required",
        str(min_examples_required),
        "--language",
        language,
    ]

    if discover_entity_types:
        command.append("--discover-entity-types")
    else:
        command.append("--no-discover-entity-types")

    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GraphRAG indexing utilities.")
    parser.add_argument(
        "command",
        choices=("build", "prompt-tune", "prompt-tune-clinical"),
        help="Run GraphRAG indexing or prompt tuning.",
    )
    parser.add_argument("--config", default="settings.yaml", help="Path to GraphRAG settings YAML.")
    parser.add_argument("--method", default="standard", help="GraphRAG indexing method.")
    parser.add_argument("--domain", default=DEFAULT_DOMAIN, help="Domain text for prompt tuning.")
    parser.add_argument("--output", default="prompts", help="Prompt output directory relative to project root.")
    parser.add_argument(
        "--selection-method",
        default="auto",
        choices=("all", "random", "top", "auto"),
        help="Text chunk selection method for prompt tuning.",
    )
    parser.add_argument("--limit", type=int, default=30, help="Number of documents to load for prompt tuning.")
    parser.add_argument("--max-tokens", type=int, default=4000, help="Maximum token budget for prompt generation.")
    parser.add_argument(
        "--min-examples-required",
        type=int,
        default=3,
        help="Minimum examples to generate/include in the extraction prompt.",
    )
    parser.add_argument("--language", default="English", help="Primary prompt language.")
    parser.add_argument(
        "--discover-entity-types",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow GraphRAG to discover entity types during prompt tuning.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    if args.command == "build":
        from rag.index import build_graphrag_index

        asyncio.run(build_graphrag_index(Path(args.config), args.method))
    elif args.command == "prompt-tune":
        tune_prompts(
            domain=args.domain,
            output=args.output,
            selection_method=args.selection_method,
            limit=args.limit,
            max_tokens=args.max_tokens,
            min_examples_required=args.min_examples_required,
            discover_entity_types=args.discover_entity_types,
            language=args.language,
        )
    elif args.command == "prompt-tune-clinical":
        tune_prompts(
            domain=CLINICAL_ARGUMENTATION_DOMAIN,
            output=args.output,
            selection_method=args.selection_method,
            limit=args.limit,
            max_tokens=args.max_tokens,
            min_examples_required=args.min_examples_required,
            discover_entity_types=args.discover_entity_types,
            language=args.language,
        )


if __name__ == "__main__":
    main()
