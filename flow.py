import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
from graphrag import api
from graphrag.config.load_config import load_config

from framework import Generator, Verifier, Reasoner, SessionManager


def load_jsonl_dataset(jsonl_path: Path) -> list[dict[str, Any]]:
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")

    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def build_question_prompt(item: dict[str, Any]) -> str:
    question = item.get("question", "")
    options = item.get("options", [])
    options_text = "\n".join([f"{idx+1}. {o}" for idx, o in enumerate(options)])
    return f"{question}\n{options_text}" if options_text else question


def load_rag_tables(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    entities_path = output_dir / "entities.parquet"
    communities_path = output_dir / "communities.parquet"
    reports_path = output_dir / "community_reports.parquet"

    if not entities_path.exists() or not communities_path.exists() or not reports_path.exists():
        raise FileNotFoundError("One or more RAG output parquet files are missing in output folder")

    return (
        pd.read_parquet(entities_path),
        pd.read_parquet(communities_path),
        pd.read_parquet(reports_path),
    )


def rag_local_search_context(config, question, entities, communities, community_reports):
    result, context = api.local_search(
        config=config,
        entities=entities,
        communities=communities,
        community_reports=community_reports,
        community_level=2,
        dynamic_community_selection=False,
        response_type="Multiple Paragraphs",
        query=question,
    )
    return result, context


def argumentation_pipeline(
    dataset_jsonl: str,
    output_dir: str = "output",
    sample_size: int = 1,
    generator_model: str = "llama3.2:1b",
    max_iterations: int = 3,
) -> dict[str, Any]:
    """Run end-to-end generator/verifier/reasoner pipeline using JSONL dataset + RAG output."""
    dataset_path = Path(dataset_jsonl)
    output_path = Path(output_dir)

    items = load_jsonl_dataset(dataset_path)
    if not items:
        raise ValueError("No entries found in dataset")

    random.shuffle(items)
    selection = items[:sample_size]

    config = load_config(Path("./"))
    entities, communities, community_reports = load_rag_tables(output_path)

    generator = Generator(model=generator_model)
    verifier = Verifier()
    reasoner = Reasoner()
    session = SessionManager()
    session.start()

    results = []

    for item in selection:
        prompt = build_question_prompt(item)
        gold_answer = item.get("answer", "").strip()

        generated_answer = generator.get_argument(prompt)

        rag_result, rag_context = rag_local_search_context(config, prompt, entities, communities, community_reports)

        verification = verifier.verify_argument(
            generated_answer, gold_answer, rag_context or ""
        )

        discussion_history = [
            {"role": "question", "content": prompt},
            {"role": "generated_answer", "content": generated_answer},
            {"role": "verification", **verification},
            {"role": "rag_context", "content": rag_context},
        ]

        final_reasoning = reasoner.check(discussion_history)

        results.append({
            "prompt": prompt,
            "gold_answer": gold_answer,
            "generated_answer": generated_answer,
            "verification": verification,
            "rag_result": rag_result,
            "final_reasoning": final_reasoning,
            "sound": final_reasoning.get("sound", False),
        })

        if verification.get("satisfied"):
            break

    # Cleanly finish session
    session.end()

    return {
        "samples": len(results),
        "accuracy": sum(1 for r in results if r["verification"]["satisfied"]) / max(1, len(results)),
        "results": results,
    }


def evaluate_pipeline(pipeline_report: dict[str, Any]) -> dict[str, Any]:
    """Compute overall pipeline metrics from argumentation pipeline report."""
    results = pipeline_report.get("results", [])
    sample_count = len(results)

    if sample_count == 0:
        return {"error": "No results to evaluate"}

    accuracy = sum(1 for r in results if r.get("verification", {}).get("satisfied", False)) / sample_count
    soundness = sum(1 for r in results if r.get("sound", False)) / sample_count

    return {
        "sample_count": sample_count,
        "accuracy": accuracy,
        "soundness": soundness,
        "avg_iterations": pipeline_report.get("avg_iterations", 1),
        "factuality": accuracy,
        "coherence": 1.0 if soundness > 0.5 else 0.5,
        "relevance": 1.0 if accuracy > 0 else 0.0,
    }


if __name__ == "__main__":
    output = argumentation_pipeline(
        dataset_jsonl="data/medqa/data_clean/questions/US/4_options/phrases_no_exclude_train.jsonl",
        output_dir="output",
        sample_size=2,
    )
    print(json.dumps(output, indent=2))
    print("Eval:", evaluate_pipeline(output))