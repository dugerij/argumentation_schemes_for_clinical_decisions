from pathlib import Path

import pandas as pd
from graphrag import api
from graphrag.config.load_config import load_config


REQUIRED_INDEX_FILES = (
    "entities.parquet",
    "communities.parquet",
    "community_reports.parquet",
)


def missing_index_files(output_dir: Path) -> list[Path]:
    return [output_dir / name for name in REQUIRED_INDEX_FILES if not (output_dir / name).exists()]


def index_needs_rebuild(output_dir: Path) -> tuple[bool, list[str]]:
    reasons = [f"missing {path}" for path in missing_index_files(output_dir)]

    entities_path = output_dir / "entities.parquet"
    if entities_path.exists():
        entity_columns = set(pd.read_parquet(entities_path).columns)
        if "id" not in entity_columns:
            reasons.append("entities.parquet was built by an incompatible workflow and has no id column")

    return bool(reasons), reasons


async def build_graphrag_index(config_path: Path, method: str = "standard"):
    config = load_config(config_path)
    return await api.build_index(config=config, method=method)


async def ensure_index(config, output_dir: Path, method: str) -> None:
    needs_rebuild, reasons = index_needs_rebuild(output_dir)
    if not needs_rebuild:
        print(f"Using existing GraphRAG index in {output_dir}")
        return

    print("GraphRAG index is missing or incompatible:")
    for reason in reasons:
        print(f"  - {reason}")

    print(f"Building GraphRAG index with method={method!r}...")
    await api.build_index(config=config, method=method)

    missing = missing_index_files(output_dir)
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "GraphRAG indexing finished, but required artifacts are still missing:\n"
            f"{missing_text}\n"
            "Check the GraphRAG logs and prompt output. Drift search requires community reports."
        )

    entity_columns = set(pd.read_parquet(output_dir / "entities.parquet").columns)
    if "id" not in entity_columns:
        raise ValueError("GraphRAG indexing finished, but entities.parquet still has no id column.")


def load_graph_tables(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_parquet(output_dir / "entities.parquet"),
        pd.read_parquet(output_dir / "communities.parquet"),
        pd.read_parquet(output_dir / "community_reports.parquet"),
    )
