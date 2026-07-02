from __future__ import annotations

import json
import os
import re
from html import escape
from pathlib import Path
from typing import Iterable

import pandas as pd
from pyvis.network import Network


DEFAULT_ENTITY_TYPES = {
    "ANATOMICAL_STRUCTURE",
    "ANATOMY",
    "BIOMARKER",
    "COMPLICATION",
    "DISEASE",
    "DRUG",
    "FINDING",
    "LAB_TEST",
    "LABORATORY_TEST",
    "MEDICATION",
    "PROCEDURE",
    "RISK_FACTOR",
    "SYMPTOM",
    "THERAPY",
}

ENTITY_COLORS = {
    "DISEASE": "#d64545",
    "SYMPTOM": "#f28e2b",
    "FINDING": "#edc948",
    "DRUG": "#4e79a7",
    "MEDICATION": "#4e79a7",
    "THERAPY": "#59a14f",
    "PROCEDURE": "#76b7b2",
    "LAB_TEST": "#b07aa1",
    "LABORATORY_TEST": "#b07aa1",
    "BIOMARKER": "#9c755f",
    "ANATOMY": "#bab0ab",
    "ANATOMICAL_STRUCTURE": "#bab0ab",
    "RISK_FACTOR": "#ff9da7",
    "COMPLICATION": "#e15759",
}

UMLS_HINT_RE = re.compile(
    r"^-\s+(?P<type>[A-Z_]+):\s+(?P<title>.+?)\s+"
    r"\(CUI\s+(?P<cui>[^,]+),\s+source\s+(?P<source>[^,]+),\s+semantic type\s+(?P<semantic>[^)]+)\)"
)

GENERIC_ENTITY_TITLES = {
    "Disease",
    "Interventional procedure",
}

ENTITY_COLUMNS = ("title", "type", "description", "frequency", "degree")
RELATIONSHIP_COLUMNS = ("source", "target", "description", "weight", "combined_degree")


def _as_type_set(entity_types: Iterable[str] | None) -> set[str]:
    if entity_types is None:
        return set(DEFAULT_ENTITY_TYPES)
    return {str(entity_type).upper() for entity_type in entity_types}


def _tooltip(entity_type: str, description: object) -> str:
    description_text = "" if pd.isna(description) else str(description)
    parts = [f"<b>{escape(entity_type)}</b>"]
    if description_text:
        parts.append(escape(description_text[:900]))
    return "<br><br>".join(parts)


def _is_readable_entity_title(title: str, source_vocabulary: str | None = None) -> bool:
    if title in GENERIC_ENTITY_TITLES:
        return False
    if len(title) > 120:
        return False
    if source_vocabulary == "HCPCS" and len(title) > 80:
        return False
    return True


def _legend_html(type_counts: dict[str, int], node_count: int, edge_count: int) -> str:
    items = []
    for entity_type, count in sorted(type_counts.items()):
        color = ENTITY_COLORS.get(entity_type, "#8f8f8f")
        items.append(
            f'<div class="legend-item">'
            f'<span class="swatch" style="background:{color}"></span>'
            f'<span>{escape(entity_type.replace("_", " ").title())}</span>'
            f'<strong>{count}</strong>'
            f"</div>"
        )
    return f"""
<style>
  #clinical-legend {{
    position: fixed;
    top: 14px;
    left: 14px;
    z-index: 20;
    width: 260px;
    max-height: calc(100vh - 28px);
    overflow: auto;
    background: rgba(255, 255, 255, 0.94);
    border: 1px solid #d8dee9;
    border-radius: 8px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.14);
    color: #1f2937;
    font: 13px/1.35 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    padding: 12px;
  }}
  #clinical-legend h2 {{
    font-size: 14px;
    margin: 0 0 4px;
  }}
  #clinical-legend .meta {{
    color: #5b6472;
    margin-bottom: 10px;
  }}
  #clinical-legend .legend-item {{
    align-items: center;
    display: grid;
    gap: 8px;
    grid-template-columns: 12px 1fr auto;
    margin: 6px 0;
  }}
  #clinical-legend .swatch {{
    border-radius: 50%;
    display: inline-block;
    height: 12px;
    width: 12px;
  }}
</style>
<div id="clinical-legend">
  <h2>Clinical Entity Graph</h2>
  <div class="meta">{node_count} entities, {edge_count} relationships</div>
  {''.join(items)}
</div>
"""


def _ensure_graph_table_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    normalized = frame.copy()
    for column in columns:
        if column not in normalized.columns:
            normalized[column] = pd.Series(dtype="object")
    return normalized.loc[:, list(columns)]


def _load_parquet_graph(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    entities_path = output_dir / "entities.parquet"
    relationships_path = output_dir / "relationships.parquet"
    if not entities_path.exists() or not relationships_path.exists():
        missing = [path.name for path in (entities_path, relationships_path) if not path.exists()]
        raise FileNotFoundError(
            "Missing clinical graph table(s) in "
            f"{output_dir}: {', '.join(missing)}. "
            "This LlamaIndex build usually persists property_graph_store.json; "
            "use source='auto' or source='property_store' to visualize it."
        )

    entities = _ensure_graph_table_columns(pd.read_parquet(entities_path), ENTITY_COLUMNS)
    relationships = _ensure_graph_table_columns(pd.read_parquet(relationships_path), RELATIONSHIP_COLUMNS)
    return entities, relationships


def _load_property_store_hint_graph(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    store_path = output_dir / "property_graph_store.json"
    if not store_path.exists():
        raise FileNotFoundError(f"Missing LlamaIndex property graph store: {store_path}")

    store = json.loads(store_path.read_text(encoding="utf-8"))
    entity_records: dict[str, dict[str, object]] = {}
    edge_records: dict[tuple[str, str], dict[str, object]] = {}

    for node in store.get("nodes", {}).values():
        if node.get("label") != "text_chunk":
            continue
        text = str(node.get("text") or "")
        source_name = str((node.get("properties") or {}).get("source_name") or "source chunk")
        chunk_entities = []
        for line in text.splitlines():
            match = UMLS_HINT_RE.match(line.strip())
            if not match:
                continue

            title = match.group("title").strip()
            entity_type = match.group("type").strip().upper()
            source_vocabulary = match.group("source").strip()
            if not _is_readable_entity_title(title, source_vocabulary):
                continue
            record = entity_records.setdefault(
                title,
                {
                    "title": title,
                    "type": entity_type,
                    "description": (
                        f"CUI {match.group('cui')}; source {source_vocabulary}; "
                        f"semantic type {match.group('semantic')}"
                    ),
                    "frequency": 0,
                    "degree": 0,
                },
            )
            record["frequency"] = int(record["frequency"]) + 1
            chunk_entities.append(title)

        unique_titles = sorted(set(chunk_entities))
        for index, source in enumerate(unique_titles):
            for target in unique_titles[index + 1 :]:
                key = (source, target)
                edge = edge_records.setdefault(
                    key,
                    {
                        "source": source,
                        "target": target,
                        "description": f"Co-mentioned in UMLS hints from {source_name}",
                        "weight": 0.0,
                        "combined_degree": 0,
                    },
                )
                edge["weight"] = float(edge["weight"]) + 1.0

    for source, target in edge_records:
        if source in entity_records:
            entity_records[source]["degree"] = int(entity_records[source]["degree"]) + 1
        if target in entity_records:
            entity_records[target]["degree"] = int(entity_records[target]["degree"]) + 1

    for edge in edge_records.values():
        source_degree = int(entity_records[str(edge["source"])]["degree"])
        target_degree = int(entity_records[str(edge["target"])]["degree"])
        edge["combined_degree"] = source_degree + target_degree

    entities = _ensure_graph_table_columns(pd.DataFrame(entity_records.values()), ENTITY_COLUMNS)
    relationships = _ensure_graph_table_columns(pd.DataFrame(edge_records.values()), RELATIONSHIP_COLUMNS)
    return entities, relationships


def _load_graph_tables(output_dir: Path, source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if source == "parquet":
        return _load_parquet_graph(output_dir)
    if source == "property_store":
        return _load_property_store_hint_graph(output_dir)
    if source != "auto":
        raise ValueError("source must be 'auto', 'parquet', or 'property_store'")

    try:
        return _load_parquet_graph(output_dir)
    except FileNotFoundError:
        return _load_property_store_hint_graph(output_dir)


def _select_graph_subset(
    output_dir: Path,
    *,
    entity_types: Iterable[str] | None = None,
    source: str = "auto",
    max_nodes: int = 80,
    max_edges: int = 160,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    allowed_types = _as_type_set(entity_types)

    entities, relationships = _load_graph_tables(output_dir, source=source)
    entities = _ensure_graph_table_columns(entities, ENTITY_COLUMNS)
    relationships = _ensure_graph_table_columns(relationships, RELATIONSHIP_COLUMNS)

    entities["type"] = entities["type"].fillna("UNKNOWN").astype(str).str.upper()
    entities["title"] = entities["title"].fillna("").astype(str)
    entities["degree"] = pd.to_numeric(entities["degree"], errors="coerce").fillna(0)
    entities["frequency"] = pd.to_numeric(entities["frequency"], errors="coerce").fillna(0)
    relationships["source"] = relationships["source"].fillna("").astype(str)
    relationships["target"] = relationships["target"].fillna("").astype(str)
    relationships["weight"] = pd.to_numeric(relationships["weight"], errors="coerce").fillna(0.0)
    relationships["combined_degree"] = pd.to_numeric(relationships["combined_degree"], errors="coerce").fillna(0)

    keep = entities[entities["type"].isin(allowed_types)].copy()
    keep = keep.sort_values(
        by=["degree", "frequency", "title"],
        ascending=[False, False, True],
        kind="stable",
    ).head(max_nodes)
    selected_titles = set(keep["title"])

    edges = relationships[
        relationships["source"].isin(selected_titles) & relationships["target"].isin(selected_titles)
    ].copy()
    edges = edges.sort_values(
        by=["weight", "combined_degree", "source", "target"],
        ascending=[False, False, True, True],
        kind="stable",
    ).head(max_edges)
    return keep, edges


def save_clinical_entity_graph(
    output_dir: Path,
    html_path: Path | None = None,
    *,
    entity_types: Iterable[str] | None = None,
    source: str = "auto",
    max_nodes: int = 80,
    max_edges: int = 160,
    height: str = "760px",
    width: str = "100%",
) -> Path:
    """Render a readable graph from entity and relationship parquet files."""

    output_dir = Path(output_dir)
    html_path = Path(html_path) if html_path is not None else output_dir / "hh_property_graph.html"
    keep, edges = _select_graph_subset(
        output_dir,
        entity_types=entity_types,
        source=source,
        max_nodes=max_nodes,
        max_edges=max_edges,
    )

    net = Network(height=height, width=width, directed=True, notebook=False, cdn_resources="in_line")
    net.force_atlas_2based(gravity=-55, central_gravity=0.012, spring_length=150, spring_strength=0.05)

    for row in keep.itertuples(index=False):
        degree = int(getattr(row, "degree", 0) or 0)
        entity_type = str(getattr(row, "type"))
        title = str(getattr(row, "title"))
        node_color = ENTITY_COLORS.get(entity_type, "#8f8f8f")
        net.add_node(
            title,
            label=title,
            title=_tooltip(entity_type, getattr(row, "description", "")),
            group=entity_type,
            size=min(42, 16 + degree * 2),
            font={"size": 17, "face": "arial", "strokeWidth": 3, "strokeColor": "#ffffff"},
        )
        net.node_map[title]["color"] = node_color

    for row in edges.itertuples(index=False):
        weight = float(getattr(row, "weight", 1.0) or 1.0)
        description = "" if pd.isna(getattr(row, "description", "")) else str(getattr(row, "description", ""))
        net.add_edge(
            str(getattr(row, "source")),
            str(getattr(row, "target")),
            title=escape(description),
            value=max(1.0, weight),
            arrows="to",
            color={"color": "#7b8794", "opacity": 0.62},
        )

    type_counts = keep["type"].value_counts().to_dict()
    html_path.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(html_path), notebook=False, open_browser=False)
    html = html_path.read_text(encoding="utf-8")
    html = html.replace(
        "</body>",
        _legend_html(type_counts, node_count=len(keep), edge_count=len(edges)) + "\n</body>",
    )
    html_path.write_text(html, encoding="utf-8")
    return html_path


def save_clinical_entity_graph_jpeg(
    output_dir: Path,
    jpeg_path: Path | None = None,
    *,
    entity_types: Iterable[str] | None = None,
    source: str = "auto",
    max_nodes: int = 80,
    max_edges: int = 160,
    title: str = "Clinical Entity Property Graph",
    seed: int = 42,
) -> Path:
    """Render a static JPEG plot of the same clinical graph subset used by the HTML view."""

    output_dir = Path(output_dir)
    jpeg_path = Path(jpeg_path) if jpeg_path is not None else output_dir / "hh_property_graph.jpg"
    keep, edges = _select_graph_subset(
        output_dir,
        entity_types=entity_types,
        source=source,
        max_nodes=max_nodes,
        max_edges=max_edges,
    )

    jpeg_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(jpeg_path.parent / ".matplotlib"))
    os.environ.setdefault("MPLBACKEND", "Agg")

    import matplotlib.pyplot as plt
    import networkx as nx

    graph = nx.DiGraph()
    for row in keep.itertuples(index=False):
        title_text = str(getattr(row, "title"))
        graph.add_node(
            title_text,
            entity_type=str(getattr(row, "type")),
            degree=int(getattr(row, "degree", 0) or 0),
        )
    for row in edges.itertuples(index=False):
        graph.add_edge(
            str(getattr(row, "source")),
            str(getattr(row, "target")),
            weight=float(getattr(row, "weight", 1.0) or 1.0),
        )

    fig, ax = plt.subplots(figsize=(18, 12), dpi=180)
    ax.set_title(title, fontsize=18, pad=16)
    ax.axis("off")

    if graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No clinical entities found", ha="center", va="center", fontsize=18)
    else:
        pos = nx.spring_layout(graph, seed=seed, k=1.4 / max(1, graph.number_of_nodes() ** 0.5))
        node_types = [graph.nodes[node].get("entity_type", "UNKNOWN") for node in graph.nodes]
        node_colors = [ENTITY_COLORS.get(entity_type, "#8f8f8f") for entity_type in node_types]
        node_sizes = [min(1800, 420 + graph.nodes[node].get("degree", 0) * 90) for node in graph.nodes]
        edge_widths = [min(4.0, 0.8 + float(data.get("weight", 1.0)) * 0.25) for _, _, data in graph.edges(data=True)]

        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=12,
            edge_color="#7b8794",
            alpha=0.5,
            width=edge_widths,
            connectionstyle="arc3,rad=0.06",
        )
        nx.draw_networkx_nodes(
            graph,
            pos,
            ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            linewidths=1.1,
            edgecolors="#ffffff",
            alpha=0.94,
        )
        labels = {node: (node[:42] + "..." if len(node) > 45 else node) for node in graph.nodes}
        nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=8, font_color="#1f2937")

        legend_types = sorted(set(node_types))
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=ENTITY_COLORS.get(entity_type, "#8f8f8f"),
                label=entity_type.replace("_", " ").title(),
                markersize=9,
            )
            for entity_type in legend_types
        ]
        ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=8)

    fig.tight_layout()
    fig.savefig(jpeg_path, format="jpg", pil_kwargs={"quality": 92})
    plt.close(fig)
    return jpeg_path
