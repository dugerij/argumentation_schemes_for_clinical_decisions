import json

from retrieval.visualize import _load_property_store_hint_graph, _select_graph_subset


def test_property_store_loader_returns_expected_empty_schema(tmp_path):
    output_dir = tmp_path / "graph"
    output_dir.mkdir()
    (output_dir / "property_graph_store.json").write_text(
        json.dumps(
            {
                "nodes": {
                    "node-1": {
                        "label": "text_chunk",
                        "text": "plain note text without umls bullets",
                        "properties": {"source_name": "sample note"},
                    }
                },
                "relations": {},
                "triplets": [],
            }
        ),
        encoding="utf-8",
    )

    entities, relationships = _load_property_store_hint_graph(output_dir)

    assert list(entities.columns) == ["title", "type", "description", "frequency", "degree"]
    assert entities.empty
    assert list(relationships.columns) == ["source", "target", "description", "weight", "combined_degree"]
    assert relationships.empty


def test_select_graph_subset_handles_empty_property_store(tmp_path):
    output_dir = tmp_path / "graph"
    output_dir.mkdir()
    (output_dir / "property_graph_store.json").write_text(
        json.dumps({"nodes": {}, "relations": {}, "triplets": []}),
        encoding="utf-8",
    )

    keep, edges = _select_graph_subset(output_dir, source="property_store")

    assert keep.empty
    assert list(keep.columns) == ["title", "type", "description", "frequency", "degree"]
    assert edges.empty
    assert list(edges.columns) == ["source", "target", "description", "weight", "combined_degree"]
