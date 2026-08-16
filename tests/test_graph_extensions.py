from dataclasses import replace

import pytest

from clinical_cds.direct import load_direct_dataset
from clinical_cds.graph_extensions import (
    extend_direct_dataset,
    load_project_graph_extensions,
)


def test_gastritis_extension_has_complete_source_provenance():
    extensions = load_project_graph_extensions()

    assert len(extensions) == 1
    extension = extensions[0]
    assert extension.extension_id == "gastritis-kg-extension-v1"
    assert extension.graph.category == "Gastritis"
    assert set(extension.graph.leaf_labels) == {
        "Acute Gastritis",
        "Chronic Non-atrophic Gastritis",
        "Chronic Atrophic Gastritis",
    }
    premise_ids = {
        node.node_id
        for node in extension.graph.nodes
        if node.kind == "premise"
    }
    assert set(extension.premise_sources) == premise_ids
    assert all(extension.premise_sources.values())
    assert all(
        node.knowledge_source_ids == extension.premise_sources[node.node_id]
        for node in extension.graph.nodes
        if node.kind == "premise"
    )
    assert len(extension.provenance_sha256) == 64


def test_gastritis_extension_closes_direct_graph_gap(direct_root):
    direct = load_direct_dataset(direct_root)
    extension = load_project_graph_extensions()[0]
    gastritis_case = replace(
        direct.cases[0],
        case_id="gastritis-case",
        disease_category="Gastritis",
        gold_label="Chronic Non-atrophic Gastritis",
    )
    direct = replace(direct, cases=(gastritis_case,))

    extended = extend_direct_dataset(direct, (extension,))

    assert len(extended.graphs) == len(direct.graphs) + 1
    assert extended.audit.missing_graph_categories == ()
    assert extended.audit.conclusions_outside_graph == ()


def test_graph_extension_refuses_category_collision(direct_root):
    direct = load_direct_dataset(direct_root)
    extension = load_project_graph_extensions()[0]
    collision = replace(
        extension,
        graph=replace(
            extension.graph,
            category=direct.graphs[0].category,
        ),
    )

    with pytest.raises(ValueError, match="collide"):
        extend_direct_dataset(direct, (collision,))
