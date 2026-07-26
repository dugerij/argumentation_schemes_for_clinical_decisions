from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from clinical_cds.schema import (
    AnnotationEdge,
    AnnotationNode,
    ClinicalCase,
    DiagnosticGraph,
    GraphEdge,
    GraphNode,
)


INPUT_SECTIONS = {
    "input1": "chief_complaint",
    "input2": "history_of_present_illness",
    "input3": "past_medical_history",
    "input4": "family_history",
    "input5": "physical_exam",
    "input6": "pertinent_results",
}
ANNOTATION_SUFFIX_RE = re.compile(
    r"^(?P<text>.*)\$(?P<kind>Input[1-6]|Cause|Intermedia)(?:_(?P<index>\d+))?$",
    flags=re.DOTALL,
)


@dataclass(frozen=True)
class DirectAudit:
    case_count: int
    graph_count: int
    disease_category_count: int
    diagnosis_count: int
    empty_section_count: int
    folder_conclusion_mismatch_count: int
    missing_graph_categories: tuple[str, ...]
    conclusions_outside_graph: tuple[str, ...]
    quality_flag_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_count": self.case_count,
            "graph_count": self.graph_count,
            "disease_category_count": self.disease_category_count,
            "diagnosis_count": self.diagnosis_count,
            "empty_section_count": self.empty_section_count,
            "folder_conclusion_mismatch_count": self.folder_conclusion_mismatch_count,
            "missing_graph_categories": list(self.missing_graph_categories),
            "conclusions_outside_graph": list(self.conclusions_outside_graph),
            "quality_flag_counts": dict(sorted(self.quality_flag_counts.items())),
        }


@dataclass(frozen=True)
class DirectDataset:
    cases: tuple[ClinicalCase, ...]
    graphs: tuple[DiagnosticGraph, ...]
    audit: DirectAudit


def normalize_label(value: str) -> str:
    return " ".join(value.replace("\r", " ").replace("\n", " ").split()).strip()


def label_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_label(value).casefold())


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", normalize_label(value).casefold()).strip("-")
    return slug or "unknown"


def _stable_id(prefix: str, value: str, length: int = 12) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}-{digest}"


def _split_premises(value: object) -> list[str]:
    if value is None:
        return []
    parts = re.split(r";+", str(value))
    return [normalize_label(part) for part in parts if normalize_label(part)]


def _resolve_direct_root(path: Path) -> Path:
    candidates = [
        path,
        path / "unpacked",
        path / "mimic_iv_ext_direct" / "unpacked",
    ]
    for candidate in candidates:
        if (
            (candidate / "Finished").is_dir()
            and (candidate / "Diagnosis_flowchart").is_dir()
        ):
            return candidate
    raise FileNotFoundError(
        f"Could not find Finished and Diagnosis_flowchart under {path}."
    )


def prepare_direct_archive(
    archive_path: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> Path:
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    if not archive_path.exists():
        raise FileNotFoundError(f"Missing DiReCT archive: {archive_path}")

    ready_root = output_dir / "unpacked"
    if (ready_root / "Finished").is_dir() and (ready_root / "Diagnosis_flowchart").is_dir():
        if not overwrite:
            return ready_root

    extracted = output_dir / "archive"
    extracted.mkdir(parents=True, exist_ok=True)
    ready_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extracted)

    rar_files = sorted(extracted.rglob("*.rar"))
    if len(rar_files) != 2:
        raise ValueError(
            f"Expected two RAR files in the DiReCT archive, found {len(rar_files)}."
        )
    extractor = shutil.which("bsdtar") or shutil.which("unar")
    if extractor is None:
        raise RuntimeError(
            "Extracting DiReCT requires bsdtar or unar. Install one and rerun prepare-direct."
        )
    for rar_path in rar_files:
        if Path(extractor).name == "unar":
            command = [extractor, "-f", "-o", str(ready_root), str(rar_path)]
        else:
            command = [extractor, "-xf", str(rar_path), "-C", str(ready_root)]
        subprocess.run(command, check=True, capture_output=True, text=True)
    return _resolve_direct_root(ready_root)


def _diagnosis_node_id(category: str, label: str) -> str:
    return f"diagnosis:{_slug(category)}:{_stable_id('d', label_key(label), 10)}"


def _compile_diagnostic_tree(
    category: str,
    tree: object,
) -> tuple[list[GraphNode], list[GraphEdge], dict[str, tuple[str, ...]]]:
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    paths: dict[str, tuple[str, ...]] = {}
    seen: set[str] = set()

    def visit(branch: object, parent_id: str | None, path: tuple[str, ...]) -> None:
        if not isinstance(branch, dict):
            return
        for raw_label, children in branch.items():
            label = normalize_label(str(raw_label))
            node_id = _diagnosis_node_id(category, label)
            current_path = path + (label,)
            if node_id not in seen:
                seen.add(node_id)
                nodes.append(
                    GraphNode(
                        node_id=node_id,
                        label=label,
                        kind="diagnosis",
                        category=category,
                    )
                )
            paths[label_key(label)] = current_path
            if parent_id is not None:
                edges.append(
                    GraphEdge(
                        source_id=parent_id,
                        target_id=node_id,
                        relation="refines_to",
                    )
                )
            visit(children, node_id, current_path)

    visit(tree, None, ())
    return nodes, edges, paths


def load_direct_graph(path: Path) -> DiagnosticGraph:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(payload) != {"diagnostic", "knowledge"}:
        raise ValueError(f"Unexpected DiReCT graph keys in {path}: {sorted(payload)}")

    category = normalize_label(Path(path).stem)
    nodes, edges, paths = _compile_diagnostic_tree(category, payload["diagnostic"])
    diagnosis_ids = {
        label_key(node.label): node.node_id
        for node in nodes
        if node.kind == "diagnosis"
    }

    for raw_diagnosis, knowledge in payload["knowledge"].items():
        diagnosis = normalize_label(str(raw_diagnosis))
        diagnosis_id = diagnosis_ids.get(label_key(diagnosis))
        if diagnosis_id is None:
            diagnosis_id = _diagnosis_node_id(category, diagnosis)
            diagnosis_ids[label_key(diagnosis)] = diagnosis_id
            nodes.append(
                GraphNode(
                    node_id=diagnosis_id,
                    label=diagnosis,
                    kind="diagnosis",
                    category=category,
                )
            )
            paths[label_key(diagnosis)] = (diagnosis,)

        premise_groups = (
            knowledge
            if isinstance(knowledge, dict)
            else {"Diagnostic Criteria": knowledge}
        )
        for raw_type, raw_text in premise_groups.items():
            premise_type = normalize_label(str(raw_type))
            for premise_text in _split_premises(raw_text):
                identity = f"{category}|{diagnosis}|{premise_type}|{premise_text}"
                premise_id = f"premise:{_slug(category)}:{_stable_id('p', identity, 12)}"
                nodes.append(
                    GraphNode(
                        node_id=premise_id,
                        label=premise_text,
                        kind="premise",
                        category=category,
                        text=premise_text,
                        premise_type=premise_type,
                        diagnosis_label=diagnosis,
                    )
                )
                edges.append(
                    GraphEdge(
                        source_id=premise_id,
                        target_id=diagnosis_id,
                        relation="supports",
                    )
                )

    return DiagnosticGraph(
        graph_id=f"direct:{_slug(category)}",
        category=category,
        nodes=tuple(nodes),
        edges=tuple(edges),
        diagnostic_paths=paths,
    )


def load_direct_graphs(root: Path) -> tuple[DiagnosticGraph, ...]:
    direct_root = _resolve_direct_root(Path(root))
    paths = sorted((direct_root / "Diagnosis_flowchart").glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No DiReCT guideline graphs found under {direct_root}.")
    return tuple(load_direct_graph(path) for path in paths)


def _parse_annotation_key(raw_key: str) -> tuple[str, str, str | None, int | None]:
    match = ANNOTATION_SUFFIX_RE.match(raw_key)
    if match is None:
        raise ValueError(f"Invalid DiReCT annotation key: {raw_key!r}")
    text = normalize_label(match.group("text"))
    kind = match.group("kind")
    index = int(match.group("index")) if match.group("index") else None
    if kind.startswith("Input"):
        section = INPUT_SECTIONS[kind.casefold()]
        return text, "observation", section, index
    if kind == "Cause":
        return text, "rationale", None, index
    return text, "diagnosis", None, index


def _parse_annotation(
    case_id: str,
    root_key: str,
    root_children: object,
) -> tuple[tuple[AnnotationNode, ...], tuple[AnnotationEdge, ...], str]:
    nodes: list[AnnotationNode] = []
    edges: list[AnnotationEdge] = []
    counter = 0

    def visit(raw_key: str, children: object, parent_id: str | None) -> str:
        nonlocal counter
        counter += 1
        node_id = f"{case_id}:a{counter:04d}"
        text, role, source_section, annotation_index = _parse_annotation_key(raw_key)
        nodes.append(
            AnnotationNode(
                node_id=node_id,
                text=text,
                role=role,
                source_section=source_section,
                annotation_index=annotation_index,
            )
        )
        if parent_id is not None:
            edges.append(
                AnnotationEdge(
                    source_id=node_id,
                    target_id=parent_id,
                    relation="supports",
                )
            )
        if not isinstance(children, dict):
            raise ValueError(f"Annotation children must be an object for {raw_key!r}.")
        for child_key, grand_children in children.items():
            visit(str(child_key), grand_children, node_id)
        return node_id

    visit(root_key, root_children, None)
    conclusion, role, _, _ = _parse_annotation_key(root_key)
    if role != "diagnosis":
        raise ValueError(f"DiReCT annotation root is not a diagnosis: {root_key!r}")
    return tuple(nodes), tuple(edges), conclusion


def _canonical_label(value: str, graphs: Iterable[DiagnosticGraph]) -> str:
    labels = {
        label_key(node.label): node.label
        for graph in graphs
        for node in graph.nodes
        if node.kind == "diagnosis"
    }
    return labels.get(label_key(value), normalize_label(value))


def _load_direct_case(
    path: Path,
    sample_root: Path,
    graphs: tuple[DiagnosticGraph, ...],
) -> ClinicalCase:
    payload = json.loads(path.read_text(encoding="utf-8"))
    input_keys = {key for key in payload if key.startswith("input")}
    if input_keys != set(INPUT_SECTIONS):
        raise ValueError(f"Expected input1-input6 in {path}, found {sorted(input_keys)}")
    root_items = [(key, value) for key, value in payload.items() if key not in INPUT_SECTIONS]
    if len(root_items) != 1:
        raise ValueError(f"Expected one annotation root in {path}, found {len(root_items)}")

    relative = path.relative_to(sample_root)
    disease_category = normalize_label(relative.parts[0])
    directory_label = normalize_label(relative.parts[-2])
    case_id = _stable_id("direct", relative.as_posix())
    root_key, root_children = root_items[0]
    annotation_nodes, annotation_edges, conclusion = _parse_annotation(
        case_id,
        str(root_key),
        root_children,
    )
    gold_label = _canonical_label(conclusion, graphs)
    if label_key(gold_label) == label_key(directory_label):
        gold_label = directory_label
    sections = {
        section_name: normalize_label(str(payload[input_key]))
        for input_key, section_name in INPUT_SECTIONS.items()
        if normalize_label(str(payload[input_key]))
    }

    graph_by_category = {label_key(graph.category): graph for graph in graphs}
    graph = graph_by_category.get(label_key(disease_category))
    graph_labels = {
        label_key(node.label)
        for node in graph.nodes
        if node.kind == "diagnosis"
    } if graph else set()
    graph_leaves = {label_key(label) for label in graph.leaf_labels} if graph else set()

    flags: list[str] = []
    if graph is None:
        flags.append("missing_guideline_graph")
    elif label_key(gold_label) not in graph_labels:
        flags.append("conclusion_outside_category_graph")
    elif label_key(gold_label) not in graph_leaves:
        flags.append("conclusion_not_leaf")
    if label_key(directory_label) != label_key(gold_label):
        flags.append("directory_conclusion_mismatch")
    if len(sections) != len(INPUT_SECTIONS):
        flags.append("empty_input_section")

    return ClinicalCase(
        case_id=case_id,
        dataset="direct",
        task="diagnosis",
        sections=sections,
        gold_label=gold_label,
        disease_category=disease_category,
        directory_label=directory_label,
        annotation_nodes=annotation_nodes,
        annotation_edges=annotation_edges,
        metadata={
            "annotation_conclusion": conclusion,
            "section_count": len(sections),
        },
        quality_flags=tuple(flags),
    )


def load_direct_dataset(root: Path) -> DirectDataset:
    direct_root = _resolve_direct_root(Path(root))
    graphs = load_direct_graphs(direct_root)
    sample_root = direct_root / "Finished"
    case_paths = sorted(sample_root.rglob("*.json"))
    if not case_paths:
        raise FileNotFoundError(f"No DiReCT cases found under {sample_root}.")
    cases = tuple(_load_direct_case(path, sample_root, graphs) for path in case_paths)

    graph_categories = {label_key(graph.category) for graph in graphs}
    diagnosis_labels = {
        label_key(node.label)
        for graph in graphs
        for node in graph.nodes
        if node.kind == "diagnosis"
    }
    missing_graph_categories = sorted(
        {
            case.disease_category
            for case in cases
            if case.disease_category and label_key(case.disease_category) not in graph_categories
        }
    )
    outside_by_key = {
        label_key(case.gold_label): case.gold_label
        for case in cases
        if label_key(case.gold_label) not in diagnosis_labels
    }
    conclusions_outside_graph = sorted(outside_by_key.values())
    flag_counts = Counter(flag for case in cases for flag in case.quality_flags)
    audit = DirectAudit(
        case_count=len(cases),
        graph_count=len(graphs),
        disease_category_count=len({case.disease_category for case in cases}),
        diagnosis_count=len({label_key(case.gold_label) for case in cases}),
        empty_section_count=sum(len(INPUT_SECTIONS) - len(case.sections) for case in cases),
        folder_conclusion_mismatch_count=flag_counts["directory_conclusion_mismatch"],
        missing_graph_categories=tuple(missing_graph_categories),
        conclusions_outside_graph=tuple(conclusions_outside_graph),
        quality_flag_counts=dict(flag_counts),
    )
    return DirectDataset(cases=cases, graphs=graphs, audit=audit)


def select_direct_partition(
    cases: Iterable[ClinicalCase],
    partition: str,
    *,
    development_fraction: float = 0.2,
    seed: int = 17,
) -> tuple[ClinicalCase, ...]:
    normalized_partition = partition.strip().casefold()
    case_list = tuple(cases)
    if normalized_partition == "all":
        return case_list
    if normalized_partition not in {"development", "test"}:
        raise ValueError("DiReCT partition must be all, development, or test.")
    if not 0.0 < development_fraction < 1.0:
        raise ValueError("development_fraction must be between zero and one.")

    grouped: dict[str, list[ClinicalCase]] = {}
    for case in case_list:
        grouped.setdefault(label_key(case.gold_label), []).append(case)

    development_ids: set[str] = set()
    for diagnosis, diagnosis_cases in grouped.items():
        ranked = sorted(
            diagnosis_cases,
            key=lambda case: hashlib.sha256(
                f"{seed}|{diagnosis}|{case.case_id}".encode("utf-8")
            ).hexdigest(),
        )
        development_count = (
            max(1, round(len(ranked) * development_fraction))
            if len(ranked) >= 5
            else 0
        )
        development_ids.update(case.case_id for case in ranked[:development_count])

    if normalized_partition == "development":
        return tuple(case for case in case_list if case.case_id in development_ids)
    return tuple(case for case in case_list if case.case_id not in development_ids)
