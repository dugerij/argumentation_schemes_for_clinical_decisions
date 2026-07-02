import re
from math import sqrt
from typing import Any

from retrieval.index import build_embed_model, build_llm


RELATION_STRING_RE = re.compile(
    r"^(?P<source>[0-9a-fA-F-]{32,})\s+->\s+(?P<label>[A-Z_]+)\s+->\s+(?P<target>[0-9a-fA-F-]{32,})$"
)


def _compact_text(text: str, limit: int = 700) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."


def _unwrap_source_node(source_node: Any) -> Any:
    return getattr(source_node, "node", None) or source_node


def _node_metadata(node: Any) -> dict[str, Any]:
    metadata = getattr(node, "metadata", None)
    if isinstance(metadata, dict):
        return metadata
    properties = getattr(node, "properties", None)
    if isinstance(properties, dict):
        return properties
    return {}


def _source_label(node: Any) -> str:
    metadata = _node_metadata(node)
    source_name = metadata.get("source_name")
    if source_name:
        return f" [{source_name}]"
    return ""


def _node_title(node: Any) -> str:
    if node is None:
        return ""

    for attr in ("name", "label"):
        value = getattr(node, attr, None)
        if value and str(value) not in {"entity", "text_chunk"}:
            return str(value)

    text = getattr(node, "text", None)
    if text:
        return _compact_text(str(text), limit=140)

    if hasattr(node, "get_content"):
        content = node.get_content()
        if content:
            return _compact_text(content, limit=140)

    node_id = getattr(node, "id", None) or getattr(node, "id_", None)
    return str(node_id or node)


def _graph_store(index: Any) -> Any | None:
    graph_store = getattr(index, "property_graph_store", None)
    if graph_store is not None:
        return graph_store

    storage_context = getattr(index, "storage_context", None)
    if storage_context is not None:
        return getattr(storage_context, "property_graph_store", None)

    return None


def _graph_node_by_id(graph_store: Any | None, node_id: str) -> Any | None:
    if graph_store is None or not node_id:
        return None
    try:
        nodes = graph_store.get(ids=[node_id])
    except Exception:
        return None
    return nodes[0] if nodes else None


def _relation_text(node: Any, graph_store: Any | None) -> str | None:
    source_id = getattr(node, "source_id", None)
    target_id = getattr(node, "target_id", None)
    label = getattr(node, "label", None)

    if not (source_id and target_id and label):
        match = RELATION_STRING_RE.match(str(node).strip())
        if not match:
            return None
        source_id = match.group("source")
        target_id = match.group("target")
        label = match.group("label")

    source = _graph_node_by_id(graph_store, str(source_id))
    target = _graph_node_by_id(graph_store, str(target_id))
    source_text = _node_title(source) or str(source_id)
    target_text = _node_title(target) or str(target_id)
    return f"Relation: {source_text} -> {label} -> {target_text}"


def _source_node_text(source_node: Any, graph_store: Any | None = None) -> str:
    node = _unwrap_source_node(source_node)

    relation = _relation_text(node, graph_store)
    if relation:
        return relation

    if hasattr(node, "get_content"):
        content = node.get_content()
        if content and content.strip():
            return f"Text{_source_label(node)}: {_compact_text(content)}"

    text = getattr(node, "text", None)
    if text and str(text).strip():
        return f"Text{_source_label(node)}: {_compact_text(str(text))}"

    return _compact_text(str(node))


def _source_node_score(source_node: Any) -> float | None:
    score = getattr(source_node, "score", None)
    if score is None:
        return None
    try:
        return float(score)
    except (TypeError, ValueError):
        return None


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _norm(values: list[float]) -> float:
    return sqrt(sum(value * value for value in values))


def _cosine_similarity(left: list[float], right: list[float]) -> float | None:
    if not left or not right or len(left) != len(right):
        return None
    left_norm = _norm(left)
    right_norm = _norm(right)
    if left_norm == 0 or right_norm == 0:
        return None
    return _dot(left, right) / (left_norm * right_norm)


def _embed_text(embed_model: Any, text: str) -> list[float]:
    if hasattr(embed_model, "get_text_embedding"):
        return list(embed_model.get_text_embedding(text))
    if hasattr(embed_model, "embed"):
        embedded = embed_model.embed(text)
        if isinstance(embedded, list):
            return [float(value) for value in embedded]
    raise AttributeError("Embedding model does not expose a supported embedding method.")


async def query_index_with_diagnostics(
    index,
    query: str,
    similarity_top_k: int = 5,
    llm: Any | None = None,
    embed_model: Any | None = None,
) -> tuple[str, str, dict[str, Any]]:
    index_llm = getattr(index, "_llm", None)
    query_engine = index.as_query_engine(
        similarity_top_k=similarity_top_k,
        llm=llm or index_llm or build_llm(),
    )
    response = query_engine.query(query)
    source_nodes = getattr(response, "source_nodes", None) or []
    graph_store = _graph_store(index)
    context_parts = []
    retrieval_scores: list[float] = []
    for node in source_nodes:
        text = _source_node_text(node, graph_store=graph_store)
        if text.strip():
            context_parts.append(text)
        score = _source_node_score(node)
        if score is not None:
            retrieval_scores.append(score)

    context = "\n\n".join(context_parts) if context_parts else str(response)

    cosine_similarities: list[float] = []
    if source_nodes:
        resolved_embed_model = embed_model or getattr(index, "_embed_model", None) or build_embed_model()
        query_embedding = _embed_text(resolved_embed_model, query)
        for node in source_nodes:
            node_text = _source_node_text(node, graph_store=graph_store)
            if not node_text.strip():
                continue
            similarity = _cosine_similarity(query_embedding, _embed_text(resolved_embed_model, node_text))
            if similarity is not None:
                cosine_similarities.append(similarity)

    diagnostics = {
        "source_count": len(source_nodes),
        "retrieval_scores": retrieval_scores,
        "cosine_similarities": cosine_similarities,
        "top_retrieval_score": retrieval_scores[0] if retrieval_scores else None,
        "mean_retrieval_score": (sum(retrieval_scores) / len(retrieval_scores)) if retrieval_scores else None,
        "retrieval_score_margin": (
            retrieval_scores[0] - retrieval_scores[1]
            if len(retrieval_scores) >= 2
            else None
        ),
        "top_cosine_similarity": cosine_similarities[0] if cosine_similarities else None,
        "mean_cosine_similarity": (sum(cosine_similarities) / len(cosine_similarities)) if cosine_similarities else None,
        "cosine_similarity_margin": (
            cosine_similarities[0] - cosine_similarities[1]
            if len(cosine_similarities) >= 2
            else None
        ),
    }
    return str(response), context, diagnostics


async def query_index_context(
    index,
    query: str,
    similarity_top_k: int = 5,
    llm: Any | None = None,
) -> tuple[str, str]:
    response, context, _diagnostics = await query_index_with_diagnostics(
        index=index,
        query=query,
        similarity_top_k=similarity_top_k,
        llm=llm,
    )
    return str(response), context
