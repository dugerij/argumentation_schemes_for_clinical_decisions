"""Bounded Microsoft GraphRAG fallback preparation utilities."""

from .corpus import (
    FALLBACK_ID,
    GRAPHRAG_VERSION,
    build_controlled_corpus,
    prepare_fallback_workspace,
)

__all__ = [
    "FALLBACK_ID",
    "GRAPHRAG_VERSION",
    "build_controlled_corpus",
    "prepare_fallback_workspace",
]
