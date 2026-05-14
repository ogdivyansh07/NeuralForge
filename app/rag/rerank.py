"""
Reranking placeholder: swap in a cross-encoder or LLM reranker later.

Current behavior: identity (preserve input order).
"""
from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


def rerank_chunks(_query: str, chunks: list[T], _scores: list[float] | None = None) -> list[T]:
    """Placeholder reranker — returns `chunks` unchanged."""
    return list(chunks)
