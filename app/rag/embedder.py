from __future__ import annotations

import os

import numpy as np
from sentence_transformers import SentenceTransformer

if os.environ.get("NEURALFORGE_DEBUG_RAG_IMPORT"):
    print("[embedder.py] module loaded", flush=True)

# Semantic retrieval model (384-dim); cosine search uses L2-normalized vectors.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_MODEL: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Load and cache SentenceTransformer (MiniLM)."""
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _MODEL


def l2_normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """Unit L2 norm per row so dot product equals cosine similarity."""
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (arr / norms).astype(np.float32)


def embed_texts(chunks: list[str], *, normalize_for_cosine: bool = True) -> np.ndarray:
    """
    Encode texts with all-MiniLM-L6-v2.

    When normalize_for_cosine is True (default), rows are L2-normalized for
    cosine similarity via inner product (e.g. FAISS IndexFlatIP).
    """
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    model = get_model()
    vectors = model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    if normalize_for_cosine:
        vectors = l2_normalize_rows(vectors)
    return vectors


def embed_texts_safe(chunks: list[str], *, normalize_for_cosine: bool = True) -> np.ndarray:
    """
    Same as `embed_texts` but wraps failures in RuntimeError with a clear message.
    Callers (e.g. Streamlit) should catch and display a user-facing error.
    """
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    try:
        return embed_texts(chunks, normalize_for_cosine=normalize_for_cosine)
    except Exception as exc:
        raise RuntimeError(f"Embedding failed ({EMBEDDING_MODEL_NAME}): {exc}") from exc
