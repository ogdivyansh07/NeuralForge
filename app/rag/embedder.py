from __future__ import annotations

from sentence_transformers import SentenceTransformer


_MODEL: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Load and cache the embedding model."""
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def embed_texts(chunks: list[str]):
    """Embed text chunks into dense vectors."""
    if not chunks:
        return []
    model = get_model()
    return model.encode(chunks, convert_to_numpy=True)
