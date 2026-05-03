from __future__ import annotations

from dotenv import load_dotenv
import os
import numpy as np
import faiss
from google import genai

from .embedder import embed_texts

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("API key missing")
client = genai.Client(api_key=api_key)


_INDEX: faiss.Index | None = None
_CHUNKS: list[str] = []


def build_index(embeddings):
    """Build a FAISS L2 index from embeddings."""
    global _INDEX

    vectors = np.asarray(embeddings, dtype="float32")
    if vectors.ndim != 2:
        raise ValueError("embeddings must be a 2D array-like")

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    _INDEX = index
    return index


def set_chunks(chunks: list[str]) -> None:
    """Store chunks for retrieval output."""
    global _CHUNKS
    _CHUNKS = chunks


def search(query_embedding, top_k: int = 3) -> list[str]:
    """Search nearest chunks for a query embedding."""
    if _INDEX is None:
        raise RuntimeError("Index is not built. Call build_index first.")

    query_vec = np.asarray(query_embedding, dtype="float32")
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)

    _, indices = _INDEX.search(query_vec, top_k)

    results: list[str] = []
    for idx in indices[0]:
        if 0 <= idx < len(_CHUNKS):
            results.append(_CHUNKS[idx])
    return results


def fallback_answer(query: str) -> str:
    import requests

    url = "https://api.duckduckgo.com/"
    params = {
        "q": query,
        "format": "json",
    }
    try:
        res = requests.get(url, params=params, timeout=10).json()
        return res.get("AbstractText", "No answer found") or "No answer found"
    except Exception:
        return "No answer found"


def generate_answer(context: str, query: str) -> str:
    prompt = f"""
    Answer the question using the context below.

    Context:
    {context}

    Question:
    {query}

    Give a clear and correct answer.
    """

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return response.text


def answer_query(query: str) -> str:
    """
    Retrieve top chunks and generate an answer from context.
    """
    if _INDEX is None:
        return "No relevant answer found in document."

    query_embedding = embed_texts([query])
    query_vec = np.asarray(query_embedding, dtype="float32")
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)

    top_k = 3
    distances, indices = _INDEX.search(query_vec, top_k)
    # Convert L2 distance to a simple [0, 1]-like similarity score.
    similarities = 1.0 / (1.0 + distances[0])
    threshold = 0.3

    if similarities.size == 0 or float(similarities[0]) < threshold:
        return fallback_answer(query)

    top_chunks: list[str] = []
    for idx, score in zip(indices[0], similarities):
        if score >= threshold and 0 <= idx < len(_CHUNKS):
            top_chunks.append(_CHUNKS[idx])

    if not top_chunks:
        return fallback_answer(query)

    relevant_chunks = top_chunks
    context = "\n".join(relevant_chunks)
    answer = generate_answer(context, query)
    if not answer:
        return "No relevant answer found in document."
    return answer
