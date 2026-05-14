"""app.rag — set NEURALFORGE_DEBUG_RAG_IMPORT=1 for extra submodule import tracing."""
from __future__ import annotations

import os

print("app.rag package initialized", flush=True)

if os.environ.get("NEURALFORGE_DEBUG_RAG_IMPORT"):
    print("[app.rag.__init__.py] loading submodule chain...", flush=True)

from .pdf_loader import load_pdf, load_pdf_pages
from .chunker import ChunkRecord, chunk_pages_semantic, chunk_text
from .embedder import (
    EMBEDDING_MODEL_NAME,
    embed_texts,
    embed_texts_safe,
    l2_normalize_rows,
)
from .retriever import (
    PDFAnswerBundle,
    RetrievalConfig,
    RetrievalHit,
    answer_query,
    answer_query_full,
    build_index,
    get_retrieval_config,
    load_index_bundle,
    retrieve_hits,
    save_index_bundle,
    search,
    set_chunks,
    set_retrieval_config,
)

if os.environ.get("NEURALFORGE_DEBUG_RAG_IMPORT"):
    print("[app.rag.__init__.py] imported answer_query_full:", answer_query_full, flush=True)

__all__ = [
    "load_pdf",
    "load_pdf_pages",
    "chunk_text",
    "chunk_pages_semantic",
    "ChunkRecord",
    "EMBEDDING_MODEL_NAME",
    "embed_texts",
    "embed_texts_safe",
    "l2_normalize_rows",
    "RetrievalConfig",
    "RetrievalHit",
    "PDFAnswerBundle",
    "build_index",
    "search",
    "retrieve_hits",
    "answer_query",
    "answer_query_full",
    "set_chunks",
    "set_retrieval_config",
    "get_retrieval_config",
    "save_index_bundle",
    "load_index_bundle",
]
