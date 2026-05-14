"""
Backward-compatible imports for PDF loading and chunking.

Prefer importing from `pdf_loader` and `chunker` in new code.
"""
from __future__ import annotations

from .chunker import ChunkRecord, chunk_pages_semantic, chunk_text
from .pdf_loader import PageText, load_pdf, load_pdf_pages

__all__ = [
    "ChunkRecord",
    "PageText",
    "chunk_pages_semantic",
    "chunk_text",
    "load_pdf",
    "load_pdf_pages",
]
