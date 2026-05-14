"""
PDF ingestion: extract text per page for chunk metadata (page numbers).
"""
from __future__ import annotations

from dataclasses import dataclass

from PyPDF2 import PdfReader


@dataclass(frozen=True)
class PageText:
    """One PDF page: 1-based page index for display, raw extracted text."""

    page_number: int
    text: str


def load_pdf_pages(path: str) -> list[PageText]:
    """
    Load a PDF and return ordered page texts.

    Raises:
        FileNotFoundError: if path does not exist.
        ValueError: if PDF has no extractable pages or all pages empty.
    """
    reader = PdfReader(path)
    if len(reader.pages) == 0:
        raise ValueError("PDF has no pages.")

    pages: list[PageText] = []
    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        pages.append(PageText(page_number=i, text=raw))

    if not any(p.text.strip() for p in pages):
        raise ValueError("PDF appears empty (no extractable text).")

    return pages


def load_pdf(path: str) -> str:
    """
    Load a PDF and return full document text (joined pages).

    Backward-compatible API for callers that only need a single string.
    """
    parts = [p.text for p in load_pdf_pages(path)]
    return "\n".join(parts).strip()
