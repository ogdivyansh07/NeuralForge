from __future__ import annotations

from PyPDF2 import PdfReader


def load_pdf(path: str) -> str:
    """Load a PDF and return extracted text."""
    reader = PdfReader(path)
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def chunk_text(text: str, chunk_size: int = 300) -> list[str]:
    """Split text into chunks of roughly `chunk_size` words."""
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks
