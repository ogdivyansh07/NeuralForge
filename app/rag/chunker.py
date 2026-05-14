"""
Chunk PDF text with character windows, sentence-boundary snapping, and overlap.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 50


@dataclass
class ChunkRecord:
    """One retrieval unit with provenance for citations and UI."""

    text: str
    chunk_id: str
    page_number: int
    document_id: str = "default"


def _normalize(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def _snap_end_to_sentence(text_window: str, min_keep: int) -> int:
    """Return end index within text_window preferring last .?! after min_keep."""
    if len(text_window) <= min_keep:
        return len(text_window)
    sub = text_window[min_keep:]
    best = -1
    for m in re.finditer(r"[.!?](?:\s|$)", sub):
        best = min_keep + m.end()
    return best if best > min_keep else len(text_window)


def chunk_page_text(
    text: str,
    page_number: int,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    document_id: str = "default",
) -> list[ChunkRecord]:
    """
    Build overlapping character windows (~chunk_size), snapped to sentence ends when sensible.
    """
    text = _normalize(text)
    if not text:
        return []

    records: list[ChunkRecord] = []
    start = 0
    n = len(text)
    min_snap = max(40, chunk_size // 3)

    while start < n:
        end = min(n, start + chunk_size)
        window = text[start:end]
        if end < n:
            snapped = _snap_end_to_sentence(window, min_snap)
            if snapped < len(window):
                end = start + snapped
        piece = text[start:end].strip()
        if piece:
            records.append(
                ChunkRecord(
                    text=piece,
                    chunk_id=str(uuid.uuid4())[:12],
                    page_number=page_number,
                    document_id=document_id,
                )
            )
        if end >= n:
            break
        start = max(start + 1, end - overlap)

    return records


def chunk_pages_semantic(
    pages: list,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    document_id: str = "default",
) -> list[ChunkRecord]:
    """Chunk each page separately so page_number metadata stays accurate."""
    all_records: list[ChunkRecord] = []
    for pg in pages:
        page_no = int(getattr(pg, "page_number", 0) or 0)
        raw = getattr(pg, "text", "") or ""
        all_records.extend(
            chunk_page_text(
                raw,
                page_no,
                chunk_size=chunk_size,
                overlap=overlap,
                document_id=document_id,
            )
        )
    return all_records


def chunk_text(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    document_id: str = "default",
) -> list[str]:
    """Backward-compatible: single blob of text (page 0) → chunk strings."""
    recs = chunk_page_text(text, 0, chunk_size=chunk_size, overlap=overlap, document_id=document_id)
    return [r.text for r in recs if r.text.strip()]
