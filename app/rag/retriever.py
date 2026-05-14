from __future__ import annotations

print("retriever import started", flush=True)

import json
import os

if os.environ.get("NEURALFORGE_DEBUG_RAG_IMPORT"):
    print("[retriever.py] module load start", flush=True)
import re
from dataclasses import dataclass, replace

try:
    import faiss
    import numpy as np
except Exception as exc:
    print(f"[retriever] failed importing faiss/numpy: {exc!r}", flush=True)
    raise

try:
    from .embedder import embed_texts, l2_normalize_rows
    from .generator import NOT_FOUND_MESSAGE, gemini_failed, generate_answer
    from .hybrid import SimpleBM25, fuse_semantic_bm25
    from .memory import append_turn
    from .rerank import rerank_chunks
except Exception as exc:
    print(f"[retriever] failed importing app.rag submodules: {exc!r}", flush=True)
    raise

if os.environ.get("NEURALFORGE_DEBUG_RAG_IMPORT"):
    print("[retriever.py] imports complete (embedder, generator, hybrid, memory, rerank)", flush=True)

# --- Defaults (override via set_retrieval_config) ---
@dataclass
class RetrievalConfig:
    """Tunable retrieval knobs for production deployments."""

    top_k: int = 3
    similarity_threshold: float = 0.26
    use_hybrid: bool = False
    hybrid_alpha: float = 0.72
    max_candidates: int = 24


@dataclass
class RetrievalHit:
    """One retrieved chunk with score and metadata."""

    text: str
    score: float
    chunk_id: str = ""
    page_number: int = 0
    document_id: str = "default"


@dataclass
class PDFAnswerBundle:
    """Structured QA result for UI (Streamlit) without breaking string API."""

    answer_text: str
    best_similarity: float | None
    retrieved_previews: list[str]
    source_meta: list[dict[str, str | int]]


_INDEX: faiss.Index | None = None
_CHUNKS: list[str] = []
_CHUNK_METADATA: list[dict[str, str | int]] = []
_CHUNK_MATRIX: np.ndarray | None = None
_RETRIEVAL_CONFIG = RetrievalConfig()

# --- Retrieval / generation tuning ---

# Sentence quality (semantic cleaning)
_MIN_SENTENCE_CHARS = 25
_MAX_COMMA_COUNT = 8
_MAX_NON_ALNUM_RATIO = 0.38

# Fallback answer shape
_FALLBACK_TOP_SENTENCES = 4
_FALLBACK_MIN_KEYWORD_HITS = 1
_MAX_FALLBACK_WORDS = 160

# Tighter context for Gemini (same API, less noise)
_LLM_CONTEXT_MAX_SENTENCES = 7
_LLM_CONTEXT_MAX_WORDS = 280

_ANSWER_HEADER = "📘 Answer from uploaded PDF:"
# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _word_count(text: str) -> int:
    return len(re.findall(r"[a-zA-Z0-9]+(?:'[a-zA-Z]+)?", text))


def _truncate_to_max_words(text: str, max_words: int) -> str:
    tokens = text.split()
    if len(tokens) <= max_words:
        return text.strip()
    trimmed = " ".join(tokens[:max_words]).strip()
    if trimmed and trimmed[-1] not in ".!?":
        trimmed += "."
    return trimmed


# ---------------------------------------------------------------------------
# Query keywords & sentence scoring
# ---------------------------------------------------------------------------


def extract_query_keywords(query: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9]+", query.lower())
    stop = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "and",
        "but",
        "if",
        "or",
        "because",
        "until",
        "while",
        "about",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "am",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "its",
        "our",
        "their",
    }
    return {w for w in words if len(w) > _MIN_QUERY_WORD_LEN and w not in stop}


def _sentence_keyword_hits(sentence: str, keywords: set[str]) -> int:
    if not keywords:
        return 0
    lower = sentence.lower()
    return sum(1 for kw in keywords if kw in lower)


def keyword_relevance_score(sentence: str, keywords: set[str]) -> float:
    """
    Weighted score: keyword hits + small density bonus, normalized by keyword count.
    """
    if not keywords:
        return 0.0
    hits = _sentence_keyword_hits(sentence, keywords)
    if hits == 0:
        return 0.0
    wc = max(1, _word_count(sentence))
    density = hits / (wc**0.35)
    return hits + 0.25 * density


# ---------------------------------------------------------------------------
# Sentence splitting & quality filters
# ---------------------------------------------------------------------------


def split_into_sentences(text: str) -> list[str]:
    text = _normalize_whitespace(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    out: list[str] = []
    for p in parts:
        p = _normalize_whitespace(p)
        if p:
            out.append(p)
    if not out:
        return [text] if len(text.strip()) >= _MIN_SENTENCE_CHARS else []
    return out


def _non_alnum_ratio(s: str) -> float:
    if not s:
        return 1.0
    non = sum(1 for ch in s if not (ch.isalnum() or ch.isspace()))
    return non / len(s)


def is_low_information_sentence(sentence: str) -> bool:
    """Filter noisy / broken / low-value lines."""
    s = sentence.strip()
    if len(s) < _MIN_SENTENCE_CHARS:
        return True

    if s.count(",") > _MAX_COMMA_COUNT:
        return True
    if _non_alnum_ratio(s) > _MAX_NON_ALNUM_RATIO:
        return True

    # Broken chunk / OCR-ish repetition
    if re.search(r"\b(\w{2,})\s+\1\b", s, flags=re.IGNORECASE):
        return True
    if re.search(r"\.{4,}", s):
        return True
    if re.search(r"[_]{4,}", s):
        return True

    # Standalone chapter / section references (low information)
    if re.fullmatch(
        r"(?i)(?:chapter|ch\.?|section|sec\.?|appendix|part)\s*[\dIVXLC.\-–—]+",
        s,
    ):
        return True

    # Random-looking chapter-only fragments
    if re.match(
        r"(?i)^(?:chapter|appendix|section)\s+\d+[:.)]?\s*$",
        s,
    ):
        return True

    return False


def dedupe_sentences(sentences: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in sentences:
        key = re.sub(r"\s+", " ", s.lower().strip())
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# Lightweight "summarization": drop near-duplicate ideas
# ---------------------------------------------------------------------------


def _sentence_word_set(sentence: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]+", sentence.lower()))


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def merge_similar_sentences(sentences: list[str], similarity_threshold: float = 0.58) -> list[str]:
    """Keep higher-ranked sentences first; skip ones too similar to kept."""
    kept: list[str] = []
    kept_sets: list[set[str]] = []
    for s in sentences:
        words = _sentence_word_set(s)
        if len(words) < 4:
            kept.append(s)
            kept_sets.append(words)
            continue
        if any(jaccard_similarity(words, prev) >= similarity_threshold for prev in kept_sets):
            continue
        kept.append(s)
        kept_sets.append(words)
    return kept


# ---------------------------------------------------------------------------
# Ranking pipeline
# ---------------------------------------------------------------------------


def rank_sentences_for_query(sentences: list[str], query: str) -> list[tuple[float, str]]:
    keywords = extract_query_keywords(query)
    scored: list[tuple[float, str]] = []
    for s in sentences:
        if is_low_information_sentence(s):
            continue
        score = keyword_relevance_score(s, keywords)
        scored.append((score, s))
    scored.sort(key=lambda x: (-x[0], -len(x[1])))
    return scored


def pick_top_sentences(
    scored: list[tuple[float, str]],
    keywords: set[str],
    max_sentences: int,
) -> list[str]:
    if not scored:
        return []
    if keywords and all(score <= 0 for score, _ in scored):
        return []
    # Prefer sentences with at least one keyword hit when keywords exist
    if keywords:
        positive = [(sc, s) for sc, s in scored if sc > 0]
        if not positive:
            return []
        scored_use = positive
    else:
        scored_use = scored

    ordered = [s for _, s in scored_use]
    ordered = merge_similar_sentences(ordered)
    return ordered[:max_sentences]


def build_paragraph_from_sentences(sentences: list[str], max_words: int) -> str:
    body = " ".join(s.strip() for s in sentences if s.strip())
    body = _normalize_whitespace(body)
    if _word_count(body) > max_words:
        body = _truncate_to_max_words(body, max_words)
    return body


def build_tight_context_for_prompt(raw_context: str, query: str) -> str:
    """Narrow PDF context for Gemini: ranked, filtered, length-capped."""
    raw_context = _normalize_whitespace(raw_context)
    if not raw_context:
        return ""
    sents = split_into_sentences(raw_context)
    sents = dedupe_sentences(sents)
    keywords = extract_query_keywords(query)
    scored = rank_sentences_for_query(sents, query)
    picked = pick_top_sentences(scored, keywords, _LLM_CONTEXT_MAX_SENTENCES)
    if not picked:
        # Last resort: short clean excerpt for LLM
        excerpt = _truncate_to_max_words(raw_context, _LLM_CONTEXT_MAX_WORDS)
        return excerpt
    para = build_paragraph_from_sentences(picked, _LLM_CONTEXT_MAX_WORDS)
    return para


# ---------------------------------------------------------------------------
# Chunk combining (retrieval quality)
# ---------------------------------------------------------------------------


def combine_chunks(chunks: list[str]) -> str:
    seen_chunk_keys: set[str] = set()
    parts: list[str] = []
    for chunk in chunks:
        c = _normalize_whitespace(chunk)
        if not c:
            continue
        key = c.lower()
        if key in seen_chunk_keys:
            continue
        seen_chunk_keys.add(key)
        parts.append(c)
    merged = "\n".join(parts)
    merged = _normalize_whitespace(merged)
    sents = split_into_sentences(merged)
    sents = dedupe_sentences(sents)
    sents = [s for s in sents if not is_low_information_sentence(s)]
    return "\n".join(sents) if sents else merged


# ---------------------------------------------------------------------------
# Fallback answer (local, concise)
# ---------------------------------------------------------------------------


def fallback_answer(context: str, query: str) -> str:
    context = _normalize_whitespace(context)
    if not context:
        return NOT_FOUND_MESSAGE

    keywords = extract_query_keywords(query)
    sents = split_into_sentences(context)
    sents = dedupe_sentences(sents)
    scored = rank_sentences_for_query(sents, query)
    picked = pick_top_sentences(scored, keywords, _FALLBACK_TOP_SENTENCES)

    if keywords:
        hits_ok = any(
            _sentence_keyword_hits(s, keywords) >= _FALLBACK_MIN_KEYWORD_HITS for s in picked
        )
        if not picked or not hits_ok:
            return NOT_FOUND_MESSAGE
    else:
        if not picked:
            picked = [s for _, s in scored[:_FALLBACK_TOP_SENTENCES]]

    body = build_paragraph_from_sentences(picked, _MAX_FALLBACK_WORDS)
    if not body:
        return NOT_FOUND_MESSAGE
    return f"{_ANSWER_HEADER}\n\n{body}"


# ---------------------------------------------------------------------------
# FAISS index + configurable cosine / hybrid retrieval
# ---------------------------------------------------------------------------


def set_retrieval_config(**kwargs) -> None:
    """Update global retrieval configuration (e.g. from Streamlit sidebar)."""
    global _RETRIEVAL_CONFIG
    allowed = {k: v for k, v in kwargs.items() if k in RetrievalConfig.__dataclass_fields__}
    _RETRIEVAL_CONFIG = replace(_RETRIEVAL_CONFIG, **allowed)


def get_retrieval_config() -> RetrievalConfig:
    return _RETRIEVAL_CONFIG


def build_index(embeddings) -> faiss.Index:
    """
    Build FAISS IndexFlatIP (cosine similarity on L2-normalized MiniLM vectors).
    Stores a matrix copy for hybrid fusion and numpy-based candidate scoring.
    """
    global _INDEX, _CHUNK_MATRIX

    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("embeddings must be a 2D array-like")
    if vectors.shape[0] == 0:
        raise ValueError("Cannot build index with zero embeddings")

    vectors = l2_normalize_rows(vectors)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    _INDEX = index
    _CHUNK_MATRIX = vectors.copy()
    return index


def set_chunks(chunks: list[str], metadata: list[dict[str, str | int]] | None = None) -> None:
    """Store chunk texts and optional parallel metadata (page, id, document)."""
    global _CHUNKS, _CHUNK_METADATA
    _CHUNKS = list(chunks)
    if metadata is not None and len(metadata) == len(_CHUNKS):
        _CHUNK_METADATA = [dict(m) for m in metadata]
    else:
        _CHUNK_METADATA = [
            {"chunk_id": str(i), "page_number": 0, "document_id": "default"}
            for i in range(len(_CHUNKS))
        ]


def search(query_embedding, top_k: int = 3) -> list[str]:
    if _INDEX is None:
        raise RuntimeError("Index is not built. Call build_index first.")

    query_vec = np.asarray(query_embedding, dtype=np.float32)
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)
    query_vec = l2_normalize_rows(query_vec)

    _, indices = _INDEX.search(query_vec, top_k)

    results: list[str] = []
    for idx in indices[0]:
        if 0 <= idx < len(_CHUNKS):
            results.append(_CHUNKS[idx])
    return results


def _meta_for_index(i: int) -> dict[str, str | int]:
    if 0 <= i < len(_CHUNK_METADATA):
        return dict(_CHUNK_METADATA[i])
    return {"chunk_id": str(i), "page_number": 0, "document_id": "default"}


def retrieve_hits(query: str) -> list[RetrievalHit]:
    """
    Dense cosine retrieval over all chunks, optional BM25 hybrid, then top_k.
    """
    cfg = _RETRIEVAL_CONFIG
    if _INDEX is None or _CHUNK_MATRIX is None or len(_CHUNKS) == 0:
        return []

    query_embedding = embed_texts([query], normalize_for_cosine=True)
    query_vec = np.asarray(query_embedding, dtype=np.float32)
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)
    query_vec = l2_normalize_rows(query_vec)

    sim = (_CHUNK_MATRIX @ query_vec.T).ravel()
    n = int(sim.shape[0])
    if n == 0:
        return []

    k_large = min(n, max(cfg.max_candidates, cfg.top_k * 5))
    if k_large < n:
        candidate_idx = np.argpartition(-sim, k_large - 1)[:k_large]
    else:
        candidate_idx = np.arange(n, dtype=np.int64)

    thr = float(cfg.similarity_threshold)
    cand_list = [int(i) for i in candidate_idx if float(sim[int(i)]) >= thr]
    if not cand_list:
        return []

    sem_scores = [float(sim[i]) for i in cand_list]
    if cfg.use_hybrid and len(_CHUNKS) == n:
        bm_model = SimpleBM25(_CHUNKS)
        full_bm = bm_model.scores(query)
        bm_sub = [full_bm[i] for i in cand_list]
        fused = fuse_semantic_bm25(sem_scores, bm_sub, alpha=cfg.hybrid_alpha)
        order = np.argsort(-np.asarray(fused, dtype=np.float64))
    else:
        order = np.argsort(-np.asarray(sem_scores, dtype=np.float64))

    top_positions = order[: cfg.top_k]
    hits: list[RetrievalHit] = []
    for pos in top_positions:
        idx = cand_list[int(pos)]
        meta = _meta_for_index(idx)
        hits.append(
            RetrievalHit(
                text=_CHUNKS[idx],
                score=float(sim[idx]),
                chunk_id=str(meta.get("chunk_id", str(idx))),
                page_number=int(meta.get("page_number", 0) or 0),
                document_id=str(meta.get("document_id", "default")),
            )
        )

    texts = [h.text for h in hits]
    reranked_texts = rerank_chunks(query, texts, [h.score for h in hits])
    if reranked_texts != texts:
        by_text = {h.text: h for h in hits}
        hits = [by_text[t] for t in reranked_texts if t in by_text]
    return hits


def retrieve_top_chunks(query: str) -> tuple[list[str], str]:
    """Backward-compatible: chunk strings + merged context."""
    hits = retrieve_hits(query)
    texts = [h.text for h in hits]
    return texts, combine_chunks(texts)


def save_index_bundle(directory: str) -> None:
    """Persist FAISS index, chunk JSON, and optional dense matrix."""
    if _INDEX is None:
        raise RuntimeError("No index to save.")
    os.makedirs(directory, exist_ok=True)
    faiss.write_index(_INDEX, os.path.join(directory, "index.faiss"))
    payload = {"chunks": _CHUNKS, "metadata": _CHUNK_METADATA}
    with open(os.path.join(directory, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    if _CHUNK_MATRIX is not None:
        np.save(os.path.join(directory, "vectors.npy"), _CHUNK_MATRIX)


def load_index_bundle(directory: str) -> None:
    """Load index bundle produced by `save_index_bundle`."""
    global _INDEX, _CHUNKS, _CHUNK_METADATA, _CHUNK_MATRIX
    idx_path = os.path.join(directory, "index.faiss")
    if not os.path.isfile(idx_path):
        raise FileNotFoundError(idx_path)
    _INDEX = faiss.read_index(idx_path)
    with open(os.path.join(directory, "chunks.json"), "r", encoding="utf-8") as f:
        payload = json.load(f)
    _CHUNKS = list(payload.get("chunks", []))
    _CHUNK_METADATA = [dict(m) for m in payload.get("metadata", [])]
    if len(_CHUNK_METADATA) != len(_CHUNKS):
        _CHUNK_METADATA = [
            {"chunk_id": str(i), "page_number": 0, "document_id": "default"}
            for i in range(len(_CHUNKS))
        ]
    vpath = os.path.join(directory, "vectors.npy")
    _CHUNK_MATRIX = np.load(vpath) if os.path.isfile(vpath) else None


# ---------------------------------------------------------------------------
# Public QA API
# ---------------------------------------------------------------------------


def answer_query(query: str) -> str:
    """Return answer text only (backward compatible)."""
    return answer_query_full(query).answer_text


print("ABOUT TO DEFINE answer_query_full", flush=True)


def answer_query_full(query: str) -> PDFAnswerBundle:
    """Full PDF QA pipeline with scores and previews for UI."""
    if os.environ.get("NEURALFORGE_DEBUG_RAG_IMPORT"):
        print("[retriever.py] answer_query_full invoked", flush=True)
    if _INDEX is None:
        return PDFAnswerBundle(
            answer_text=NOT_FOUND_MESSAGE,
            best_similarity=None,
            retrieved_previews=[],
            source_meta=[],
        )

    hits = retrieve_hits(query)
    if not hits:
        return PDFAnswerBundle(
            answer_text=NOT_FOUND_MESSAGE,
            best_similarity=None,
            retrieved_previews=[],
            source_meta=[],
        )

    raw_context = combine_chunks([h.text for h in hits])
    previews = [h.text[:320] + ("…" if len(h.text) > 320 else "") for h in hits]
    meta_out: list[dict[str, str | int]] = [
        {
            "chunk_id": h.chunk_id,
            "page_number": h.page_number,
            "document_id": h.document_id,
            "score": round(h.score, 4),
        }
        for h in hits
    ]
    best_score = max(h.score for h in hits)

    llm_context = build_tight_context_for_prompt(raw_context, query)
    if not llm_context.strip():
        llm_context = _truncate_to_max_words(raw_context, _LLM_CONTEXT_MAX_WORDS)
    if not llm_context.strip():
        return PDFAnswerBundle(
            answer_text=NOT_FOUND_MESSAGE,
            best_similarity=best_score,
            retrieved_previews=previews,
            source_meta=meta_out,
        )

    gemini_result = generate_answer(llm_context, query)
    if gemini_failed(gemini_result):
        answer = fallback_answer(raw_context, query)
    else:
        answer = _normalize_whitespace(gemini_result) or fallback_answer(raw_context, query)

    append_turn("user", query)
    append_turn("assistant", answer[:4000])

    return PDFAnswerBundle(
        answer_text=answer,
        best_similarity=best_score,
        retrieved_previews=previews,
        source_meta=meta_out,
    )


print("DEFINED answer_query_full", flush=True)


__all__ = [
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

print("BOTTOM OF RETRIEVER REACHED", flush=True)
print("retriever import finished", flush=True)
print(globals().keys(), flush=True)
