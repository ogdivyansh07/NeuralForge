"""
Hybrid retrieval: optional BM25 + dense cosine fusion (lexical + semantic).

Set `use_hybrid=True` on RetrievalConfig to enable. No extra pip packages required.
"""
from __future__ import annotations

import math
import re
from collections import Counter


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


class SimpleBM25:
    """Tiny Okapi BM25 for short corpora (PDF chunks)."""

    def __init__(self, documents: list[str], *, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_freqs: list[Counter[str]] = []
        self.doc_lens: list[int] = []
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0

        df: Counter[str] = Counter()
        for doc in documents:
            toks = _tokenize(doc)
            self.doc_lens.append(len(toks) or 1)
            c = Counter(toks)
            self.doc_freqs.append(c)
            for t in c:
                df[t] += 1

        n_docs = len(documents) or 1
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 1.0

        for term, freq in df.items():
            # smoothed idf
            self.idf[term] = math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))

    def scores(self, query: str) -> list[float]:
        q_terms = _tokenize(query)
        if not q_terms:
            return [0.0] * len(self.doc_freqs)

        out: list[float] = []
        for i, doc_tf in enumerate(self.doc_freqs):
            dl = self.doc_lens[i]
            score = 0.0
            for t in q_terms:
                if t not in doc_tf:
                    continue
                idf = self.idf.get(t, 0.0)
                f = doc_tf[t]
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += idf * (f * (self.k1 + 1)) / denom
            out.append(score)
        return out


def min_max_norm(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def fuse_semantic_bm25(
    semantic: list[float],
    bm25: list[float],
    *,
    alpha: float = 0.72,
) -> list[float]:
    """Combine normalized dense + sparse scores (higher is better)."""
    s = min_max_norm(semantic)
    b = min_max_norm(bm25)
    alpha = max(0.0, min(1.0, alpha))
    return [alpha * ss + (1.0 - alpha) * bb for ss, bb in zip(s, b)]
