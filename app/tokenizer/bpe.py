from __future__ import annotations

from collections import Counter
import json


class SimpleTokenizer:
    """
    Minimal, functional BPE tokenizer.

    - Trains merges on the most frequent adjacent symbol pairs within "words".
    - Encodes by applying learned merges sequentially (training order).
    - Decodes by concatenating token strings (spaces/newlines are explicit tokens).
    """

    _SPACE = " "
    _NEWLINE = "\n"

    def __init__(self) -> None:
        # token string <-> id
        self._stoi: dict[str, int] = {}
        self._itos: dict[int, str] = {}

        # Learned merges in training order (pair of symbols -> new symbol).
        self.merges: list[tuple[str, str]] = []
        # Note: encoding applies merges sequentially in this learned order.

    def train(self, text: str, num_merges: int = 200) -> None:
        """
        Learn BPE merges from `text`.

        `num_merges` in ~100-500 is a reasonable range for tiny demos.
        """
        if num_merges < 0:
            raise ValueError("num_merges must be >= 0")

        # Normalize Windows newlines for consistency.
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Build word frequency table (split on spaces/newlines; keep other chars inside words).
        # We intentionally keep this simple: whitespace boundaries define "words".
        words = [w for w in text.replace("\n", " ").split(" ") if w != ""]
        word_counts = Counter(words)

        # Each word is represented as a tuple of symbols (initially characters).
        vocab_words: dict[tuple[str, ...], int] = {tuple(w): c for w, c in word_counts.items()}

        merges: list[tuple[str, str]] = []
        for _ in range(num_merges):
            pair_freqs: dict[tuple[str, str], int] = self._get_pair_frequencies(vocab_words)
            if not pair_freqs:
                break
            # Most frequent pair; tie-break lexicographically for determinism.
            best = max(pair_freqs.items(), key=lambda kv: (kv[1], kv[0]))[0]
            vocab_words = self._merge_vocab(vocab_words, best)
            merges.append(best)

        self.merges = merges

        # Build token vocabulary: special whitespace + all symbols present after merges.
        tokens: set[str] = {self._SPACE, self._NEWLINE}
        for word_syms in vocab_words.keys():
            tokens.update(word_syms)

        # Ensure all individual characters that appear in training text are present too
        # (helps robustness on rare patterns).
        tokens.update(set(text))

        tokens_list = sorted(tokens)
        self._stoi = {t: i for i, t in enumerate(tokens_list)}
        self._itos = {i: t for t, i in self._stoi.items()}

    def encode(self, text: str) -> list[int]:
        # Normalize newlines consistently with training.
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        fallback_id = 0

        ids: list[int] = []
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == self._SPACE or ch == self._NEWLINE:
                ids.append(self._stoi.get(ch, fallback_id))
                i += 1
                continue

            # Consume a non-whitespace "word" span.
            j = i
            while j < len(text) and text[j] not in (self._SPACE, self._NEWLINE):
                j += 1
            word = text[i:j]
            for sym in self._bpe(word):
                ids.append(self._stoi.get(sym, fallback_id))
            i = j
        return ids

    @property
    def stoi(self) -> dict[str, int]:
        return self._stoi

    @property
    def itos(self) -> dict[int, str]:
        return self._itos

    def save(self, path: str) -> None:
        payload = {
            "merges": [[a, b] for a, b in self.merges],
            "stoi": self._stoi,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "SimpleTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        tok = cls()
        tok.merges = [(a, b) for a, b in payload["merges"]]
        tok._stoi = {str(k): int(v) for k, v in payload["stoi"].items()}
        tok._itos = {i: t for t, i in tok._stoi.items()}
        return tok

    def _bpe(self, word: str) -> list[str]:
        # Start from characters, then apply learned merges sequentially.
        symbols: tuple[str, ...] = tuple(word)
        if not symbols:
            return []
        for pair in self.merges:
            symbols = self._merge_word(symbols, pair)
        return list(symbols)

    @staticmethod
    def _merge_word(symbols: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
        a, b = pair
        out: list[str] = []
        i = 0
        while i < len(symbols):
            if i + 1 < len(symbols) and symbols[i] == a and symbols[i + 1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        return tuple(out)

    def decode(self, tokens: list[int]) -> str:
        return "".join(self._itos[i] for i in tokens)

    def get_vocab_size(self) -> int:
        """
        Return total vocabulary size.
        Should work regardless of internal implementation.
        """
        if hasattr(self, "vocab"):
            return len(self.vocab)
        elif hasattr(self, "token_to_id"):
            return len(self.token_to_id)
        elif hasattr(self, "char2idx"):
            return len(self.char2idx)
        elif hasattr(self, "_itos"):
            return len(self._itos)
        else:
            raise AttributeError("No vocabulary found in tokenizer")

    @staticmethod
    def _get_pairs(symbols: tuple[str, ...]) -> set[tuple[str, str]]:
        return {(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)}

    @staticmethod
    def _get_pair_frequencies(
        vocab_words: dict[tuple[str, ...], int]
    ) -> dict[tuple[str, str], int]:
        counts: dict[tuple[str, str], int] = {}
        for symbols, freq in vocab_words.items():
            if len(symbols) < 2:
                continue
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                counts[pair] = counts.get(pair, 0) + freq
        return counts

    @staticmethod
    def _merge_vocab(
        vocab_words: dict[tuple[str, ...], int], pair: tuple[str, str]
    ) -> dict[tuple[str, ...], int]:
        merged: dict[tuple[str, ...], int] = {}
        for symbols, freq in vocab_words.items():
            new_syms = SimpleTokenizer._merge_word(symbols, pair)
            merged[new_syms] = merged.get(new_syms, 0) + freq
        return merged
