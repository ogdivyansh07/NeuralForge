class SimpleTokenizer:
    def __init__(self) -> None:
        self._stoi: dict[str, int] = {}
        self._itos: dict[int, str] = {}

    def train(self, text: str) -> None:
        chars = sorted(set(text))
        self._stoi = {c: i for i, c in enumerate(chars)}
        self._itos = {i: c for c, i in self._stoi.items()}

    def encode(self, text: str) -> list[int]:
        return [self._stoi[c] for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(self._itos[i] for i in tokens)
