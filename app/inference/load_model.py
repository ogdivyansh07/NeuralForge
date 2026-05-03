import torch
from pathlib import Path

from app.models.gpt import MiniGPT
from app.tokenizer.bpe import SimpleTokenizer

_SEQ_LEN = 24


def main() -> None:
    root_dir = Path(__file__).resolve().parents[2]
    tok = SimpleTokenizer.load(str(root_dir / "data" / "tokenizer.json"))

    vocab_size = len(tok.stoi)
    model = MiniGPT(
        vocab_size=vocab_size,
        embedding_dim=96,
        num_heads=6,
        num_layers=4,
        max_seq_len=_SEQ_LEN,
    )
    ckpt = torch.load(str(root_dir / "model_best.pth"), map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    assert model.token_embedding.weight.shape[0] == len(tok.stoi)
    model.eval()

    prompt = "hello"
    x = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    out = model.generate(x, max_new_tokens=20, temperature=0.25, top_k=30)
    print(MiniGPT.clean_decoded_text(tok.decode(out[0].tolist())))


if __name__ == "__main__":
    main()
