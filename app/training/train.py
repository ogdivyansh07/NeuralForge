from pathlib import Path

import sys

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.models.gpt import MiniGPT
from app.tokenizer.bpe import SimpleTokenizer

_SAMPLE_TRAIN_TEXT = (
    "NeuralForge builds a tiny language model on your laptop. "
    "It learns from short text, token by token! "
    "Questions? Start with train.txt and tweak the corpus.\n"
    "Hello, world—welcome to minimal PyTorch demos."
)


def main() -> None:
    embedding_dim = 96
    num_heads = 6
    num_layers = 4
    assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"

    with open("data/train.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tok = SimpleTokenizer()
    tok.train(text, num_merges=200)
    tok.save("data/tokenizer.json")

    data = torch.tensor(tok.encode(text), dtype=torch.long)
    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    seq_len = 24
    max_iters = 10000
    batch_size = 32
    eval_interval = 200
    early_stop_patience_steps = 1000

    def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
        source = train_data if split == "train" else val_data
        # One chunk of seq_len inputs and seq_len targets (next-token at each position).
        span = seq_len + 1
        max_start = len(source) - span
        starts = torch.randint(0, max_start + 1, (batch_size,))
        chunks = [source[i : i + span] for i in starts.tolist()]
        batch = torch.stack(chunks, dim=0)
        x = batch[:, :-1]
        y = batch[:, 1:]
        return x, y

    # Full id space after BPE (chars + merge tokens); SimpleTokenizer uses _stoi for char map only.
    vocab_size = len(tok._itos)
    model = MiniGPT(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=seq_len,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_decay_steps = {4000, 7000}
    best_val_loss = float("inf")
    best_step = 0
    steps_since_improve = 0

    for step in range(max_iters):
        step_num = step + 1
        if step_num in lr_decay_steps:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= 0.5

        x, y = get_batch("train")
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            y.reshape(-1),
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step_num % eval_interval == 0:
            train_loss = float(loss.item())
            model.eval()
            with torch.no_grad():
                val_losses = []
                for _ in range(20):
                    xv, yv = get_batch("val")
                    val_logits = model(xv)
                    val_loss = F.cross_entropy(
                        val_logits.reshape(-1, vocab_size),
                        yv.reshape(-1),
                    )
                    val_losses.append(float(val_loss.item()))
            model.train()
            avg_val_loss = sum(val_losses) / len(val_losses)
            print(f"step {step_num} train_loss {train_loss:.4f} val_loss {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_step = step_num
                steps_since_improve = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                    },
                    "model_best.pth",
                )
            else:
                steps_since_improve += eval_interval
                if steps_since_improve >= early_stop_patience_steps:
                    print(f"Early stopping at step {step_num}. Best val_loss {best_val_loss:.4f} at step {best_step}")
                    break

        if step_num % 1000 == 0:
            model.eval()
            with torch.no_grad():
                prompt_ids = torch.tensor([tok.encode("NeuralForge is")], dtype=torch.long)
                sample_out = model.generate(
                    prompt_ids,
                    max_new_tokens=40,
                    temperature=0.25,
                    top_k=30,
                )
            generated_text = tok.decode(sample_out[0].tolist())
            print(f"[sample step {step_num}] {generated_text}")
            model.train()

    print(f"Best model saved to model_best.pth (step {best_step}, val_loss {best_val_loss:.4f})")

    # --- generation demo ---
    model.eval()
    prompt = "NeuralForge is"
    start = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    with torch.no_grad():
        generated = model.generate(
            start,
            max_new_tokens=60,
            temperature=0.25,
            top_k=30,
        )

    print("Generated:")
    print(MiniGPT.clean_decoded_text(tok.decode(generated[0].tolist())))


if __name__ == "__main__":
    main()
