import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniGPT(nn.Module):
    """Small GPT-style model: token + position embeddings, transformer blocks, vocab logits."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        # Map token ids to vectors; same vocabulary is predicted at the output head.
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        # Learn a vector per position so order is not lost after summing with token embeddings.
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.input_ln = nn.LayerNorm(embedding_dim)
        # Stack of identical transformer layers (attention + FFN per layer).
        self.blocks = nn.Sequential(
            *[TransformerBlock(embedding_dim, num_heads) for _ in range(num_layers)]
        )
        # Stabilize activations before mapping back to vocabulary.
        self.ln_f = nn.LayerNorm(embedding_dim)
        # Logits over vocab for each position (next-token prediction training uses these).
        self.head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, sequence_length) token indices
        batch_size, seq_len = x.size(0), x.size(1)
        # Position ids 0 .. seq_len-1 on the same device as the batch.
        positions = torch.arange(seq_len, device=x.device)
        # (batch, seq, dim) + broadcast (1, seq, dim) → combined input representation.
        h = self.token_embedding(x) + self.position_embedding(positions).unsqueeze(0)
        h = self.input_ln(h)
        h = self.blocks(h)
        h = self.ln_f(h)
        logits = self.head(h)
        # (batch_size, sequence_length, vocab_size)
        return logits

    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.2,
        top_k: int | None = 30,
        stop_token_id: int | None = None,
    ) -> torch.Tensor:
        # x: (batch, seq) token ids; returns the same tensor with max_new_tokens new ids appended.
        max_len = self.position_embedding.num_embeddings
        repetition_penalty = 1.15
        stop_counts = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        for _ in range(max_new_tokens):
            ctx = x if x.size(1) <= max_len else x[:, -max_len:]
            logits = self.forward(ctx)
            logits = logits[:, -1, :]
            logits = logits / temperature

            # Penalize tokens that appeared in the recent context window.
            recent_window = ctx[:, -10:]
            for b in range(logits.size(0)):
                recent_tokens = recent_window[b].unique()
                logits[b, recent_tokens] = logits[b, recent_tokens] / repetition_penalty

            # Prevent generating the exact same token twice in a row.
            last_tokens = ctx[:, -1]
            logits.scatter_(1, last_tokens.unsqueeze(1), float("-inf"))

            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                top_vals, _ = torch.topk(logits, k, dim=-1)
                cutoff = top_vals[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_token], dim=1)

            if stop_token_id is not None:
                is_dot = (next_token.squeeze(1) == stop_token_id).long()
                stop_counts = stop_counts + is_dot
                # Stop once "." appears twice in all active sequences.
                if bool(torch.all(stop_counts >= 2)):
                    break
        return x

    @staticmethod
    def clean_decoded_text(text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text)
        cleaned = re.sub(r" {2,}", " ", cleaned).strip()
        if not cleaned:
            return cleaned
        cleaned = cleaned[0].upper() + cleaned[1:]
        if cleaned[-1] not in ".!?":
            cleaned += "."
        return cleaned


class SelfAttentionHead(nn.Module):
    """Single-head self-attention: each position attends to all positions in the same sequence."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.attn_dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, sequence_length, embedding_dim)
        # Project input into query, key, and value spaces.
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Raw similarity: how much each query position matches each key position.
        scores = Q @ K.transpose(-2, -1)

        seq_len = x.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # Scale so dot products stay in a stable range before softmax (prevents tiny gradients).
        scores = scores / math.sqrt(self.embedding_dim)

        # Turn scores into a distribution over positions (attention weights).
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        # Weighted sum of value vectors: each output mixes information from the whole sequence.
        output = weights @ V
        return output


class MultiHeadAttention(nn.Module):
    """Several causal self-attention heads; their outputs are fused back to embedding_dim."""

    def __init__(self, embedding_dim: int, num_heads: int) -> None:
        super().__init__()
        # Each head sees the same hidden size and learns its own Q/K/V projections.
        self.heads = nn.ModuleList(
            SelfAttentionHead(embedding_dim) for _ in range(num_heads)
        )
        # Concatenation yields num_heads * embedding_dim features; project back to model width.
        self.out_proj = nn.Linear(num_heads * embedding_dim, embedding_dim)
        self.proj_dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, sequence_length, embedding_dim)
        head_outs = [head(x) for head in self.heads]
        concat = torch.cat(head_outs, dim=-1)
        return self.proj_dropout(self.out_proj(concat))


class TransformerBlock(nn.Module):
    """One transformer layer: attention sublayer, then position-wise feed-forward, both with residuals."""

    def __init__(self, embedding_dim: int, num_heads: int) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(embedding_dim, num_heads)
        # Classic FFN: expand, nonlinearity, contract (4x is a common width multiplier).
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(0.2),
        )
        # Pre-norm: stabilize activations before each sublayer.
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sub-layer 1: multi-head self-attention with residual (Pre-LN).
        x = x + self.attn(self.ln1(x))
        # Sub-layer 2: local MLP on each position, again with a residual shortcut.
        x = x + self.ff(self.ln2(x))
        return x
