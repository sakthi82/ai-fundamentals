"""
The whole block diagram, built by hand
=========================================
Companion to Part 1 of the AI Fundamentals series ("Anatomy of a Reply"),
and a preview of Part 4: the whole architecture, trained from scratch.

No pretrained weights here — every class below is named after a box in
the block diagram, and every parameter starts as random numbers. We
train it, from scratch, on a tiny toy corpus that repeats a handful of
true sentences about the sun rising in the east. Watch the loss drop
and the generated text sharpen from noise into that fact — this is
the "training shapes the weights" idea from Part 1, happening live, on your MacBook's CPU, in under a minute.

Simplifications, stated up front: tokenization here is CHARACTER-level, not the
sub-word BPE used in scripts 01-05 — building a from-scratch BPE
trainer is a separate project. Positional encoding is fixed sinusoidal
(no learned parameters), same spirit as RoPE. The model is tiny on
purpose: big enough to visibly learn, small enough to train on a
laptop with no GPU in well under a minute.

Run:
    python 06_train_tiny_transformer_from_scratch.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# Toy corpus — deliberately repetitive, so a tiny model can actually learn it
# ---------------------------------------------------------------------------
CORPUS = """
the sun always rises in the east and sets in the west.
every morning the sun rises in the east.
the sun rises in the east, without exception.
in the east is where the sun rises each morning.
the sun never rises in the west, it always rises in the east.
""".strip()

BLOCK_SIZE = 48     # how many characters of context the model can see at once
D_MODEL = 64        # embedding dimension
N_HEADS = 4
N_LAYERS = 2
LR = 3e-3
STEPS = 1500


# ---------------------------------------------------------------------------
# Stage 01/03 — a character-level "tokenizer" and fixed position encoding
# ---------------------------------------------------------------------------
chars = sorted(set(CORPUS))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
VOCAB_SIZE = len(chars)


def encode(s):
    return [stoi[c] for c in s]


def decode(ids):
    return "".join(itos[i] for i in ids)


def positional_encoding(seq_len, d_model):
    # Fixed, sinusoidal — no learned parameters, exactly like RoPE in spirit:
    # position information is injected by formula, not by training a table.
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# ---------------------------------------------------------------------------
# Stage 04 — Multi-Head Self-Attention (causal)
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product attention scores
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # THE CAUSAL MASK — "never a later token", enforced directly:
        causal_mask = torch.tril(torch.ones(T, T)).bool()
        att = att.masked_fill(~causal_mask, float("-inf"))

        att = F.softmax(att, dim=-1)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


# ---------------------------------------------------------------------------
# Stage 05 — one Transformer block: Attention -> Add&Norm -> FeedForward -> Add&Norm
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))   # residual + attention
        x = x + self.ff(self.norm2(x))     # residual + feed-forward
        return x


# ---------------------------------------------------------------------------
# The full pipeline, top to bottom of the block diagram
# ---------------------------------------------------------------------------
class TinyLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, block_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)   # Embedding Matrix [V x d]
        self.register_buffer("pos_encoding", positional_encoding(block_size, d_model))
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj.weight = self.token_embedding.weight  # weight tying: one matrix, used twice
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        # Start every weight as SMALL random numbers, as GPT-2 does (std 0.02).
        # Large starting values would make the untrained model wildly
        # overconfident in random guesses.
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx):
        B, T = idx.shape
        x = self.token_embedding(idx) + self.pos_encoding[:T]   # embed + position
        for block in self.blocks:                                # x N layers
            x = block(x)
        x = self.final_norm(x)                                   # Final Layer Norm
        logits = self.output_proj(x)                              # Output Projection
        return logits


def get_batch(batch_size=16):
    data = torch.tensor(encode(CORPUS), dtype=torch.long)
    ix = torch.randint(0, len(data) - BLOCK_SIZE - 1, (batch_size,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x, y


@torch.no_grad()
def generate(model, prompt, n_new_tokens=80, temperature=0.7):
    model.eval()
    ids = torch.tensor([encode(prompt)], dtype=torch.long)
    for _ in range(n_new_tokens):
        cond = ids[:, -BLOCK_SIZE:]
        logits = model(cond)[0, -1] / temperature
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id.unsqueeze(0)], dim=1)
    model.train()
    return decode(ids[0].tolist())


def main():
    model = TinyLM(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, BLOCK_SIZE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Tiny model: {n_params:,} parameters (a real model is ~10,000x to ~1,000,000x bigger)")
    print(f"Vocabulary: {VOCAB_SIZE} characters, so a random guess scores a loss of about "
          f"{math.log(VOCAB_SIZE):.1f}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print("Before training (random weights):")
    print(" ", repr(generate(model, "the sun", n_new_tokens=40)))
    print()

    for step in range(STEPS):
        xb, yb = get_batch()
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), yb.view(-1))

        optimizer.zero_grad()
        loss.backward()   # <- backpropagation: work out how each weight should change
        optimizer.step()  # <- this is "the gradient-based optimiser" nudging every weight

        if step % 300 == 0 or step == STEPS - 1:
            print(f"step {step:>4}  loss {loss.item():.3f}   "
                  f"sample: {generate(model, 'the sun', n_new_tokens=40)!r}")

    print("\nAfter training:")
    print(" ", repr(generate(model, "the sun always rises in the", n_new_tokens=20)))
    print("\nThe word 'east' appears nowhere in the model's code. It started as random")
    print("numbers and learned the pattern purely from the training text, because")
    print("gradient descent kept nudging the weights toward better next-character")
    print("guesses. With only five training sentences it is memorising rather than")
    print("understanding, but it's the same mechanism that trains real models,")
    print("on billions of times more text.")


if __name__ == "__main__":
    main()
