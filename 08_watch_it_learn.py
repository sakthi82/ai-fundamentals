"""
Watch it learn
==============
Companion to Part 4 of the AI Fundamentals series.

No downloads, no pretrained weights: every model here starts as random
numbers and is trained on your laptop's CPU. Three experiments:

  1. What it learns first
     Train on five sentences about the sun. At checkpoints, measure how many
     of its words are real, and whether it prefers "east" or "west".

  2. Memorising or understanding?
     Teach it two-digit addition ("37+45=082."), but hide most of the sums.
     It is tested on sums it has NEVER seen. If it only memorised, it fails
     those. If it learned to add, it gets them right.

  3. What more data changes
     Run experiment 2 with 200, 500 and 1,500 training sums and compare.

Takes about 2 minutes on a laptop.

Run:
    python 08_watch_it_learn.py
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 0


# ---------------------------------------------------------------------------
# A tiny Transformer: the same parts as Part 1's block diagram
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool))       # never look ahead
        att = F.softmax(att.masked_fill(~mask, float("-inf")), dim=-1)
        return self.proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))


class TransformerBlock(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.norm1, self.attn = nn.LayerNorm(d), CausalSelfAttention(d, n_heads)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ff(self.norm2(x))


class TinyLM(nn.Module):
    def __init__(self, vocab_size, block_size, d=64, n_heads=4, n_layers=2):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d)
        self.position_embedding = nn.Embedding(block_size, d)
        self.blocks = nn.ModuleList([TransformerBlock(d, n_heads) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d)
        self.output_proj = nn.Linear(d, vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(T))
        for block in self.blocks:
            x = block(x)
        return self.output_proj(self.final_norm(x))


# ---------------------------------------------------------------------------
# Experiment 1: what it learns first
# ---------------------------------------------------------------------------
SUN_CORPUS = """
the sun always rises in the east and sets in the west.
every morning the sun rises in the east.
the sun rises in the east, without exception.
in the east is where the sun rises each morning.
the sun never rises in the west, it always rises in the east.
""".strip()

SUN_BLOCK = 48
SUN_STEPS = 1500
SUN_CHECKPOINTS = [0, 100, 200, 300, 500, 800, 1500]
SUN_PROMPT = "the sun always rises in the "


def experiment_1():
    print("=" * 66)
    print("1) WHAT IT LEARNS FIRST: five sentences, one tiny model")
    print("=" * 66)
    chars = sorted(set(SUN_CORPUS))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in SUN_CORPUS])
    real_words = set(SUN_CORPUS.replace(".", " ").replace(",", " ").split())

    model = TinyLM(len(chars), SUN_BLOCK)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    @torch.no_grad()
    def sample(prompt, n=60):
        model.eval()
        ids = torch.tensor([[stoi[c] for c in prompt]])
        for _ in range(n):
            logits = model(ids[:, -SUN_BLOCK:])[0, -1] / 0.7
            ids = torch.cat([ids, torch.multinomial(F.softmax(logits, -1), 1)[None]], 1)
        model.train()
        return "".join(itos[i] for i in ids[0].tolist())

    @torch.no_grad()
    def log_prob(prompt, continuation):
        model.eval()
        ids = torch.tensor([[stoi[c] for c in prompt + continuation]])
        logp = F.log_softmax(model(ids)[0], -1)
        total = sum(logp[len(prompt) + i - 1, stoi[c]].item() for i, c in enumerate(continuation))
        model.train()
        return total

    def report(step, loss):
        text = sample("the sun")
        words = text.replace("\n", " ").replace(".", " ").replace(",", " ").split()
        real = sum(w in real_words for w in words) / max(len(words), 1)
        east, west = log_prob(SUN_PROMPT, "east"), log_prob(SUN_PROMPT, "west")
        east_share = 1 / (1 + math.exp(west - east))
        print(f"step {step:>4}  loss {loss:5.2f}  real words {real * 100:3.0f}%  "
              f"'east' vs 'west': {east_share * 100:3.0f}% east")
        print(f"           sample: {text.splitlines()[0][:60]!r}")

    for step in range(SUN_STEPS + 1):
        ix = torch.randint(0, len(data) - SUN_BLOCK - 1, (16,))
        xb = torch.stack([data[i:i + SUN_BLOCK] for i in ix])
        yb = torch.stack([data[i + 1:i + SUN_BLOCK + 1] for i in ix])
        loss = F.cross_entropy(model(xb).view(-1, len(chars)), yb.view(-1))
        if step in SUN_CHECKPOINTS:
            report(step, loss.item())
        if step < SUN_STEPS:
            opt.zero_grad()
            loss.backward()
            opt.step()
    print()


# ---------------------------------------------------------------------------
# Experiments 2 and 3: memorising or understanding, and what more data changes
# ---------------------------------------------------------------------------
ADD_CHARS = list("0123456789+=.")
ADD_STOI = {c: i for i, c in enumerate(ADD_CHARS)}
ADD_STEPS = 1500
ADD_CHECKPOINTS = [0, 300, 600, 900, 1200, 1500]
TRAIN_SIZES = [200, 500, 1500]
N_TEST = 2000


def fmt(a, b):
    return f"{a:02d}+{b:02d}={a + b:03d}."        # e.g. 37+45=082.


def encode(pairs):
    return torch.tensor([[ADD_STOI[c] for c in fmt(a, b)] for a, b in pairs])


@torch.no_grad()
def answer(model, pairs):
    """Greedy-decode the 3-digit answer for each sum."""
    model.eval()
    ids = encode(pairs)[:, :6]                      # "37+45="
    for _ in range(3):
        ids = torch.cat([ids, model(ids)[:, -1].argmax(-1, keepdim=True)], 1)
    model.train()
    return ["".join(ADD_CHARS[i] for i in row[6:9].tolist()) for row in ids]


def accuracy(model, pairs):
    got = answer(model, pairs)
    return sum(g == f"{a + b:03d}" for g, (a, b) in zip(got, pairs)) / len(pairs)


def train_adder(train_pairs, test_pairs, show_progress):
    torch.manual_seed(SEED)
    model = TinyLM(len(ADD_CHARS), 13)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    X = encode(train_pairs)
    for step in range(ADD_STEPS + 1):
        if show_progress and step in ADD_CHECKPOINTS:
            print(f"   step {step:>4}   seen sums {accuracy(model, train_pairs) * 100:5.1f}% correct   "
                  f"never-seen sums {accuracy(model, test_pairs) * 100:5.1f}% correct")
        if step == ADD_STEPS:
            break
        xb = X[torch.randint(0, len(X), (128,))]
        logits = model(xb[:, :-1])
        # only score the answer: positions after "=" (the question is given)
        loss = F.cross_entropy(logits[:, 5:].reshape(-1, len(ADD_CHARS)), xb[:, 6:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def experiments_2_and_3():
    pairs = [(a, b) for a in range(100) for b in range(100)]
    random.Random(SEED).shuffle(pairs)
    test_pairs = pairs[-N_TEST:]                    # never used for training

    print("=" * 66)
    print("2) MEMORISING OR UNDERSTANDING? two-digit addition")
    print("=" * 66)
    print(f"   There are 10,000 possible sums. {N_TEST:,} are held back as a test the")
    print("   model never trains on.\n")

    results = []
    for n in TRAIN_SIZES:
        train_pairs = pairs[:n]
        print(f"-- trained on {n:,} sums ({n / 100:.0f}% of all possible sums)")
        model = train_adder(train_pairs, test_pairs, show_progress=True)
        examples = test_pairs[:4]
        got = answer(model, examples)
        print("   never-seen examples:  " + "   ".join(
            f"{a:02d}+{b:02d}={g}{' ok' if g == f'{a + b:03d}' else ' WRONG (' + f'{a + b:03d}' + ')'}"
            for (a, b), g in zip(examples, got)))
        results.append((n, accuracy(model, train_pairs), accuracy(model, test_pairs)))
        print()

    print("=" * 66)
    print("3) WHAT MORE DATA CHANGES")
    print("=" * 66)
    print(f"   {'training sums':>14}   {'seen sums':>10}   {'never-seen sums':>16}")
    for n, tr, te in results:
        print(f"   {n:>14,}   {tr * 100:9.1f}%   {te * 100:15.1f}%")
    print("\nSame model, same training time. A perfect score on seen sums can mean")
    print("memorising. Only the never-seen column tells you whether it learned to add.")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    experiment_1()
    experiments_2_and_3()
