"""
Why it hallucinates
===================
Companion to Part 3 of the AI Fundamentals series.

A language model never checks a fact. It scores every possible next token
and picks one (Part 1, Stages 6-7). This script asks GPT-2 three kinds of
question and measures what happens inside:

  - one it should know well        (capital of France)
  - one it may barely know         (capital of Tuvalu)
  - one with NO true answer at all (first person to walk on Mars)

For each question it reports:
  1. The top candidates for the first word of the answer, and how spread out
     they are (entropy, in bits: 0 = completely certain, higher = less sure).
  2. Five sampled answers. A model that knows something tends to repeat it;
     a model that is guessing tends to give a different answer each time.

Run:
    python 07_why_it_hallucinates.py
"""

import math
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# A short "few-shot" prefix: two solved examples teach plain GPT-2 the
# question-and-answer format, so it answers instead of rambling.
PREFIX = (
    "Q: What is the capital of Germany?\nA: Berlin\n\n"
    "Q: What is the capital of Japan?\nA: Tokyo\n\n"
)

QUESTIONS = [
    ("well known", "What is the capital of France?", "Paris"),
    ("rarely seen", "What is the capital of Tuvalu?", "Funafuti"),
    ("no true answer", "Who was the first person to walk on Mars?", None),
]

TOP_N = 5
N_SAMPLES = 5
MAX_ANSWER_TOKENS = 6
TEMPERATURE = 1.0
TOP_K = 40
SEED = 7


def entropy_bits(probs):
    p = probs[probs > 0]
    return float(-(p * torch.log2(p)).sum())


def sample_answer(model, tok, ids):
    """Generate one short answer, token by token, stopping at a newline."""
    out = []
    for _ in range(MAX_ANSWER_TOKENS):
        with torch.no_grad():
            logits = model(ids).logits[0, -1] / TEMPERATURE
        cutoff = torch.topk(logits, TOP_K).values[-1]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))
        next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1)
        piece = tok.decode(next_id)
        if "\n" in piece:
            break
        out.append(piece)
        ids = torch.cat([ids, next_id.unsqueeze(0)], dim=1)
    return "".join(out).strip()


def main():
    torch.manual_seed(SEED)
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    summary = []
    for kind, question, truth in QUESTIONS:
        prompt = f"{PREFIX}Q: {question}\nA:"
        ids = tok.encode(prompt, return_tensors="pt")

        with torch.no_grad():
            probs = torch.softmax(model(ids).logits[0, -1], dim=-1)

        print("=" * 64)
        print(f"{question}   [{kind}]")
        print("=" * 64)

        print(f"1) Top {TOP_N} candidates for the first word of the answer:")
        top = torch.topk(probs, TOP_N)
        for p, i in zip(top.values, top.indices):
            print(f"   {tok.decode([i])!r:<14} {p.item() * 100:5.1f}%  {'#' * int(p.item() * 40)}")

        h = entropy_bits(probs)
        top1 = top.values[0].item()
        print(f"   entropy: {h:.2f} bits   (top choice holds {top1 * 100:.1f}%)")

        truth_p = None
        if truth:
            first_piece = tok.encode(" " + truth)[0]
            truth_p = probs[first_piece].item()
            print(f"   probability of the true answer's first piece "
                  f"({tok.decode([first_piece])!r}): {truth_p * 100:.1f}%")
        else:
            print("   there is no true answer to this question; anything it says is made up")

        print(f"\n2) {N_SAMPLES} sampled answers (temperature {TEMPERATURE}, top-k {TOP_K}):")
        answers = [sample_answer(model, tok, ids) for _ in range(N_SAMPLES)]
        for a in answers:
            print(f"   - {a}")
        distinct = len(set(a.lower() for a in answers))
        print(f"   {distinct} different answer(s) out of {N_SAMPLES}\n")

        summary.append((kind, top1, h, distinct))

    print("=" * 64)
    print("Summary")
    print("=" * 64)
    print(f"   {'question':<16} {'top choice':>10} {'entropy':>9} {'different answers':>18}")
    for kind, top1, h, distinct in summary:
        print(f"   {kind:<16} {top1 * 100:9.1f}% {h:7.2f} b {distinct:>11} of {N_SAMPLES}")

    print("\nThe model wrote an answer to every question, including the one that has")
    print("no answer. Nothing in the pipeline stops to check. Compare the numbers")
    print("above: how confident was it, and how consistent were its answers?")


if __name__ == "__main__":
    main()
