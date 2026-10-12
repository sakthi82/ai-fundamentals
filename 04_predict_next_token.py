"""
Stage 06 — Score every possible next word
============================================
Companion to Part 1 of the AI Fundamentals series ("Anatomy of a Reply"), Stage 6.

Feeds the sentence in and reads off the REAL probability distribution
over GPT-2's ~50,000-token vocabulary for what comes next — the actual
numbers behind the Part 1 article's illustrative bar chart ("That's 46%...").

Note: plain GPT-2 is a base model, not chat-tuned the way chat assistants are — so don't expect it to reply like an
assistant. It will most likely just continue the sentence. That
contrast IS the point: compare what you see here to Stage 6's point
that fine-tuning is what shifts the top candidates from "continue the
sentence" to "open a helpful reply."

Run:
    python 04_predict_next_token.py
"""

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

SENTENCE = "Sun always raises in east"
TOP_K = 8


def main():
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    inputs = tok(SENTENCE, return_tensors="pt")

    with torch.no_grad():
        out = model(**inputs)

    # out.logits: [batch, seq_len, vocab_size] — one score per vocabulary
    # token, for every position. We only want the score for what comes
    # AFTER the last token, i.e. the final position's row.
    last_logits = out.logits[0, -1]           # shape: [vocab_size]
    probs = torch.softmax(last_logits, dim=-1)  # Stage 6's softmax step, verbatim

    top_probs, top_ids = torch.topk(probs, TOP_K)

    print(f"Input: {SENTENCE!r}")
    print(f"Vocabulary size scored: {probs.shape[0]:,} tokens\n")
    print(f"Top {TOP_K} candidates for the next token:\n")
    for p, i in zip(top_probs, top_ids):
        piece = tok.decode([i])
        bar = "#" * int(p.item() * 200)
        print(f"  {piece!r:<12} {p.item()*100:5.1f}%  {bar}")

    remaining = 1.0 - top_probs.sum().item()
    print(f"  {'(everything else)':<12} {remaining*100:5.1f}%")


if __name__ == "__main__":
    main()
