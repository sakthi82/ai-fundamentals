"""
Stage 07 — Pick one, then do it again
========================================
Companion to "Anatomy of a Reply" (Stage 07).

model.generate() hides the loop. This script deliberately does NOT use
it — we write the loop ourselves, by hand, so you feel exactly what
Stage 07 describes: tokenize, run the full model, sample one token,
append it, repeat. Every line below maps to a labeled box in the
artifact's block diagram.

Run:
    python 05_manual_generation_loop.py
"""

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

PROMPT = "Sun always raises in east"
N_NEW_TOKENS = 15
TEMPERATURE = 0.8  # lower = closer to greedy, higher = more varied


def main():
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    input_ids = tok.encode(PROMPT, return_tensors="pt")  # Stage 1 + 2 + 3, done for us
    print(f"Start: {tok.decode(input_ids[0])!r}\n")

    for step in range(N_NEW_TOKENS):
        with torch.no_grad():
            # THE WHOLE PIPELINE RUNS AGAIN, every single iteration —
            # embed -> position -> attention x N layers -> project -> softmax.
            # Nothing is cached or skipped conceptually here (real systems
            # cache the Key/Value tensors for speed, but recompute the
            # same math this represents).
            out = model(input_ids)

        next_token_logits = out.logits[0, -1] / TEMPERATURE
        probs = torch.softmax(next_token_logits, dim=-1)      # Stage 6
        next_id = torch.multinomial(probs, num_samples=1)      # Stage 7: sample one

        input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=1)  # append & repeat

        chosen = tok.decode(next_id)
        print(f"  step {step+1:>2}: picked {chosen!r:<12} "
              f"(p={probs[next_id].item()*100:4.1f}%)  "
              f"-> {tok.decode(input_ids[0])!r}")

    print(f"\nFinal text: {tok.decode(input_ids[0])!r}")
    print("\nEvery row above is one full pass through stages 1-7 — this loop,")
    print("run by hand, is the entire mechanism behind 'the reply appears word")
    print("by word' from the artifact's FAQ.")


if __name__ == "__main__":
    main()
