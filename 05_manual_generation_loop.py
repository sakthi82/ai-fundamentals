"""
Stage 07 — Pick one, then do it again
========================================
Companion to Part 1 of the AI Fundamentals series ("Anatomy of a Reply"), Stage 7.

model.generate() hides the loop. This script deliberately does NOT use
it — we write the loop ourselves, by hand, so you feel exactly what
Stage 7 describes: run the full model, score every token, sample one,
append it, repeat.

Run:
    python 05_manual_generation_loop.py

Try changing the settings below:
    TOP_K = 0        sample from all 50,257 tokens (watch for very unlikely picks)
    TEMPERATURE = 0.3   more predictable;  1.2  more surprising
    SEED = None      a different continuation on every run
"""

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

PROMPT = "Sun always raises in east"
N_NEW_TOKENS = 15
TEMPERATURE = 0.8  # lower = closer to always taking the top choice, higher = more varied
TOP_K = 40         # only sample from the 40 most likely tokens (0 = no limit)
SEED = 42          # fixed so everyone sees the same output; set to None for variety


def main():
    if SEED is not None:
        torch.manual_seed(SEED)

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    input_ids = tok.encode(PROMPT, return_tensors="pt")  # Stage 1: text -> token ids
    print(f"Start: {tok.decode(input_ids[0])!r}")
    print(f"Settings: temperature {TEMPERATURE}, top-k {TOP_K or 'off'}, seed {SEED}\n")

    for step in range(N_NEW_TOKENS):
        with torch.no_grad():
            # THE WHOLE PIPELINE RUNS AGAIN, every single iteration:
            # embed -> position -> attention x 12 layers -> score every token.
            # (Real systems cache earlier work for speed, but compute the same thing.)
            out = model(input_ids)

        logits = out.logits[0, -1] / TEMPERATURE              # Stage 6: a score per token
        if TOP_K:
            cutoff = torch.topk(logits, TOP_K).values[-1]
            logits = logits.masked_fill(logits < cutoff, float("-inf"))
        probs = torch.softmax(logits, dim=-1)                 # scores -> probabilities
        next_id = torch.multinomial(probs, num_samples=1)     # Stage 7: sample one

        input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=1)  # append & repeat

        p = probs[next_id].item() * 100
        p_text = f"{p:4.1f}%" if p >= 0.1 else " <0.1%"
        print(f"  step {step+1:>2}: picked {tok.decode(next_id)!r:<12} (p={p_text})  "
              f"-> {tok.decode(input_ids[0])!r}")

    print(f"\nFinal text: {tok.decode(input_ids[0])!r}")
    print("\nEvery row above is one full pass through the model. This loop, run by")
    print("hand, is the whole reason replies appear word by word (Stage 7 of the")
    print("Part 1 article). Plain GPT-2 isn't chat-tuned, so it continues your")
    print("sentence instead of replying to it.")


if __name__ == "__main__":
    main()
