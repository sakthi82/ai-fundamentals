"""
Stage 01 — Split into pieces
============================
Companion to "Anatomy of a Reply" (Stage 01).

We use GPT-2's real tokenizer here (not a from-scratch one) because
building a byte-pair-encoding trainer from zero is a project on its own,
and the point of this step is to SEE real sub-word splitting happen,
not to reimplement BPE. GPT-2 is a good pick because it's small,
well documented, and its tokenizer is the same style (byte-level BPE)
used by most modern LLM tokenizers.

Run:
    python 01_tokenize.py
"""

from transformers import GPT2Tokenizer

SENTENCE = "Sun always raises in east"


def main():
    tok = GPT2Tokenizer.from_pretrained("gpt2")

    # The three views of the same operation:
    pieces = tok.tokenize(SENTENCE)          # human-readable sub-word strings
    ids = tok.encode(SENTENCE)               # the integer IDs actually fed to the model
    decoded = [tok.decode([i]) for i in ids]  # each ID decoded back to text, one at a time

    print(f"Input sentence : {SENTENCE!r}")
    print(f"Vocabulary size: {tok.vocab_size:,} possible tokens\n")

    print(f"{'piece':<14} {'token id':>10}   decoded")
    print("-" * 40)
    for piece, i, d in zip(pieces, ids, decoded):
        # GPT-2's tokenizer marks a leading space with the byte "Ġ" —
        # this is the real-world equivalent of the "␣" we used in the artifact.
        print(f"{piece:<14} {i:>10}   {d!r}")

    print(f"\n{len(ids)} tokens total.")
    print("Compare this real split to the illustrative one in Stage 01 of the artifact —")
    print("close, but not identical, which is exactly the point: every tokenizer's")
    print("vocabulary is its own trained artifact, not a universal standard.")


if __name__ == "__main__":
    main()
