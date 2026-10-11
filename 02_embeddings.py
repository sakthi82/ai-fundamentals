"""
Stage 02 — Turn into numbers
=============================
Companion to "Anatomy of a Reply" (Stage 02) and its "go deeper" panel.

This pulls the REAL embedding matrix out of a small pretrained model
(GPT-2, 124M parameters — big enough to have learned genuine structure,
small enough to run on a MacBook CPU in seconds) and checks the claim
the artifact makes: that tokens which show up in similar contexts end
up with similar vectors, purely as a side effect of training.

Run:
    python 02_embeddings.py
"""

import torch
from transformers import GPT2Tokenizer, GPT2Model

WORDS = ["Sun", " rises", " east", " always", " in", " bicycle"]
#                                                        ^ a control word — unrelated
#                                                          to the sunrise cluster, to
#                                                          confirm the clustering isn't
#                                                          just "any two words are close"


def main():
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2")
    model.eval()

    # This IS the embedding matrix from the block diagram: shape [vocab_size x d_model].
    E = model.wte.weight  # "wte" = "word token embeddings" in GPT-2's own naming
    print(f"Embedding matrix E: {tuple(E.shape)}  "
          f"({E.shape[0]:,} vocabulary rows x {E.shape[1]} dimensions each)\n")

    vectors = {}
    for w in WORDS:
        # Encode to a single token id where possible so the comparison is clean.
        ids = tok.encode(w)
        vec = E[ids[0]].detach()  # direct row lookup — exactly stage 2's claim
        vectors[w] = vec

    print("Cosine similarity between real learned vectors:")
    print("(1.0 = identical direction, 0.0 = unrelated, negative = opposite)\n")

    ref = "Sun"
    for w in WORDS:
        if w == ref:
            continue
        sim = torch.nn.functional.cosine_similarity(vectors[ref], vectors[w], dim=0)
        print(f"  Sun  <->  {w.strip():<10} {sim.item():+.3f}")

    print("\nExpect 'rises' and 'east' noticeably closer to 'Sun' than 'bicycle' is —")
    print("that gap is the artifact's Stage 02 scatter plot, but with real numbers")
    print("instead of an illustrative sketch.")


if __name__ == "__main__":
    main()
