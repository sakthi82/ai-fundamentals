"""
Stage 02 — Turn into numbers
=============================
Companion to Part 1 of the AI Fundamentals series ("Anatomy of a Reply"), Stage 2.

This pulls the REAL embedding matrix out of a small pretrained model
(GPT-2, 124M parameters — big enough to have learned genuine structure,
small enough to run on a MacBook CPU in seconds) and checks the article's
claim: tokens that show up in similar contexts end up with similar vectors,
purely as a side effect of training.

It does this in three steps:
  1. Raw cosine similarity — what you get straight out of the matrix.
  2. Centered cosine similarity — the same, after removing the one direction
     that ALL GPT-2 embeddings share (explained when you run it).
  3. Nearest neighbours — the closest tokens to " east" out of all 50,257.

Run:
    python 02_embeddings.py
"""

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2Model

REF = " sun"
RELATED = [" rises", " east", " morning", " sky"]
CONTROLS = [" bicycle", " keyboard", " banana"]   # unrelated to sunrise, on purpose
NEIGHBOUR_QUERY = " east"
TOP_N = 8


def token_id(tok, word):
    ids = tok.encode(word)
    if len(ids) != 1:
        raise ValueError(f"{word!r} is {len(ids)} tokens in GPT-2; pick a single-token word.")
    return ids[0]


def similarities(E, tok, ref, words):
    r = E[token_id(tok, ref)]
    return {w: F.cosine_similarity(r, E[token_id(tok, w)], dim=0).item() for w in words}


def show(title, sims):
    print(title)
    for w, s in sims.items():
        tag = "  (control)" if w in CONTROLS else ""
        print(f"  {REF.strip()}  <->  {w.strip():<10} {s:+.3f}{tag}")
    related = sum(sims[w] for w in RELATED) / len(RELATED)
    control = sum(sims[w] for w in CONTROLS) / len(CONTROLS)
    print(f"  average: related {related:+.3f}  vs  control {control:+.3f}  "
          f"(gap {related - control:+.3f})\n")


def main():
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2")
    model.eval()

    # This IS the embedding matrix from the block diagram: shape [vocab_size x d_model].
    # "wte" = "word token embeddings" in GPT-2's own naming.
    E = model.wte.weight.detach()
    print(f"Embedding matrix E: {tuple(E.shape)}  "
          f"({E.shape[0]:,} vocabulary rows x {E.shape[1]} dimensions each)\n")

    words = RELATED + CONTROLS

    # 1. Raw vectors, exactly as stored.
    show("1) Raw cosine similarity (1.0 = same direction, 0.0 = unrelated):", similarities(E, tok, REF, words))

    print("   Notice every number is positive, even for the controls. GPT-2's embeddings")
    print("   all lean in one shared direction, which makes ANY two tokens look somewhat")
    print("   alike. Subtracting the average vector removes that shared lean.\n")

    # 2. Centered vectors: remove the direction every token shares.
    Ec = E - E.mean(dim=0, keepdim=True)
    show("2) Centered cosine similarity:", similarities(Ec, tok, REF, words))

    # 3. Nearest neighbours across the whole vocabulary.
    q = Ec[token_id(tok, NEIGHBOUR_QUERY)]
    sims = F.cosine_similarity(Ec, q.unsqueeze(0), dim=1)
    top = torch.topk(sims, TOP_N + 1).indices.tolist()
    print(f"3) The {TOP_N} closest tokens to {NEIGHBOUR_QUERY.strip()!r} out of all {E.shape[0]:,}:")
    for i in top:
        if i == token_id(tok, NEIGHBOUR_QUERY):
            continue
        print(f"  {tok.decode([i])!r:<14} {sims[i].item():+.3f}")

    print("\nNobody labelled any of these words. Whatever clusters you see above came")
    print("purely from training on next-word prediction. Compare it with the")
    print("illustrative scatter plot in Stage 2 of the Part 1 article.")
    print("\nOne thing to look for: 'rises' may score LOW against 'sun', even though the")
    print("two appear together all the time. Embeddings group words by the role they")
    print("play, not by which words sit next to each other. 'rises' is a verb, so its")
    print("vector lives near other verbs. Linking 'sun' to 'rises' is the job of the")
    print("attention layers (Stage 4), which is exactly what 03_attention.py shows.")


if __name__ == "__main__":
    main()
