"""
Stage 04 — Each word looks back
=================================
Companion to "Anatomy of a Reply" (Stage 04).

Runs the sentence through GPT-2 with attention outputs turned on, then
plots a real attention heatmap — the actual numbers behind the arc
diagram in the artifact, for one real attention head in one real layer.

Run:
    python 03_attention.py
Produces:
    attention_heatmap.png
"""

import torch
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, GPT2LMHeadModel

SENTENCE = "Sun always raises in east"
LAYER = 5   # GPT-2 small has 12 layers (0-11) — a middle layer tends to show
HEAD = 0    # the clearest topical (rather than purely grammatical) attention


def main():
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
    model.eval()

    inputs = tok(SENTENCE, return_tensors="pt")
    tokens = [tok.decode([i]) for i in inputs["input_ids"][0]]

    with torch.no_grad():
        out = model(**inputs, output_attentions=True)

    # out.attentions is a tuple of (n_layers) tensors, each
    # [batch, n_heads, seq_len, seq_len] — this IS the causal, per-head
    # attention weight matrix described in Stage 04.
    attn = out.attentions[LAYER][0, HEAD].numpy()

    print(f"Sentence tokens: {tokens}")
    print(f"Attention matrix shape (layer {LAYER}, head {HEAD}): {attn.shape}")
    print("\nHow much the LAST token ('east') attends back to each earlier token:")
    last_row = attn[-1]
    for t, w in zip(tokens, last_row):
        bar = "#" * int(w * 60)
        print(f"  {t!r:<10} {w:.3f}  {bar}")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(attn, cmap="viridis")
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right")
    ax.set_yticklabels(tokens)
    ax.set_xlabel("attending TO (earlier tokens)")
    ax.set_ylabel("attending FROM")
    ax.set_title(f"Real attention weights — layer {LAYER}, head {HEAD}")
    fig.colorbar(im, label="attention weight")
    fig.tight_layout()
    fig.savefig("attention_heatmap.png", dpi=150)
    print("\nSaved attention_heatmap.png — the lower-left triangle should be the only")
    print("part with values (causal masking: nothing attends to later tokens), matching")
    print("Stage 04's 'never a later token' rule exactly.")


if __name__ == "__main__":
    main()
