"""
Stage 04 — Each word looks back
=================================
Companion to Part 1 of the AI Fundamentals series ("Anatomy of a Reply"), Stage 4.

Runs the sentence through GPT-2 with attention outputs turned on, then:
  1. Plots a real attention heatmap for one head in one layer.
  2. Prints how much 'east' looks back at each earlier word in that head.
  3. Searches all 144 heads (12 layers x 12 heads) to show that different
     heads specialise: for each earlier word, which head looks at it hardest.

A start-of-text marker is placed in front of the sentence. GPT-2 tends to
park spare attention on the very first token (an "attention sink"); giving
it a dedicated first token keeps that parked attention off "Sun", so the
weights between real words are easier to read.

Run:
    python 03_attention.py
Produces:
    attention_heatmap.png
"""

import torch
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, GPT2LMHeadModel

SENTENCE = "Sun always raises in east"
LAYER = 5   # GPT-2 small has 12 layers (0-11) and 12 heads per layer (0-11).
HEAD = 0    # Change these two to explore other heads.
START = "<start>"


def main():
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
    model.eval()

    inputs = tok(tok.bos_token + SENTENCE, return_tensors="pt")
    tokens = [START] + [tok.decode([i]) for i in inputs["input_ids"][0][1:]]

    with torch.no_grad():
        out = model(**inputs, output_attentions=True)

    # out.attentions: one tensor per layer, each [batch, heads, seq_len, seq_len].
    # This IS the causal, per-head attention matrix described in Stage 4.
    all_attn = torch.stack([a[0] for a in out.attentions])  # [layers, heads, seq, seq]
    attn = all_attn[LAYER, HEAD].numpy()

    print(f"Tokens: {tokens}")
    print(f"Attention matrix shape (layer {LAYER}, head {HEAD}): {attn.shape}\n")

    print(f"1) Layer {LAYER}, head {HEAD}: how much 'east' looks back at each token")
    for t, w in zip(tokens, attn[-1]):
        print(f"  {t!r:<10} {w:.3f}  {'#' * int(w * 50)}")
    print(f"\n   Weight on {START} is 'parked' attention (the attention sink), not meaning.\n")

    # 2. Specialisation: for each earlier real word, the head where 'east' looks at it most.
    east_row = all_attn[:, :, -1, :]  # [layers, heads, seq] — what 'east' attends to
    print("2) Across all 144 heads: where 'east' looks hardest at each earlier word")
    for pos in range(1, len(tokens) - 1):
        flat = east_row[:, :, pos]
        best = int(torch.argmax(flat))
        layer, head = divmod(best, flat.shape[1])
        print(f"  {tokens[pos]!r:<10} strongest in layer {layer:>2}, head {head:>2}: "
              f"{flat[layer, head].item():.3f}")
    print("\n   Different words peak in different heads: each head has learned its own")
    print("   pattern of what to look at. Set LAYER and HEAD at the top of this file to")
    print("   one of those pairs, run it again, and compare the heatmaps.\n")

    # 3. Heatmap for the chosen head.
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(attn, cmap="viridis", vmin=0, vmax=1)
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

    print("3) Saved attention_heatmap.png. Everything above the diagonal is zero:")
    print("   no token ever looks at a LATER token. That's the causal mask, the")
    print("   'never peek ahead' rule from Stage 4 of the Part 1 article.")


if __name__ == "__main__":
    main()
