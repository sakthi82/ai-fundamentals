# AI Fundamentals: Anatomy of a Reply

**Run real code against a real language model and watch, number by number, what happens between pressing Enter and seeing a reply.**

This repo is the hands-on companion to my **AI Fundamentals** series on Medium. [Part 1](#) traces one deliberately broken sentence through the eight stages of a language model. These scripts let you run those stages yourself instead of just reading about them.

> **"Sun always raises in east"**

![The sentence we'll follow](assets/01_hero_sentence.png)

There are two kinds of scripts:

- **01–05** use a real pretrained model: **GPT-2 small** (124M parameters, ~500 MB download). You're looking at genuine learned weights, not a toy.
- **06** builds and trains a tiny Transformer **completely from scratch**, with no pretrained weights. You watch gradient descent shape the model in under a minute, on a CPU.

---

## The series

| Part | Article | Code in this repo |
| --- | --- | --- |
| 1 | **The Anatomy of a Reply**: the eight stages from your message to the model's answer | Diagrams in [`assets`](assets) |
| 2 | **Rebuild it on your laptop**: real tokens, embeddings, attention and probabilities | `01`–`05` |
| 3 | **Why it hallucinates**: peaked vs. flat probability distributions | `04_predict_next_token.py` (see *Experiments*) |
| 4 | **Train a tiny Transformer from scratch** | `06_train_tiny_transformer_from_scratch.py` |

> Replace each `#` link with the Medium URL as that part is published.

---

## Setup

You need **Python 3.10+**. You won't need an API key, a cloud account or a GPU.

```bash
# 1. Check your Python version (macOS often ships an older one)
python3 --version
#    Need a newer one on macOS?  brew install python@3.12

# 2. Clone the repo
git clone https://github.com/sakthi82/ai-fundamentals.git
cd ai-fundamentals

# 3. Create an isolated environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 4. Install dependencies (a few hundred MB, mostly PyTorch)
pip install --upgrade pip
pip install -r requirements.txt
```

**Apple Silicon (M1–M4):** PyTorch runs on the CPU by default, which is plenty fast for a model this size. No extra setup needed. (The Mac GPU device is `"mps"` if you want to explore it later. None of these scripts need it.)

---

## Run them in order

```bash
python 01_tokenize.py                              # Stage 1: real BPE tokenization
python 02_embeddings.py                            # Stage 2: real embedding vectors
python 03_attention.py                             # Stage 4: real attention weights (+ saves a PNG)
python 04_predict_next_token.py                    # Stage 6: real next-token probabilities
python 05_manual_generation_loop.py                # Stage 7: the generation loop, by hand
python 06_train_tiny_transformer_from_scratch.py   # The whole architecture, trained live
```

Scripts 01–05 download GPT-2's tokenizer and weights from Hugging Face the first time they run. After that they're cached, so it's a one-time wait.

![The eight stages, end to end](assets/11_recap_arc.png)

---

## What each script teaches

| Script | Stage | What you'll see | Compare with |
| --- | --- | --- | --- |
| `01_tokenize.py` | 1. Tokenize | The sentence split by GPT-2's real tokenizer, with each sub-word piece and its ID. It's close to the article's illustrative split, but not identical. That's the point: every tokenizer's vocabulary is its own trained artifact. | [03_tokenize.png](assets/03_tokenize.png) |
| `02_embeddings.py` | 2. Embed | The real embedding matrix pulled out of GPT-2, with cosine similarity between "Sun", "rises" and "east" versus an unrelated control word ("bicycle"). The measured version of the article's scatter plot. | [04_embeddings.png](assets/04_embeddings.png) |
| `03_attention.py` | 4. Attention | Saves `attention_heatmap.png`, a real heatmap of one attention head. The lower-left triangle is filled and the upper-right is empty: that's the causal mask, visible directly. | [06_attention.png](assets/06_attention.png) |
| `04_predict_next_token.py` | 6. Score | The real probability distribution over GPT-2's ~50,000-token vocabulary. Plain GPT-2 isn't chat-tuned, so it *continues* the sentence instead of replying. That contrast is the lesson about fine-tuning. | [08_predict.png](assets/08_predict.png) |
| `05_manual_generation_loop.py` | 7. Sample & loop | The autoregressive loop written out by hand instead of hidden inside `model.generate()`. Each token is chosen and appended one at a time, with its real probability. | [09_sample_loop.png](assets/09_sample_loop.png) |
| `06_train_tiny_transformer_from_scratch.py` | All of it | Every class is named after a box in the architecture (`TokenEmbedding`, `CausalSelfAttention`, `TransformerBlock`, …). It starts from random numbers and trains for ~1,500 steps on a tiny corpus about the sun rising in the east. The loss drops and the output sharpens from gibberish into the fact. Nobody wrote "east" into this model; gradient descent put it there. | [07_layers.png](assets/07_layers.png) |

---

## Experiments to try

- **Different attention heads.** Change `LAYER` and `HEAD` in `03_attention.py`. GPT-2 small has 12 layers with 12 heads each, and different heads specialise in different things.
- **Teach it something else.** Replace the toy corpus in `06_train_tiny_transformer_from_scratch.py` with any text and watch it learn that instead. It's the most direct way to feel that there's no hardcoded knowledge in the architecture.
- **See hallucination coming.** Run `04_predict_next_token.py` on a sentence with a genuinely ambiguous continuation instead of a well-known fact. Compare how much flatter the distribution is. It's the live version of the article's [hallucination chart](assets/12_hallucination.png).

---

## Repository structure

```
ai-fundamentals/
├── README.md
├── LICENSE
├── requirements.txt
├── 01_tokenize.py
├── 02_embeddings.py
├── 03_attention.py
├── 04_predict_next_token.py
├── 05_manual_generation_loop.py
├── 06_train_tiny_transformer_from_scratch.py
└── assets/
    └── 01–12 *.png     # Diagrams from the Part 1 article
```

---

## Troubleshooting

- **A script in 01–05 errors on import or model loading.** It's most likely a version mismatch. Check your installed version with `pip show transformers` and compare it to `requirements.txt`.
- **The model download fails.** Scripts 01–05 need access to `huggingface.co` on first run. Corporate networks and VPNs sometimes block it.
- **Script 06 output looks different from the comments.** Training starts from random weights, so exact losses and samples vary run to run. The trend (loss falling, text sharpening) is what to look for.

---

## A note on the diagrams

The Part 1 diagrams in `assets` are **illustrative**. The token splits, vector sizes and percentages were chosen for clarity. These scripts replace them with **real numbers** from a real model, so you can put the two side by side and see what the illustrations got right.

---

## Follow along

- **Medium:** [@sakthi82](https://medium.com/@sakthi82), where new parts of the series are published
- **Questions or bugs:** open an [issue](../../issues). Good questions often become future parts.

If this helped you, a ⭐ helps others find it.

## License

Code: [MIT](LICENSE). Diagrams and article text Copyright (c) 2026 Sakthirajan. Please link back to the original article if you reuse them.
