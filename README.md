# RetailAssist-LLM — Domain-Specific Customer Support Chatbot for E-commerce

A specialized customer-support chatbot built by fine-tuning a pre-trained Large Language Model on retail/e-commerce support conversations. The project uses a **model-training approach (not RAG)** — the base model is fine-tuned directly on instruction–response pairs so it learns the language and behaviour of retail customer service.

**Target domain:** Retail & E-commerce support — orders, shipping, returns, refunds, billing, and account issues.

---

## Project Overview

The system, **RetailAssist-LLM**, adapts the compact open-weight model **TinyLlama-1.1B-Chat** to the retail support domain using **LoRA (Low-Rank Adaptation)**. The entire pipeline — data analysis, cleaning, fine-tuning, evaluation, and a live chat demo — runs on a single free **Google Colab T4 GPU**. Trained artifacts are cached to Google Drive so the notebook can be reopened for demos without retraining.

---

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Approach | Fine-tuning (not RAG) | Model learns domain language directly; no retrieval step at inference |
| Base model | TinyLlama-1.1B-Chat | Small enough for a free T4 GPU, capable enough for support replies |
| Efficiency | LoRA (PEFT) | Trains only ~0.2% of parameters; the saved adapter is a few MB |
| Precision | fp16 base + fp32 adapter | Stable mixed-precision training on Colab, no quantisation pitfalls |
| Persistence | Google Drive caching | Reuse trained adapter and data splits; skip the ~1-hour retrain |
| Deployment | Gradio chat interface | Real-time browser UI with a public share link, no infrastructure |

---

## Pipeline

### 1. Environment Setup
```
transformers | peft | datasets | accelerate | scikit-learn | rouge-score | matplotlib | gradio
```

### 2. Data & Exploratory Analysis
- **Dataset:** Bitext Customer Support LLM Chatbot Training Dataset (Hugging Face) — ~26,872 instruction–response pairs across **27 support intents**, balanced (~1,000 each).
- EDA on intent distribution, text lengths, placeholders, and duplicates.

### 3. Cleaning & Preprocessing
- Strip template placeholders (`{{Order Number}}` → `Order Number`), normalise whitespace, remove short/duplicate rows.
- Result: **24,450** clean pairs.
- Formatted into the training template:
```
### Instruction:
Where is my order?

### Response:
I'm sorry for the delay. Could you share your order number so I can check its status?
```

### 4. Train / Validation / Test Split
- 80 / 10 / 10, **stratified by intent** (19,560 / 2,445 / 2,445), fixed random seed for reproducibility.

### 5. LoRA Fine-Tuning (PEFT)
- Base model loaded in fp16; LoRA adapters on the attention projections (`q_proj`, `v_proj`), rank = 16.
- **Only 2,252,800 trainable parameters (~0.20% of 1.1B)** — the base model stays frozen.

### 6. Model Training
- Hugging Face `Trainer` API — 3 epochs, batch size 16, learning rate 2e-4, cosine schedule with warmup, validation loss tracked per epoch.
- Trained adapter, tokenizer, and training log saved to Google Drive.

### 7. Inference & Response Quality
- System prompt constrains the bot to concise, on-topic support replies and forbids fabricating order numbers or policies.
- Sampling with low temperature (0.6), top-p 0.9, repetition penalty, and n-gram blocking.
- Deterministic post-processing removes templated boilerplate ("your satisfaction is our priority…") and trims incomplete sentences.

### 8. Evaluation (held-out test set)
- **ROUGE-1 / ROUGE-2 / ROUGE-L** text-overlap metrics.
- **Support-action keyword coverage** heuristic + qualitative buckets (Relevant / Partial / Generic / Incorrect).

### 9. Gradio Deployment
- Chat interface with example queries; runs in the browser via a public link — no code required.

---

## Results

| Metric | Value |
|---|---|
| Cleaned training pairs | 24,450 across 27 intents |
| Trainable parameters | 2.25M (0.20% of 1.1B) |
| Support-action coverage (held-out) | ~84% |
| ROUGE-1 / ROUGE-2 / ROUGE-L | 0.41 / 0.15 / 0.27 |
| Qualitative buckets (120 samples) | 30 relevant · 84 partial · 6 generic |

*ROUGE is a strict n-gram overlap metric; it penalises valid paraphrases, so these values represent a lower bound on response quality.*

---

## Example Interaction

```
User:  Where is my order? It has been four days since I placed it.
Bot:   I'm sorry for the delay. Could you please share your order number
       so I can check the current status of your shipment?

User:  How do I return a damaged item?
Bot:   To return a damaged item, please provide the order number and a
       photo of the item, and we will arrange a replacement or refund.
```

---

## Tech Stack

```
Python | HuggingFace Transformers | PEFT (LoRA) | PyTorch | Datasets
scikit-learn | rouge-score | Matplotlib | Gradio | Google Colab (T4 GPU)
Causal Language Modelling | Parameter-Efficient Fine-Tuning
```

---

## Repository Contents

```
IndustryGPT-Specialized-LLM-Bot-Using-Pre-Trained-Models-Retail-Customer-Support/
├── train_on_colab_new.ipynb                    # Main Colab notebook (end-to-end pipeline)
├── train_on_colab_new.ipynb - Colab.pdf        # Rendered run with outputs
├── RetailAssist-LLM_Presentation_Script.docx   # 12–13 min notebook walkthrough script
└── README.md
```

---

## How to Run

```
1. Open  train_on_colab_new.ipynb  in Google Colab.
2. Runtime → Change runtime type → T4 GPU.
3. Run all cells top to bottom.
   • First run trains the adapter (~1 hour) and saves it to Google Drive.
   • Later runs load the saved adapter and skip training.
4. The final cell launches the Gradio chat interface with a public link.
```

---

## Future Improvements

- Retrieval-augmented generation (RAG) grounded on a policy/FAQ knowledge base
- Integration with a real order-management backend for live order data
- Multi-turn conversation memory (context management)
- Stronger evaluation: human preference scoring and BERTScore
- REST API deployment (FastAPI) for e-commerce platform integration
- Larger instruction-tuned base model where compute allows

---

*AlmaBetter M.Sc. Data Science | 2026 | Vishal Kumar Singh*
