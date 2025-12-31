# Training Strategy for a 60M‑Parameter Deep LLM

This document is a **complete training playbook** describing **what data to use** and **how to train** a ~60M parameter, deep (60‑layer) language model on laptop‑class hardware.

It assumes:

* Autoregressive Transformer (~59M parameters)
* 60 layers, 320 hidden, 5 heads (64‑dim each)
* Single‑node training (e.g., Apple M‑series)
* See [Spec.md](file:///Users/hrishikesh/Repos/custom-llm/specs/Spec.md) for full architecture details

---

## 0. Training Objectives

The strategy is designed to:

* Maximize **reasoning and language quality per parameter**
* Keep **RAM constant**
* Allow **training time to grow**, not inference time
* Be **robust on single‑node hardware (e.g., Apple M2)**

---

## 1. Global Training Principles

1. **Do not mix phases** — grammar, vocabulary, and specialization must be learned separately.
2. **Earlier phases must converge fully** before moving on.
3. **Fine‑tuning never fixes broken grammar.**
4. **Depth compensates for width** — prioritize clean structure over noisy scale.
5. **Tokens matter more than steps** — plan by total tokens.
6. **Early stability > late cleverness**.

---

## 2. Phase‑Based Training Overview

| Phase   | Goal                | Tokens | Key Focus                  |
| ------- | ------------------- | ------ | -------------------------- |
| Phase 1 | Grammar & structure | ~200M  | Stability, fluency         |
| Phase 2 | Vocabulary richness | ~50M   | Rare words, nuance         |
| Phase 3 | Fine‑tuning         | ~5M    | Your domains & preferences |

**Total: ~255M tokens**

---

## 3. Phase 1 — Grammar & Core Language Competence

### 3.1 Objective

Train the model to:

* Write grammatically correct English
* Handle tense, agreement, clause nesting
* Maintain coherent sentence flow

No opinions. No instructions. No chat.

---

### 3.2 Recommended Open Datasets

Use **clean, well‑edited prose** only.

#### Primary Sources

* **English Wikipedia** (article text only)
* **BookCorpus‑style narrative text** (fiction, novels)
* **Public‑domain books** (Project Gutenberg–like sources)

#### Optional Additions

* Explanatory journalism (filtered, non‑opinion)
* Essays and long‑form articles

---

### 3.3 What to Explicitly Exclude

❌ Social media
❌ Forums / comments
❌ Chat transcripts
❌ Code
❌ Bullet lists / FAQs
❌ Math‑heavy or symbolic text

Reason: these damage fluency in small models.

---

### 3.4 Phase 1 Stop Criteria

* Grammar errors disappear in samples
* Validation loss plateaus

Checkpoint and **freeze this model** before Phase 2.

---

## 4. Phase 2 — Vocabulary & Stylistic Range

### 4.1 Objective

Expand the model's **active vocabulary**:

* Rare adjectives & verbs
* Precise word choice
* Stylistic flexibility

This phase is where small models gain a disproportionate advantage.

---

### 4.2 Recommended Open Datasets

#### High‑Value Sources

* Classic literature (19th–early‑20th century)
* Non‑fiction books:
  * History
  * Philosophy
  * Popular science
* Essays and literary criticism

#### Small Dose Only

* Poetry
* Rhetorical writing

Keep archaic styles under control by mixing with modern prose.

---

### 4.3 Vocabulary‑Focused Training Techniques

* **Token‑level loss reweighting**:
  * Upweight rare tokens
  * Downweight punctuation & stopwords
* Slightly lower learning rate than Phase 1

---

### 4.4 Phase 2 Stop Criteria

* Vocabulary richness improves
* Rare words used correctly

Checkpoint again before Phase 3.

---

## 5. Phase 3 — Fine‑Tuning (Your Data)

### 5.1 Objective

Imprint:

* Your domains
* Your writing style
* Your preferences

**Not** to teach grammar.

---

### 5.2 What to Fine‑Tune On

* Your own writing (notes, essays, emails)
* Domain‑specific explanations (AI, ML, systems, etc.)
* Small instruction/response datasets (optional)

Even **10k–100k tokens** can have a strong effect.

---

### 5.3 What NOT to Fine‑Tune On

❌ Large scraped internet dumps
❌ Social media
❌ Mixed‑quality chat logs

Fine‑tuning amplifies bias and noise.

---

### 5.4 Fine‑Tuning Strategy

* Very low learning rate (reduce by 5–10× from Phase 1)
* Few epochs
* Optionally freeze lower layers
* Prefer LoRA / adapters if experimenting

Always keep the Phase‑2 checkpoint intact.

---

### 5.5 Phase 3 Stop Criteria

* Style & domain behavior emerge
* No grammar regression

---

## 6. Context Length Curriculum

Attention cost is quadratic, so context is ramped deliberately.

| Training Progress | Context Length |
| ----------------- | -------------- |
| 0–30%             | 128            |
| 30–70%            | 256            |
| 70–100%           | 512            |

This applies independently within each phase.

---

## 7. Layer Freezing Strategy (Critical)

Backprop through 60 layers is expensive. Freeze early layers once they converge.

| Training Progress | Action                  |
| ----------------- | ----------------------- |
| 0–20%             | Train all layers        |
| 20–50%            | Freeze bottom 20 layers |
| 50%+              | Freeze bottom 40 layers |

Effects:

* ~40–60% reduction in backprop cost
* No loss in reasoning quality
* Improves stability in later training

---

## 8. Optimizer & Learning Rate Strategy

### Optimizer

* **Adafactor** (preferred)
* Reason: lower memory, faster on small hardware

### Learning Rate Schedule

* Warmup: 2–5% of total steps
* Peak LR: ~3e‑4
* Decay: cosine to ~1e‑4

### Fine‑Tuning LR

* Reduce LR by 5–10×
* Prefer very small updates

---

## 9. Batch Size & Gradient Accumulation

Because memory is limited:

* Micro‑batch size: 1–2 sequences
* Gradient accumulation: 8–16 steps

Effective batch size:

```
~8–32 sequences
```

This stabilizes gradients without increasing RAM.

---

## 10. Memory Management

### Mandatory

* Gradient checkpointing **ON** for every Transformer block

### Optional

* Freeze optimizer states for frozen layers
* Clear cached activations aggressively

Result:

* Peak RAM roughly constant with depth

---

## 11. Regularization Strategy

Use **minimal noise** — deep models need stability.

| Component         | Setting          |
| ----------------- | ---------------- |
| MLP dropout       | 0.05–0.10        |
| Attention dropout | 0.0–0.05         |
| Embedding dropout | ≤0.05 (optional) |

Avoid heavy dropout.

---

## 12. Validation & Checkpoints

### Validation

* Periodic validation loss
* Sample generations (short & long)

### Checkpointing

* Save at phase boundaries
* Save before freezing layers
* Never overwrite previous phase checkpoints

---

## 13. Expected Training Time (M2‑class Hardware)

| Phase     | Time             |
| --------- | ---------------- |
| Phase 1   | ~12–14 hours     |
| Phase 2   | ~4–5 hours       |
| Phase 3   | ~1 hour          |
| Overhead  | ~1–2 hours       |
| **Total** | **~18–22 hours** |

This assumes:

* Context ramping
* Layer freezing
* No major restarts

---

## 14. Common Failure Modes & Fixes

| Symptom       | Likely Cause     | Fix                     |
| ------------- | ---------------- | ----------------------- |
| NaNs          | LR too high      | Lower LR, longer warmup |
| Slow training | No freezing      | Freeze early layers     |
| Poor grammar  | Bad Phase‑1 data | Restart Phase 1         |
| Overfitting   | Too much dropout | Reduce noise            |

---

## 15. Golden Rules (Do Not Break)

1. Always checkpoint between phases
2. Never mix fine‑tuning data into pretraining
3. Clean data beats more data
4. Small models magnify data quality
5. Never change architecture mid‑training
6. Never add complexity before baseline trains

---

## 16. Optional Training Enhancements

(Only after baseline trains correctly)

* Deep supervision (auxiliary losses)
* SwiGLU activations
* Learned residual scaling
* Partial parameter sharing across depth
* Training‑only noise & regularization

Do **not** add these during Phase 1.

---

## 17. Easy Training Optimizations

Apply these **high-ROI optimizations** from the start.

### 17.1 torch.compile() (PyTorch 2.0+)

```python
model = torch.compile(model, mode="reduce-overhead")
```

* **Speedup:** 20–40%
* **Cost:** 1 line of code
* Works on MPS (Apple Silicon)

---

### 17.2 BF16 Mixed Precision

Use BF16 instead of FP16:

```python
with torch.autocast(device_type="mps", dtype=torch.bfloat16):
    loss = model(batch)
```

* **Speedup:** 30–50%
* **Benefit:** No loss scaling required (unlike FP16)
* **Requirement:** PyTorch 2.1+

---

### 17.3 Pre‑Tokenization

Tokenize data **once** during preprocessing, not during training.

```
data/processed/       → Text files
data/tokenized/       → Token ID arrays (memmap or numpy)
```

* **Speedup:** 50%+ I/O reduction
* **Implementation:** Save as memory-mapped numpy arrays

---

### 17.4 Sequence Packing

Concatenate short documents to fill context window:

```
[Doc1][EOS][Doc2][EOS][Doc3] → Single packed sequence
```

* **Speedup:** 10–20% (less padding waste)
* **Requirement:** Attention mask handles document boundaries

---

### 17.5 Fused Optimizer Kernels

Use fused implementations where available:

```python
optimizer = torch.optim.AdamW(params, fused=True)
```

* **Speedup:** 10–15%
* **Note:** Check MPS compatibility

---

### 17.6 Combined Impact

| Optimization      | Speedup  | Cumulative |
| ----------------- | -------- | ---------- |
| torch.compile     | 20–40%   | 1.3×       |
| BF16              | 30–50%   | 1.6×       |
| Pre-tokenization  | 50% I/O  | 1.8×       |
| Sequence packing  | 10–20%   | 2.0×       |

**Result:** Training time potentially halved (24h → 12–16h)

---

## 18. Final Principle

> **Grammar is learned from repetition.**
> **Vocabulary is learned from rarity.**
> **Specialization is learned from intent.**
> **Train slowly, freeze aggressively, and let depth do the work.**

If you follow this document end‑to‑end, you will train a **coherent, expressive, and controllable 60M‑parameter LLM**.

---

## END
