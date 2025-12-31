# Training‑Only Innovations for Higher Expressivity (Constant RAM & Inference Cost)

This document enumerates **architectural and training innovations** that:

* ✅ **Increase model expressivity & quality**
* ❌ **Do NOT increase parameter count / RAM**
* ❌ **Do NOT increase inference‑time FLOPs or latency**
* ✅ **May increase training time, steps, or complexity**

These techniques are ideal for **small, deep LLMs** trained on single‑node hardware.

---

## How to Use This Document

* Treat each item as an **independent toggle**
* Implement them **incrementally**, never all at once
* Keep a clean baseline before enabling any item

---

# CATEGORY A — Extra Computation During Training Only (Highest ROI)

These add learning signal **without touching inference**.

---

## A1. Structural Curriculum Learning

**What:**
Gradually increase *task structure difficulty*, not just data difficulty.

**Examples:**

* Early training: next‑token prediction only
* Later: span reconstruction, sentence reordering, compression

**Why it works:**
Forces richer intermediate representations.

**Inference impact:** None

---

## A2. Auxiliary Losses (Dropped at Inference)

**What:**
Add extra losses that shape internal representations.

**Examples:**

* Contrastive loss on hidden states
* Predict sentence order
* Predict token‑level attributes (length, type)

**Implementation note:**
Auxiliary heads are **deleted after training**.

**Inference impact:** None

---

## A3. Layer‑Wise Deep Supervision

**What:**
Attach lightweight prediction heads to intermediate layers during training.

**Why:**

* Stronger gradients
* Better abstraction per layer
* Faster convergence

**After training:**
Remove all intermediate heads.

**Inference impact:** None

---

# CATEGORY B — Smarter Parameter Reuse

Reuse existing parameters to simulate more computation.

---

## B1. Partial Parameter Sharing Across Depth

**What:**
Reuse weights across layers.

**Examples:**

* Share MLP weights every 2 layers
* Share attention weights every 4 layers
* Keep LayerNorms unique

**Effect:**
Encourages iterative refinement and abstraction reuse.

**Inference impact:** None

---

## B2. Recurrent Refinement (Training‑Only)

**What:**
Apply the same layer multiple times *during training only*.

**Procedure:**

* Training: run layer twice, backprop through both
* Inference: run layer once

**Why:**
Teaches self‑correction and refinement.

**Inference impact:** None

---

# CATEGORY C — Better Nonlinearity & Gating

Improve expressivity *per parameter*.

---

## C1. SwiGLU Activation

**What:**
Replace GELU with SwiGLU in MLP blocks.

**Why:**

* Gating improves selectivity
* Better gradient flow
* Used in modern LLMs

**Inference impact:** Same FLOPs as GELU

---

## C2. Learned Residual Scaling

**What:**
Replace:

```
x + f(x)
```

With:

```
x + α · f(x)
```

Where α is learned per layer.

**Why:**

* Stabilizes deep training
* Improves representation calibration

**Inference impact:** Negligible

---

# CATEGORY D — Cleaner Internal Representations

Improve quality without adding capacity.

---

## D1. Representation Sparsity Regularization

**What:**
Encourage sparse activations.

**Methods:**

* L1 penalty on activations
* KL / entropy regularization

**Why:**

* Reduces interference
* Improves generalization

**Inference impact:** None

---

## D2. Noise During Training Only

**What:**
Inject noise during training.

**Examples:**

* Activation noise
* Attention noise
* Drop‑path (layer drop)

**After training:**
Noise disabled.

**Inference impact:** None

---

# CATEGORY E — Better Optimization Signal

Make gradients more informative.

---

## E1. Token‑Level Loss Reweighting

**What:**
Reweight loss per token type.

**Examples:**

* Downweight punctuation
* Upweight rare or structural tokens

**Why:**
Improves sample efficiency and representation quality.

**Inference impact:** None

---

## E2. Self‑Distillation (Same Model)

**What:**
Periodically distill the model into itself.

**Procedure:**

1. Freeze model
2. Generate soft targets
3. Train to match softened logits

**Why:**
Improves calibration and consistency.

**Inference impact:** None

---

# CATEGORY F — Structured Attention Improvements (Advanced)

---

## F1. Attention Head Diversity Regularization

**What:**
Penalize correlated attention patterns across heads.

**Why:**
Encourages specialization without adding heads.

**Inference impact:** None

---

## F2. Latent Planning Tokens (Training‑Only)

**What:**
Insert special hidden tokens during training to encourage planning.

**Key rule:**

* Tokens removed at inference

**Why:**
Encourages latent reasoning without chain‑of‑thought leakage.

**Inference impact:** None

---

# What NOT to Do Under These Constraints

❌ Increase hidden size
❌ Increase number of heads
❌ Increase inference depth
❌ Add trainable MoE experts
❌ Add tool execution at inference

---

# Recommended Adoption Order (Safe → Aggressive)

1. SwiGLU + residual scaling
2. Structural curriculum
3. Deep supervision
4. Partial parameter sharing
5. Token‑level loss shaping
6. Self‑distillation
7. Advanced attention regularization

---

## Final Principle

> **Model quality improves fastest when you increase *how hard training is*, not how big the model is.**

This document defines the **entire design space** that respects your constraints.

---

## END
