# Building a Basic LLM from Scratch (Single‑Node, Research‑Grade)

This document is a **complete, self‑contained specification** you can hand to an LLM, a human engineer, or yourself in the future to build a **working basic Large Language Model**. It is optimized for:

* Single‑machine training (Apple M‑series / laptop‑class hardware)
* ~60M parameter budget
* Research clarity over production shortcuts
* Correctness over hype

Nothing essential is omitted.

---

## 0. What This Document Is (and Is Not)

**This is:**

* A *blueprint* for a real LLM
* Architecture + training + data + optimization + evaluation
* Suitable for first‑principles understanding

**This is NOT:**

* A copy of GPT / LLaMA internals
* A distributed systems guide
* A UI or deployment manual

---

## 1. High‑Level Goal

Train an **autoregressive Transformer language model** that:

* Learns next‑token prediction
* Uses self‑attention
* Can be trained end‑to‑end on a single machine
* Exhibits real reasoning

---

## 2. Core Design Decisions (Frozen Early)

These must be decided **before writing code**.

### 2.1 Training Objective

* **Autoregressive causal language modeling**
* Loss: **Cross‑entropy**
* Predict token *t+1* given tokens *≤t*

No masked LM. No bidirectional context.

---

### 2.2 Tokenization

**Required choices:**

* Tokenization type: **BPE or Unigram LM**
* Vocabulary size: **16k–32k** (smaller = faster, larger = more expressive)
* Include special tokens:

  * `<BOS>`
  * `<EOS>`
  * `<PAD>`
  * `<UNK>` (optional if BPE)

**Constraints:**

* Tokenizer must be trained *before* model training
* Token IDs must be stable forever

---

### 2.3 Model Scale Target

| Property         | Value | Rationale |
| ---------------- | ----- | --------- |
| Total parameters | ~55–60M | Laptop‑trainable budget |
| Layers           | 60    | Deep architecture for reasoning depth |
| Hidden size      | 320   | Balanced with depth |
| Attention heads  | 5     | Standard for this hidden size |
| Head dimension   | 64    | Power of 2 for hardware efficiency |
| MLP ratio        | 2×    | Standard expansion for capacity |
| Context length   | 512   | Derived from Params/Depth ratio |

**Context Length Derivation:**

Using the heuristic `Effective Context ∝ Params / Depth`:
* GPT‑2 Small (117M, 12 layers, 1024 ctx) → ratio ≈ 9.75M/layer
* This model (59M, 60 layers) → ratio ≈ 1M/layer
* Proportional context: `1024 × (1M / 9.75M) ≈ 105 tokens` (minimal)

However, with 2× MLP ratio (lower compute per layer), 512 tokens is sustainable.
The deep architecture compensates by enabling more reasoning steps per token.

---

## 3. Model Architecture (Exact Specification)

### 3.1 Overall Structure

```
Token IDs
  ↓
Embedding Layer
  ↓
[ Transformer Block ] × 60
  ↓
Final LayerNorm
  ↓
LM Head (Linear → Vocab)
```

---

### 3.2 Embedding Layer

* Token embedding matrix: `(vocab_size, hidden_size)`
* Positional encoding:

  * **Rotary (RoPE)** or **ALiBi**
  * No learned positional embeddings

Embedding output shape:

```
(batch, seq_len, hidden_size)
```

---

### 3.3 Transformer Block (Pre‑LN)

Each block contains:

1. LayerNorm
2. Multi‑Head Self‑Attention
3. Residual Add
4. LayerNorm
5. MLP
6. Residual Add

---

### 3.4 Self‑Attention (Causal)

**Parameters:**

* Heads: 5
* Hidden size: 320
* Head dim: 64

**Projections:**

* Q: `(hidden, hidden)`
* K: `(hidden, hidden)`
* V: `(hidden, hidden)`
* Output: `(hidden, hidden)`

**Masking:**

* Strict causal mask
* No token attends to future tokens

**Optional optimizations:**

* Grouped‑Query Attention (GQA)

---

### 3.5 MLP Block

* Expansion: `hidden → 2×hidden → hidden`
* Activation: **GELU** or **SwiGLU**

Weights:

* Up‑projection: `(hidden, 2×hidden)` → `(320, 640)`
* Down‑projection: `(2×hidden, hidden)` → `(640, 320)`

MLP dominates compute cost.

---

### 3.6 Normalization

* **Pre‑LayerNorm only**
* Epsilon: `1e‑5`

Pre‑LN is mandatory for 60‑layer stability.

---

## 4. Parameter Accounting (Sanity Check)

Per layer parameters:

* Attention: `4 × hidden²` = `4 × 320² = 410K`
* MLP: `2 × 2 × hidden² = 4 × hidden²` = `4 × 320² = 410K`

Total per layer:

```
≈ 8 × hidden² ≈ 8 × 320² ≈ ~820K
```

Model body:

```
60 × 820K ≈ 49M
```

Embeddings (vocab=32K):

```
32K × 320 ≈ 10M
```

**Total: ~59M parameters**

---

### 5.1 Weight Initialization (Critical for Deep Models)

Proper initialization prevents gradient explosion/vanishing in 60‑layer networks.

| Component | Initialization | Notes |
|-----------|---------------|-------|
| Token embeddings | `N(0, 0.02)` | Standard LLM practice |
| Positional (if learned) | `N(0, 0.01)` | Not used with RoPE |
| Q, K, V projections | `N(0, 0.02)` | Per‑attention weight |
| Attention output | `N(0, 0.02 / √(2 × 60))` | Scaled for residual depth |
| MLP up‑projection | `N(0, 0.02)` | Standard |
| MLP down‑projection | `N(0, 0.02 / √(2 × 60))` | Scaled for residual depth |
| LayerNorm γ | `1.0` | Identity at init |
| LayerNorm β | `0.0` | Zero bias |
| LM head | Tied to embeddings or `N(0, 0.02)` | Weight tying saves params |

**Residual Scaling Formula:**

For residual projections (attention output, MLP down):
```
std = base_std / √(2 × n_layers)
    = 0.02 / √(120)
    ≈ 0.00183
```

This ensures residual contributions don't explode with depth.

---

### 5.2 Precision

* FP16 or BF16
* Loss scaling enabled

---

### 5.3 Optimizer

**Preferred:** Adafactor

* Lower memory than Adam
* Works well for Transformers

Settings:

* Relative step size: ON
* Warmup: 2–5%
* Weight decay: ≤0.01

---

### 5.4 Learning Rate Schedule

* Linear warmup
* Cosine decay
* Peak LR: `3e‑4`
* Final LR: `1e‑4`

---

### 5.5 Batch & Accumulation

* Micro‑batch size: 1–2
* Gradient accumulation: 8–16
* Effective batch size: 8–32 sequences

---

### 5.6 Gradient Checkpointing

**Mandatory**

* Checkpoint every Transformer block
* Trades compute for memory

Without this, deep models will OOM.

---

### 5.7 Training Execution

For complete training strategy including:

* Phase‑based training (grammar → vocabulary → fine‑tuning)
* Layer freezing schedules
* Context length curriculum
* Expected training times

See [Training_Steps.md](file:///Users/hrishikesh/Repos/custom-llm/specs/Training_Steps.md).

---

## 6. Data Requirements

### 6.1 Dataset Type

* Plain text
* UTF‑8
* One document per line preferred

---

### 6.2 Dataset Size (Minimum)

| Quality      | Tokens   |
| ------------ | -------- |
| Bare minimum | 50–100M  |
| Reasonable   | 300–500M |

More data helps, but depth matters more than scale here.

---

### 6.3 Data Curriculum (Critical)

Start easy, then increase difficulty.

Example:

* Early: simple sentences
* Middle: paragraphs
* Late: mixed complexity

---

## 7. Training Loop (Logical Steps)

1. Load tokenizer
2. Load text
3. Chunk into sequences
4. Shift inputs → targets
5. Forward pass
6. Compute loss
7. Backward pass
8. Gradient accumulation
9. Optimizer step
10. Scheduler step
11. Log metrics

---

## 8. Evaluation (Do Not Skip)

### 8.1 Core Metrics

* Training loss
* Validation loss
* Perplexity

---

### 8.2 Behavioral Tests

* Short completion
* Simple reasoning (2–3 steps)
* Memorization checks

Loss alone is insufficient.

---

## 9. Expected Behavior (Reality Check)

This model **will**:

* Learn grammar
* Learn syntax
* Show shallow reasoning

This model **will NOT**:

* Know facts reliably
* Do deep math
* Compete with large LLMs

That is expected and correct.

---

## 10. Common Failure Modes

* Divergence → LR too high
* NaNs → missing Pre‑LN or FP16 instability
* Slow training → Adam + no freezing
* OOM → no checkpointing

---

## 11. Optional Extensions (After Baseline Works)

* LoRA adapters
* Shared expert pool (MoE‑lite)
* Tool tokens
* Memory tokens

**Never add these before the baseline trains.**

---

## 12. Final Principle (Important)

> A correct small LLM is worth more than a broken large one.

If you follow this document precisely, you will produce a **real LLM**, not a toy.

---

## END
