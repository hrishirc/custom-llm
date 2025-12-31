# Training Datasets Specification

This document specifies the **exact datasets** used for training the 60M parameter LLM, organized by training phase.

---

## Dataset Summary

| Phase | Dataset | Source | Target Tokens | Purpose |
|-------|---------|--------|---------------|---------|
| **1** | English Wikipedia | HuggingFace | 200M | Grammar & structure |
| **2** | PG19 (Project Gutenberg) | HuggingFace | 30M | Classic vocabulary |
| **2** | BookCorpus | HuggingFace | 20M | Modern fiction |
| **2b** | PubMed Abstracts | The Pile | 3M | Scientific reasoning |
| **2b** | PhilPapers | The Pile | 2M | Logical reasoning |
| | **Total** | | **255M** | |

---

## Phase 1 — Grammar & Core Structure

### English Wikipedia

| Property | Value |
|----------|-------|
| Source | `wikipedia` on HuggingFace |
| Version | `20220301.en` |
| Full Size | ~20 GB (~7B tokens) |
| Sampled | ~200M tokens (~3%) |
| Format | Article text only |

**Why Wikipedia:**
- Clean, well-edited prose
- Consistent grammatical structure
- Neutral encyclopedic style
- Wide topic coverage

**Preprocessing:**
- Extract article text (no tables, infoboxes)
- Remove citation markers `[1]`, `[2]`
- Unicode normalize (NFKC)
- Filter short articles (<100 chars)
- Deduplicate

---

## Phase 2 — Vocabulary & Stylistic Range

### PG19 (Project Gutenberg)

| Property | Value |
|----------|-------|
| Source | `pg19` on HuggingFace |
| Full Size | ~11 GB (~4.9B tokens) |
| Sampled | ~30M tokens |
| Content | Public domain books (pre-1919) |

**Why PG19:**
- Rich, diverse vocabulary
- Classic literary prose
- Long-form narrative structure
- No copyright restrictions

**Corpus Characteristics:**
- 28,752 books
- Average 88K tokens per book
- Primarily English literature
- Includes fiction and non-fiction

---

### BookCorpus

| Property | Value |
|----------|-------|
| Source | `bookcorpus` on HuggingFace |
| Full Size | ~5 GB (~985M tokens) |
| Sampled | ~20M tokens |
| Content | Modern self-published books |

**Why BookCorpus:**
- Modern vocabulary and idioms
- Contemporary narrative styles
- Dialog-rich content
- Complements PG19's classical style

**Note:** BookCorpus availability varies; alternative is to increase PG19 sample.

---

## Phase 2b — Scientific & Logical Reasoning

### PubMed Abstracts

| Property | Value |
|----------|-------|
| Source | The Pile (`EleutherAI/pile`) |
| Pile Subset | `PubMed Abstracts` |
| Full Size | ~19 GB (~5B tokens) |
| Sampled | ~3M tokens |
| Content | Biomedical research abstracts |

**Why PubMed Abstracts:**
- Structured scientific writing
- Causal reasoning patterns
- Technical vocabulary
- Concise, information-dense

**Corpus Characteristics:**
- 30M+ publications (1946–present)
- Life sciences, medicine, biology
- Peer-reviewed content
- Hypothesis → method → result structure

---

### PhilPapers

| Property | Value |
|----------|-------|
| Source | The Pile (`EleutherAI/pile`) |
| Pile Subset | `PhilPapers` |
| Full Size | ~2.4 GB (~600M tokens) |
| Sampled | ~2M tokens |
| Content | Philosophy academic papers |

**Why PhilPapers:**
- Logical argumentation
- Abstract reasoning
- Precise language
- Epistemological content

**Corpus Characteristics:**
- Open-access philosophy papers
- Covers ethics, logic, metaphysics
- Argument-driven structure
- High-quality academic prose

---

## Dataset Exclusions

The following are **explicitly excluded**:

| Excluded | Reason |
|----------|--------|
| Code (GitHub) | Damages prose fluency |
| Social media | Noisy, grammatically inconsistent |
| Chat/forums | Informal, fragmented |
| Math-heavy text | Symbolic, not natural language |
| OpenWebText | Too noisy for small model |
| Common Crawl | Quality too variable |

---

## Data Quality Requirements

All datasets undergo:

1. **Unicode normalization** (NFKC)
2. **HTML/markup removal**
3. **URL/email removal**
4. **Deduplication** (MD5 hash)
5. **Language filter** (>70% ASCII)
6. **Length filter** (>100 chars)

---

## Token Budget Rationale

```
Phase 1:   200M tokens — Grammar requires repetition
Phase 2:    50M tokens — Vocabulary requires variety
Phase 2b:    5M tokens — Reasoning exposure (small dose to avoid domain skew)
─────────────────────────────────────────────────────
Total:     255M tokens
```

This follows the **Chinchilla scaling** intuition (~4× tokens per parameter) while staying within single-machine constraints.

---

## Download Commands

```bash
# Phase 1: Wikipedia
python scripts/prepare_data.py --phase 1 --output-dir data

# Phase 2: PG19 + BookCorpus
python scripts/prepare_data.py --phase 2 --output-dir data

# Phase 2b: Scientific reasoning
python scripts/prepare_data.py --phase 2b --output-dir data

# All phases
python scripts/prepare_data.py --phase all --output-dir data
```

---

## References

- Wikipedia: https://huggingface.co/datasets/wikipedia
- PG19: https://huggingface.co/datasets/pg19
- BookCorpus: https://huggingface.co/datasets/bookcorpus
- The Pile: https://pile.eleuther.ai/
- The Pile Paper: Gao et al., 2020 (arXiv:2101.00027)

---

## END
