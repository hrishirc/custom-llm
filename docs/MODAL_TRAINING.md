# Modal GPU Training Guide

Deploy your LLM training to Modal's cloud GPUs for faster training.

## Prerequisites

1. Install Modal:
```bash
pip install modal
```

2. Authenticate:
```bash
modal token new
```

## Quick Start

### Step 1: Upload Your Data

First, upload your tokenized data to Modal's persistent volume:

```bash
# Create/update the volume with your data
modal volume put llm-training-data data/tokenized/ tokenized/
modal volume put llm-training-data data/tokenizer.json tokenizer.json

# Optional: Upload existing checkpoints for resume
modal volume put llm-training-data checkpoints/ checkpoints/

# Optional: Upload state DB for resume
modal volume put llm-training-data data/training_state.db training_state.db
```

### Step 2: Run Training

```bash
# Run full training pipeline on A10G GPU
modal run scripts/modal_train.py --command train
```

### Step 3: Download Results

```bash
# List available checkpoints
modal run scripts/modal_train.py --command download

# Download checkpoints locally
modal volume get llm-training-data checkpoints/ ./checkpoints_modal/
```

## GPU Options

Edit `scripts/modal_train.py` to change GPU type:

| GPU | Price | Speed (est.) | Memory |
|-----|-------|--------------|--------|
| T4 | ~$0.59/hr | ~3,000 tok/s | 16 GB |
| A10G | ~$1.10/hr | ~5,000 tok/s | 24 GB |
| A100 | ~$3.70/hr | ~10,000 tok/s | 40 GB |

## Estimated Costs

For your 60M parameter model:

| Phase | Steps | A10G Time | A10G Cost |
|-------|-------|-----------|-----------|
| Phase 1 | 50,000 | ~10 hours | ~$11 |
| Phase 2 | 12,500 | ~2.5 hours | ~$3 |
| Phase 2b | 6,250 | ~1.2 hours | ~$1.5 |
| **Total** | | ~14 hours | **~$15** |

*Note: Modal offers $30/month free credits!*

## Monitoring

View logs in real-time:
```bash
modal logs custom-llm-training
```

## Troubleshooting

### "Volume not found"
Create the volume first:
```bash
modal volume create llm-training-data
```

### Training disconnected
Your checkpoints are saved in the persistent volume. Just re-run:
```bash
modal run scripts/modal_train.py --command train
```

### GPU unavailable
Try a different GPU type in `modal_train.py` or wait a few minutes.

## Local vs Modal

| Feature | Local (MPS) | Modal (A10G) |
|---------|-------------|--------------|
| Speed | ~1,000 tok/s | ~5,000 tok/s |
| Phase 1 time | ~52 hours | ~10 hours |
| Cost | Free (electricity) | ~$11 |
| Reliability | Always available | Cloud service |
