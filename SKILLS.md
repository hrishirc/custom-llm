# SKILLS.md - Command & Skill Reference for Coding Agents

This document lists the essential commands, infrastructure workflows, and technical skills required to manage the `custom-llm` project effectively.

## 1. Environment & Infrastructure

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run development dependencies (if any)
pip install -e ".[dev]"
```

### Temporal Orchestration
The project uses Temporal for workflow management. Ensure the server and worker are running.
```bash
# Start Temporal Server (Dev Mode)
# Ensure data/ directory exists for DB
temporal server start-dev --db-filename data/temporal_state.db

# Start Temporal Worker
python scripts/temporal_worker.py
```

## 2. Training Workflows

### Full Training Pipeline
Starts all phases (Grammar, Vocabulary, Specialization).
```bash
python scripts/start_training.py start
```

### Specific Phase Training
```bash
# Phase 1 only
python scripts/start_training.py start --phases 1

# Phase 1 and 2
python scripts/start_training.py start --phases 1,2
```

### Monitoring & Control
```bash
# List workflows
python scripts/start_training.py list

# Check status
python scripts/start_training.py status <workflow-id>

# Cancel workflow
python scripts/start_training.py cancel <workflow-id>
```

## 3. Data Preparation
```bash
# Prepare all data
python scripts/prepare_data.py --output-dir data

# Prepare specific phase
python scripts/prepare_data.py --phase 1 --output-dir data

# Sample limited data (for testing)
python scripts/prepare_data.py --max-samples 1000 --output-dir data/test
```

## 4. Model Verification & Testing

### Smoke Test
Verifies model architecture, forward/backward pass, and generation compatibility.
```bash
python scripts/train.py --smoke-test
```

### Unit Tests
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/model/test_attention.py
```

## 5. Optimization & Performance
- **MPS (Apple Silicon)**: Always prefer `mps` device if available (`torch.backends.mps.is_available()`).
- **Mixed Precision**: Use `torch.autocast(device_type="mps", dtype=torch.bfloat16)` inside forward passes.
- **Layer Freezing**: Use `model.freeze_layers(n)` during phase transitions to save compute.

## 6. Logging & Metrics
- **TensorBoard**: View progress at `logs/`.
- **System Logs**: Follow worker logs with `tail -f logs/worker/worker.log`.
- **State DB**: Query the training state directly:
  ```bash
  sqlite3 data/training_state.db "SELECT * FROM training_phases"
  ```
