# AGENTS.md - Collaboration Guidelines for Coding Agents

Welcome, Agent! This repository is optimized for collaboration between human developers and AI coding assistants. This document defines the "Social Contract" for contributing to this project.

## Project Identity
- **Goal**: Train a 60M parameter LLM from scratch on Apple Silicon.
- **Architecture**: Deep (~60 layer) autoregressive Transformer with RoPE, causal masking, and depth-scaled initialization.
- **Tech Stack**: Python, PyTorch (MPS), Temporal (Workflows), SQLite (State Management).

## Core Principles
1. **Idempotency is King**: Long-running tasks (training, tokenization) MUST be idempotent.
2. **State Transparency**: Use the `StateDB` via `src/workflows/state_db.py` to track progress and allow resumes.
3. **Hardware First**: Optimize for Apple M-series hardware using MPS and mixed precision (BF16).
4. **Clean Grammar**: Follow the phase-based training strategy in [Training_Steps.md](file:///Users/hrishikesh/Repos/custom-llm/specs/Training_Steps.md).

## Agent Workflows
- **Planning**: Always create or update an `implementation_plan.md` before significant changes.
- **Execution**:
    - Use `scripts/start_training.py` for managing training workflows.
    - Ensure `temporal_worker.py` is running when initiating workflows.
- **Verification**:
    - Run `scripts/train.py --smoke-test` after model or trainer changes.
    - Maintain 100% pass rate in `tests/` using `pytest`.

## State Management (`StateDB`)
Coding agents should interface with `src/workflows/state_db.py` to:
- Check if a file (e.g., tokenized shard) already exists before starting a download/process.
- Update `last_doc_index` or `last_step_count` for granular resume points.

## Coding Style
- **Formatter**: `black` (line-length: 100)
- **Linter**: `ruff`
- **Imports**: Sorted by `ruff` (isort-compatible).

By following these guidelines, you ensure that your contributions are robust, observable, and aligned with the project's long-term stability.
