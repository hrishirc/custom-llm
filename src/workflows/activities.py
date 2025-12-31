"""Temporal activities for LLM training pipeline.

Each activity:
1. Reads initial state from SQLite
2. Performs its task with heartbeats
3. Updates SQLite on progress and completion
"""
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from temporalio import activity

# Ensure src is in path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


from .core import (
    download_core,
    clean_core,
    train_tokenizer_core,
    tokenize_core,
    train_phase_core,
)

@activity.defn
async def load_state_from_db(db_path: str) -> Dict[str, Any]:
    """Load complete training state from SQLite."""
    from src.workflows.state_db import StateDB
    db = StateDB(db_path)
    state = db.get_full_state()
    activity.heartbeat("State loaded from database")
    return state

@activity.defn
async def download_dataset(dataset: str, data_dir: str, db_path: str) -> Dict[str, Any]:
    """Download a single dataset using core logic."""
    from src.workflows.state_db import StateDB
    db = StateDB(db_path)
    db.mark_download_running(dataset)
    db.mark_stage_running(f"download_{dataset}")

    def progress_callback(msg, progress=None, details=None):
        activity.heartbeat(msg)
        if details and "documents" in details:
            # We could update DB here for more granular tracking if desired
            pass

    try:
        result = download_core(dataset, Path(data_dir), progress_callback)
        db.mark_download_complete(dataset, result["file_path"], result["size_bytes"])
        db.mark_stage_complete(f"download_{dataset}", {"file_path": result["file_path"]})
        return result
    except Exception as e:
        from src.workflows.state_db import DownloadState
        db.set_download(DownloadState(dataset, "failed"))
        raise

@activity.defn
async def clean_dataset(dataset: str, data_dir: str, db_path: str) -> Dict[str, Any]:
    """Clean a dataset using core logic."""
    from src.workflows.state_db import StateDB
    db = StateDB(db_path)
    stage_name = f"clean_{dataset}"
    db.mark_stage_running(stage_name)

    # Get resume state from DB
    stage_state = db.get_stage(stage_name)
    resume_state = stage_state.details if stage_state else None

    def progress_callback(msg, progress=None, details=None):
        activity.heartbeat(msg)
        if details:
            db.update_stage_details(stage_name, details)

    result = clean_core(dataset, Path(data_dir), progress_callback, resume_state)
    if result["status"] == "complete":
        db.mark_stage_complete(stage_name, result)
    elif result["status"] == "skipped":
        db.mark_stage_complete(stage_name, {"skipped": True, "reason": result["reason"]})
    return result

@activity.defn
async def train_tokenizer(data_dir: str, vocab_size: int, db_path: str) -> Dict[str, Any]:
    """Train tokenizer using core logic."""
    from src.workflows.state_db import StateDB
    db = StateDB(db_path)
    db.mark_stage_running("train_tokenizer")

    def progress_callback(msg, progress=None, details=None):
        activity.heartbeat(msg)

    result = train_tokenizer_core(Path(data_dir), vocab_size, progress_callback)
    db.mark_stage_complete("train_tokenizer", result)
    return result

@activity.defn
async def tokenize_data(phase: str, data_dir: str, db_path: str) -> Dict[str, Any]:
    """Tokenize data using core logic."""
    from src.workflows.state_db import StateDB
    db = StateDB(db_path)
    stage_name = f"tokenize_phase{phase}"
    db.mark_stage_running(stage_name)

    # Get resume state from DB
    stage_state = db.get_stage(stage_name)
    resume_state = stage_state.details if stage_state else None

    def progress_callback(msg, progress=None, details=None):
        activity.heartbeat(msg)
        if details:
            db.update_stage_details(stage_name, details)

    result = tokenize_core(phase, Path(data_dir), progress_callback, resume_state)
    db.mark_stage_complete(stage_name, result)
    return result

@activity.defn
async def train_phase(
    phase: str,
    data_dir: str,
    checkpoint_dir: str,
    log_dir: str,
    db_path: str,
) -> Dict[str, Any]:
    """Train a single phase using core logic."""
    from src.workflows.state_db import StateDB
    db = StateDB(db_path)
    db.mark_phase_running(phase)

    # Get resume step from DB
    phase_state = db.get_phase(phase)
    resume_step = phase_state.current_step if phase_state else 0

    def progress_callback(msg, progress=None, details=None):
        activity.heartbeat(msg)
        if details and "step" in details:
            db.update_phase_progress(phase, details["step"], details["loss"])

    result = train_phase_core(
        phase, 
        Path(data_dir), 
        Path(checkpoint_dir), 
        Path(log_dir) if log_dir else None,
        progress_callback,
        resume_step
    )
    db.mark_phase_complete(phase, result["checkpoint_path"])
    return result

@activity.defn
async def evaluate_model(checkpoint_dir: str, db_path: str) -> Dict[str, Any]:
    """Evaluate the trained model."""
    import torch
    from src.workflows.state_db import StateDB
    from src.model.config import ModelConfig
    from src.model.llm import create_model
    
    db = StateDB(db_path)
    ckpt_path = Path(checkpoint_dir)
    activity.heartbeat("Starting model evaluation...")
    
    final_ckpt = None
    for phase in ["2b", "2", "1"]:
        phase_name = {"1": "phase1_grammar", "2": "phase2_vocabulary", "2b": "phase2b_scientific"}[phase]
        ckpt = ckpt_path / f"{phase_name}_final.pt"
        if ckpt.exists():
            final_ckpt = ckpt
            break
    
    if not final_ckpt:
        return {"status": "failed", "reason": "no checkpoint found"}
    
    model_config = ModelConfig()
    model = create_model(model_config)
    checkpoint = torch.load(final_ckpt, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    
    prompt = torch.tensor([[1]], device=device)
    generated = model.generate(prompt, max_new_tokens=50, temperature=0.8)
    sample_tokens = generated[0].tolist()
    
    db.set_workflow_info("evaluation_complete", datetime.now().isoformat())
    return {
        "status": "complete",
        "checkpoint": str(final_ckpt),
        "sample_tokens": sample_tokens[:20],
    }
