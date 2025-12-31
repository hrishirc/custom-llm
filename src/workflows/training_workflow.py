"""Temporal workflow for LLM training orchestration.

Defines the main workflow that orchestrates:
1. Data download
2. Data cleaning (with deduplication)
3. Tokenizer training
4. Data tokenization
5. Multi-phase training (Grammar → Vocabulary → Scientific)
6. Evaluation
"""
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from .activities import (
        load_state_from_db,
        download_dataset,
        clean_dataset,
        train_tokenizer,
        tokenize_data,
        train_phase,
        evaluate_model,
    )


@dataclass
class TrainingParams:
    """Parameters for the training workflow."""
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    db_path: str = "data/training_state.db"
    phases: Optional[List[str]] = None
    skip_download: bool = False
    skip_clean: bool = False
    skip_tokenizer: bool = False
    skip_tokenization: bool = False
    vocab_size: int = 32000
    
    def __post_init__(self):
        if self.phases is None:
            self.phases = ["1", "2", "2b"]


@dataclass
class TrainingResult:
    """Result from the training workflow."""
    completed_phases: List[str]
    final_checkpoint: str
    total_steps: int
    final_loss: Optional[float]


# Retry policy for network operations
DOWNLOAD_RETRY = RetryPolicy(
    maximum_attempts=5,
    initial_interval=timedelta(seconds=10),
    backoff_coefficient=1.0,
    maximum_interval=timedelta(minutes=1),
)

# Retry policy for training (should resume from checkpoint)
TRAINING_RETRY = RetryPolicy(
    maximum_attempts=10,
    initial_interval=timedelta(seconds=30),
    backoff_coefficient=1.0,
    maximum_interval=timedelta(minutes=1),
)


@workflow.defn
class LLMTrainingWorkflow:
    """Durable training workflow with SQLite state mirroring.
    
    State is owned by Temporal but mirrored to SQLite for:
    - Queryable progress monitoring
    - Bootstrap on workflow start
    - External tooling integration
    """
    
    @workflow.run
    async def run(self, params: TrainingParams) -> TrainingResult:
        completed_phases = []
        total_steps = 0
        final_loss = None
        
        # ════════════════════════════════════════════════════════════
        # BOOTSTRAP: Load initial state from SQLite
        # ════════════════════════════════════════════════════════════
        
        initial_state = await workflow.execute_activity(
            load_state_from_db,
            args=[params.db_path],
            start_to_close_timeout=timedelta(seconds=60),
        )
        
        workflow.logger.info(f"Loaded initial state: {len(initial_state.get('downloads', {}))} downloads tracked")
        
        # ════════════════════════════════════════════════════════════
        # STAGE 1: DOWNLOAD DATASETS
        # ════════════════════════════════════════════════════════════
        
        if not params.skip_download:
            # Determine which datasets to download based on phases
            datasets_needed = []
            if "1" in params.phases:
                datasets_needed.append("wikipedia")
            if "2" in params.phases:
                datasets_needed.extend(["pg19", "bookcorpus"])
            if "2b" in params.phases:
                datasets_needed.extend(["pubmed", "philpapers"])
            
            for dataset in datasets_needed:
                # Check if already complete from initial state
                dl_status = initial_state.get("downloads", {}).get(dataset, {}).get("status")
                if dl_status == "complete":
                    workflow.logger.info(f"Skipping {dataset} download (already complete)")
                    continue
                
                await workflow.execute_activity(
                    download_dataset,
                    args=[dataset, params.data_dir, params.db_path],
                    start_to_close_timeout=timedelta(hours=4),
                    heartbeat_timeout=timedelta(minutes=5),
                    retry_policy=DOWNLOAD_RETRY,
                )
        
        # ════════════════════════════════════════════════════════════
        # STAGE 2: CLEAN DATASETS (with deduplication)
        # ════════════════════════════════════════════════════════════
        
        if not params.skip_clean:
            # Clean each downloaded dataset
            datasets_to_clean = []
            if "1" in params.phases:
                datasets_to_clean.append("wikipedia")
            if "2" in params.phases:
                datasets_to_clean.extend(["pg19", "bookcorpus"])
            if "2b" in params.phases:
                datasets_to_clean.extend(["pubmed", "philpapers"])
            
            for dataset in datasets_to_clean:
                stage_name = f"clean_{dataset}"
                stage_status = initial_state.get("stages", {}).get(stage_name, {}).get("status")
                if stage_status == "complete":
                    workflow.logger.info(f"Skipping {dataset} cleaning (already complete)")
                    continue
                
                await workflow.execute_activity(
                    clean_dataset,
                    args=[dataset, params.data_dir, params.db_path],
                    start_to_close_timeout=timedelta(hours=2),
                    heartbeat_timeout=timedelta(minutes=5),
                    retry_policy=DOWNLOAD_RETRY,
                )
        
        # ════════════════════════════════════════════════════════════
        # STAGE 3: TRAIN TOKENIZER (32K vocab BPE)
        # ════════════════════════════════════════════════════════════
        
        if not params.skip_tokenizer:
            tokenizer_status = initial_state.get("stages", {}).get("train_tokenizer", {}).get("status")
            if tokenizer_status != "complete":
                await workflow.execute_activity(
                    train_tokenizer,
                    args=[params.data_dir, params.vocab_size, params.db_path],
                    start_to_close_timeout=timedelta(hours=2),
                    heartbeat_timeout=timedelta(minutes=5),
                )
            else:
                workflow.logger.info("Skipping tokenizer training (already complete)")
        
        # ════════════════════════════════════════════════════════════
        # STAGE 4: TOKENIZE DATA
        # ════════════════════════════════════════════════════════════
        
        if not params.skip_tokenization:
            for phase in params.phases:
                stage_name = f"tokenize_phase{phase}"
                stage_status = initial_state.get("stages", {}).get(stage_name, {}).get("status")
                if stage_status == "complete":
                    workflow.logger.info(f"Skipping phase {phase} tokenization (already complete)")
                    continue
                
                await workflow.execute_activity(
                    tokenize_data,
                    args=[phase, params.data_dir, params.db_path],
                    start_to_close_timeout=timedelta(hours=3),
                    heartbeat_timeout=timedelta(minutes=5),
                )
        
        # ════════════════════════════════════════════════════════════
        # STAGE 5: TRAINING PHASES
        # ════════════════════════════════════════════════════════════
        
        for phase in params.phases:
            phase_status = initial_state.get("phases", {}).get(phase, {}).get("status")
            if phase_status == "complete":
                workflow.logger.info(f"Skipping phase {phase} training (already complete)")
                completed_phases.append(phase)
                continue
            
            result = await workflow.execute_activity(
                train_phase,
                args=[phase, params.data_dir, params.checkpoint_dir, 
                      params.log_dir, params.db_path],
                start_to_close_timeout=timedelta(hours=24),  # Extended for full phase training
                heartbeat_timeout=timedelta(minutes=15),  # Allow for slow training steps
                retry_policy=TRAINING_RETRY,
            )
            
            completed_phases.append(phase)
            total_steps += result.get("steps", 0)
            final_loss = result.get("loss")
        
        # ════════════════════════════════════════════════════════════
        # STAGE 6: EVALUATION
        # ════════════════════════════════════════════════════════════
        
        eval_result = await workflow.execute_activity(
            evaluate_model,
            args=[params.checkpoint_dir, params.db_path],
            start_to_close_timeout=timedelta(hours=1),
            heartbeat_timeout=timedelta(minutes=5),
        )
        
        return TrainingResult(
            completed_phases=completed_phases,
            final_checkpoint=f"{params.checkpoint_dir}/phase2b_scientific_final.pt",
            total_steps=total_steps,
            final_loss=final_loss,
        )
