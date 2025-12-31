"""Native orchestrator for LLM training pipeline.

Provides a non-Temporal alternative for running the training workflow
using the same core logic and SQLite state tracking.
"""
import time
import logging
import traceback
from pathlib import Path
from typing import List, Optional, Dict, Any
from .state_db import StateDB
from .core import (
    download_core,
    clean_core,
    train_tokenizer_core,
    tokenize_core,
    train_phase_core,
)

logger = logging.getLogger(__name__)

class NativeOrchestrator:
    def __init__(
        self,
        db_path: str = "data/training_state.db",
        data_dir: str = "data",
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        phases: Optional[List[str]] = None,
        max_retries: int = 3
    ):
        self.db = StateDB(db_path)
        self.data_dir = Path(data_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.phases = phases or ["1", "2", "2b"]
        self.max_retries = max_retries

    def _progress_callback(self, stage: str, category: str, entity: str = None):
        """Returns a callback that updates the DB and logs progress."""
        def callback(msg: str, progress: Optional[float] = None, details: Optional[Dict[str, Any]] = None):
            # Log to file
            logger.info(f"[{stage}] {msg}")
            # Also print to console for visibility
            if progress is not None:
                print(f"  → [{stage}] {msg} ({progress:.1f}%)")
            
            if details:
                if category == "stage":
                    self.db.update_stage_details(entity or stage, details)
                elif category == "phase":
                    self.db.update_phase_progress(entity, details.get("step", 0), details.get("loss", 0.0))
            
            # Use event log for granular progress if message changed significantly
            self.db._log_event(
                level="DEBUG",
                category=category,
                entity=entity or stage,
                action="progress",
                message=msg,
                extra=details
            )
        return callback

    def run(self):
        """Run the entire pipeline."""
        logger.info("Starting Native Training Orchestrator")
        
        try:
            # 1. Downloads
            datasets_needed = []
            if "1" in self.phases: datasets_needed.append("wikipedia")
            if "2" in self.phases: datasets_needed.extend(["pg19", "bookcorpus"])
            if "2b" in self.phases: datasets_needed.extend(["pubmed", "philpapers"])

            for dataset in datasets_needed:
                dl_status = self.db.get_download(dataset)
                if dl_status and dl_status.status == "complete":
                    logger.info(f"Skipping {dataset} download (complete)")
                    continue
                
                # Skip if already failed (requires manual reset to retry)
                if dl_status and dl_status.status == "failed":
                    logger.warning(f"Skipping {dataset} download (previously failed)")
                    continue
                
                self.db.mark_download_running(dataset)
                self.db.mark_stage_running(f"download_{dataset}")
                
                res = download_core(
                    dataset, 
                    self.data_dir, 
                    self._progress_callback(f"download_{dataset}", "download", dataset)
                )
                
                if res["status"] == "failed":
                    logger.error(f"Download failed for {dataset}: {res.get('error', 'Unknown error')}")
                    self.db.mark_download_failed(dataset, res.get("error", "Download returned 0 bytes"))
                    self.db.mark_stage_failed(f"download_{dataset}", res.get("error"))
                else:
                    self.db.mark_download_complete(dataset, res["file_path"], res["size_bytes"])
                    self.db.mark_stage_complete(f"download_{dataset}", {"file_path": res["file_path"]})

            # 2. Cleaning
            for dataset in datasets_needed:
                stage_name = f"clean_{dataset}"
                status = self.db.get_stage(stage_name)
                if status and status.status == "complete":
                    continue
                
                resume_state = status.details if status else None
                self.db.mark_stage_running(stage_name)
                res = clean_core(
                    dataset, 
                    self.data_dir, 
                    self._progress_callback(stage_name, "stage", stage_name),
                    resume_state
                )
                if res["status"] == "complete":
                    self.db.mark_stage_complete(stage_name, res)
                else:
                    self.db.mark_stage_complete(stage_name, {"skipped": True})

            # 3. Tokenizer
            status = self.db.get_stage("train_tokenizer")
            if not status or status.status != "complete":
                self.db.mark_stage_running("train_tokenizer")
                res = train_tokenizer_core(
                    self.data_dir, 
                    32000, 
                    self._progress_callback("train_tokenizer", "stage", "train_tokenizer")
                )
                self.db.mark_stage_complete("train_tokenizer", res)

            # 4. Tokenization
            for phase in self.phases:
                stage_name = f"tokenize_phase{phase}"
                status = self.db.get_stage(stage_name)
                if status and status.status == "complete":
                    continue
                
                resume_state = status.details if status else None
                self.db.mark_stage_running(stage_name)
                res = tokenize_core(
                    phase, 
                    self.data_dir, 
                    self._progress_callback(stage_name, "stage", stage_name),
                    resume_state
                )
                self.db.mark_stage_complete(stage_name, res)

            # 5. Training
            for phase in self.phases:
                status = self.db.get_phase(phase)
                if status and status.status == "complete":
                    continue
                
                resume_step = status.current_step if status else 0
                self.db.mark_phase_running(phase)
                res = train_phase_core(
                    phase,
                    self.data_dir,
                    self.checkpoint_dir,
                    self.log_dir,
                    self._progress_callback(f"train_phase_{phase}", "phase", phase),
                    resume_step
                )
                self.db.mark_phase_complete(phase, res["checkpoint_path"])

            logger.info("Native Pipeline Completed Successfully")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            logger.error(traceback.format_exc())
            raise
