"""SQLite state database for training progress tracking.

Temporal is the source of truth for workflow state.
SQLite mirrors state for:
- Queryable progress monitoring
- Bootstrap state on workflow start
- Human-readable status
- Debug logging with full history
"""
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import json

# Configure module logger
logger = logging.getLogger(__name__)



@dataclass
class DownloadState:
    dataset: str
    status: str  # 'pending', 'running', 'complete', 'failed'
    file_path: Optional[str] = None
    size_bytes: Optional[int] = None
    downloaded_at: Optional[str] = None


@dataclass
class StageState:
    stage: str
    status: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    details: Optional[Dict] = None


@dataclass
class PhaseState:
    phase: str
    status: str  # 'pending', 'running', 'complete'
    current_step: int = 0
    total_steps: int = 0
    loss: Optional[float] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    checkpoint_path: Optional[str] = None


class StateDB:
    """SQLite database for training state mirroring."""
    
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS downloads (
        dataset TEXT PRIMARY KEY,
        status TEXT NOT NULL DEFAULT 'pending',
        file_path TEXT,
        size_bytes INTEGER,
        downloaded_at TEXT
    );
    
    CREATE TABLE IF NOT EXISTS pipeline_stages (
        stage TEXT PRIMARY KEY,
        status TEXT NOT NULL DEFAULT 'pending',
        started_at TEXT,
        completed_at TEXT,
        details TEXT  -- JSON
    );
    
    CREATE TABLE IF NOT EXISTS training_phases (
        phase TEXT PRIMARY KEY,
        status TEXT NOT NULL DEFAULT 'pending',
        current_step INTEGER DEFAULT 0,
        total_steps INTEGER DEFAULT 0,
        loss REAL,
        started_at TEXT,
        completed_at TEXT,
        checkpoint_path TEXT
    );
    
    CREATE TABLE IF NOT EXISTS workflow_info (
        key TEXT PRIMARY KEY,
        value TEXT
    );
    
    -- Event log for debugging state changes
    CREATE TABLE IF NOT EXISTS event_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        level TEXT NOT NULL,          -- 'DEBUG', 'INFO', 'WARNING', 'ERROR'
        category TEXT NOT NULL,       -- 'download', 'stage', 'phase', 'workflow'
        entity TEXT,                  -- e.g., 'wikipedia', 'phase1', etc.
        action TEXT NOT NULL,         -- 'started', 'completed', 'failed', 'progress'
        old_value TEXT,               -- Previous state (JSON)
        new_value TEXT,               -- New state (JSON)
        message TEXT,
        extra TEXT                    -- Additional context (JSON)
    );
    
    CREATE INDEX IF NOT EXISTS idx_event_log_timestamp ON event_log(timestamp);
    CREATE INDEX IF NOT EXISTS idx_event_log_category ON event_log(category);
    """
    
    def __init__(self, db_path: str = "data/training_state.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema and set pragmas."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self.SCHEMA)
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
    
    def _now(self) -> str:
        """Current timestamp as ISO string."""
        return datetime.now().isoformat()
    
    def _log_event(
        self,
        level: str,
        category: str,
        action: str,
        entity: str = None,
        old_value: Any = None,
        new_value: Any = None,
        message: str = None,
        extra: Dict = None,
    ):
        """Log an event to both Python logger and SQLite.
        
        Args:
            level: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
            category: 'download', 'stage', 'phase', 'workflow'
            action: 'started', 'completed', 'failed', 'progress'
            entity: The specific entity (e.g., 'wikipedia', '1')
            old_value: Previous state (will be JSON serialized)
            new_value: New state (will be JSON serialized)
            message: Human-readable message
            extra: Additional context
        """
        timestamp = self._now()
        
        # Log to Python logger
        log_msg = f"[{category}:{entity}] {action}"
        if message:
            log_msg += f" - {message}"
        
        log_level = getattr(logging, level.upper(), logging.INFO)
        logger.log(log_level, log_msg)
        
        # Log to SQLite
        old_json = json.dumps(old_value) if old_value else None
        new_json = json.dumps(new_value) if new_value else None
        extra_json = json.dumps(extra) if extra else None
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO event_log 
                    (timestamp, level, category, entity, action, old_value, new_value, message, extra)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (timestamp, level, category, entity, action, 
                      old_json, new_json, message, extra_json))
        except Exception as e:
            logger.warning(f"Failed to log event to SQLite: {e}")
    
    # ════════════════════════════════════════════════════════════════
    # Downloads
    # ════════════════════════════════════════════════════════════════
    
    def get_download(self, dataset: str) -> Optional[DownloadState]:
        """Get download state for a dataset."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM downloads WHERE dataset = ?",
                (dataset,)
            ).fetchone()
            if row:
                return DownloadState(**dict(row))
        return None
    
    def get_all_downloads(self) -> Dict[str, DownloadState]:
        """Get all download states."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM downloads").fetchall()
            return {row["dataset"]: DownloadState(**dict(row)) for row in rows}
    
    def set_download(self, state: DownloadState):
        """Update or insert download state."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO downloads 
                (dataset, status, file_path, size_bytes, downloaded_at)
                VALUES (?, ?, ?, ?, ?)
            """, (state.dataset, state.status, state.file_path, 
                  state.size_bytes, state.downloaded_at))
    
    def mark_download_running(self, dataset: str):
        """Mark a download as running."""
        old_state = self.get_download(dataset)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO downloads (dataset, status) VALUES (?, 'running')
                ON CONFLICT(dataset) DO UPDATE SET status = 'running'
            """, (dataset,))
        self._log_event(
            level="INFO",
            category="download",
            entity=dataset,
            action="started",
            old_value=asdict(old_state) if old_state else None,
            new_value={"status": "running"},
            message=f"Download started for {dataset}",
        )
    
    def mark_download_complete(self, dataset: str, file_path: str, size_bytes: int):
        """Mark a download as complete."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO downloads (dataset, status, file_path, size_bytes, downloaded_at)
                VALUES (?, 'complete', ?, ?, ?)
                ON CONFLICT(dataset) DO UPDATE 
                SET status = 'complete', file_path = excluded.file_path, 
                    size_bytes = excluded.size_bytes, downloaded_at = excluded.downloaded_at
            """, (dataset, file_path, size_bytes, self._now()))
        self._log_event(
            level="INFO",
            category="download",
            entity=dataset,
            action="completed",
            new_value={"status": "complete", "file_path": file_path, "size_bytes": size_bytes},
            message=f"Download complete: {dataset} ({size_bytes:,} bytes)",
        )
    
    def mark_download_failed(self, dataset: str, error: str):
        """Mark a download as failed."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO downloads (dataset, status) VALUES (?, 'failed')
                ON CONFLICT(dataset) DO UPDATE SET status = 'failed'
            """, (dataset,))
        self._log_event(
            level="ERROR",
            category="download",
            entity=dataset,
            action="failed",
            new_value={"status": "failed", "error": error},
            message=f"Download failed: {dataset} - {error}",
        )
    
    # ════════════════════════════════════════════════════════════════
    # Pipeline Stages
    # ════════════════════════════════════════════════════════════════
    
    def get_stage(self, stage: str) -> Optional[StageState]:
        """Get pipeline stage state."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM pipeline_stages WHERE stage = ?",
                (stage,)
            ).fetchone()
            if row:
                details = json.loads(row["details"]) if row["details"] else None
                return StageState(
                    stage=row["stage"],
                    status=row["status"],
                    started_at=row["started_at"],
                    completed_at=row["completed_at"],
                    details=details,
                )
        return None
    
    def get_all_stages(self) -> Dict[str, StageState]:
        """Get all pipeline stage states."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM pipeline_stages").fetchall()
            stages = {}
            for row in rows:
                details = json.loads(row["details"]) if row["details"] else None
                stages[row["stage"]] = StageState(
                    stage=row["stage"],
                    status=row["status"],
                    started_at=row["started_at"],
                    completed_at=row["completed_at"],
                    details=details,
                )
            return stages

    def set_stage(self, state: StageState):
        """Update or insert stage state."""
        details_json = json.dumps(state.details) if state.details else None
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO pipeline_stages
                (stage, status, started_at, completed_at, details)
                VALUES (?, ?, ?, ?, ?)
            """, (state.stage, state.status, state.started_at, 
                  state.completed_at, details_json))
    
    def mark_stage_running(self, stage: str):
        """Mark a stage as running."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO pipeline_stages (stage, status, started_at) VALUES (?, 'running', ?)
                ON CONFLICT(stage) DO UPDATE SET status = 'running', started_at = excluded.started_at
            """, (stage, self._now()))
        self._log_event(
            level="INFO",
            category="stage",
            entity=stage,
            action="started",
            message=f"Stage started: {stage}",
        )
    
    def mark_stage_complete(self, stage: str, details: Optional[Dict] = None):
        """Mark a stage as complete."""
        details_json = json.dumps(details) if details else None
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO pipeline_stages (stage, status, completed_at, details) 
                VALUES (?, 'complete', ?, ?)
                ON CONFLICT(stage) DO UPDATE 
                SET status = 'complete', completed_at = excluded.completed_at, details = excluded.details
            """, (stage, self._now(), details_json))
        self._log_event(
            level="INFO",
            category="stage",
            entity=stage,
            action="completed",
            new_value=details,
            message=f"Stage complete: {stage}",
        )
    
    def update_stage_details(self, stage: str, details: Dict):
        """Update stage details without changing status or timestamps."""
        details_json = json.dumps(details)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE pipeline_stages 
                set details = ?
                WHERE stage = ?
            """, (details_json, stage))
    
    def mark_stage_failed(self, stage: str, error: str):
        """Mark a stage as failed."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO pipeline_stages (stage, status, completed_at, details) 
                VALUES (?, 'failed', ?, ?)
                ON CONFLICT(stage) DO UPDATE 
                SET status = 'failed', completed_at = excluded.completed_at, details = excluded.details
            """, (stage, self._now(), json.dumps({"error": error})))
        self._log_event(
            level="ERROR",
            category="stage",
            entity=stage,
            action="failed",
            new_value={"status": "failed", "error": error},
            message=f"Stage failed: {stage} - {error}",
        )
    
    # ════════════════════════════════════════════════════════════════
    # Training Phases
    # ════════════════════════════════════════════════════════════════
    
    def get_phase(self, phase: str) -> Optional[PhaseState]:
        """Get training phase state."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM training_phases WHERE phase = ?",
                (phase,)
            ).fetchone()
            if row:
                return PhaseState(**dict(row))
        return None
    
    def get_all_phases(self) -> Dict[str, PhaseState]:
        """Get all training phase states."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM training_phases").fetchall()
            return {row["phase"]: PhaseState(**dict(row)) for row in rows}
    
    def set_phase(self, state: PhaseState):
        """Update or insert phase state."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO training_phases
                (phase, status, current_step, total_steps, loss, 
                 started_at, completed_at, checkpoint_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (state.phase, state.status, state.current_step, state.total_steps,
                  state.loss, state.started_at, state.completed_at, state.checkpoint_path))
    
    def mark_phase_running(self, phase: str):
        """Mark a phase as running."""
        old_state = self.get_phase(phase)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO training_phases (phase, status, started_at) VALUES (?, 'running', ?)
                ON CONFLICT(phase) DO UPDATE SET status = 'running', started_at = excluded.started_at
            """, (phase, self._now()))
        self._log_event(
            level="INFO",
            category="phase",
            entity=phase,
            action="started",
            old_value=asdict(old_state) if old_state else None,
            message=f"Training phase {phase} started",
        )
    
    def update_phase_progress(self, phase: str, current_step: int, loss: float):
        """Update training progress."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE training_phases 
                SET current_step = ?, loss = ?
                WHERE phase = ?
            """, (current_step, loss, phase))
        # Log progress at DEBUG level (frequent updates)
        self._log_event(
            level="DEBUG",
            category="phase",
            entity=phase,
            action="progress",
            new_value={"step": current_step, "loss": loss},
            message=f"Phase {phase}: step {current_step}, loss={loss:.4f}",
        )
    
    def mark_phase_complete(self, phase: str, checkpoint_path: str):
        """Mark a phase as complete."""
        phase_state = self.get_phase(phase)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE training_phases 
                SET status = 'complete', completed_at = ?, checkpoint_path = ?
                WHERE phase = ?
            """, (self._now(), checkpoint_path, phase))
        self._log_event(
            level="INFO",
            category="phase",
            entity=phase,
            action="completed",
            new_value={"checkpoint_path": checkpoint_path, "steps": phase_state.current_step if phase_state else 0},
            message=f"Training phase {phase} complete: {checkpoint_path}",
        )
    
    # ════════════════════════════════════════════════════════════════
    # Workflow Info
    # ════════════════════════════════════════════════════════════════
    
    def set_workflow_info(self, key: str, value: str):
        """Store workflow metadata."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO workflow_info (key, value)
                VALUES (?, ?)
            """, (key, value))
    
    def get_workflow_info(self, key: str) -> Optional[str]:
        """Get workflow metadata."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT value FROM workflow_info WHERE key = ?",
                (key,)
            ).fetchone()
            return row[0] if row else None
    
    # ════════════════════════════════════════════════════════════════
    # Full State
    # ════════════════════════════════════════════════════════════════
    
    def get_full_state(self) -> Dict[str, Any]:
        """Get complete training state."""
        return {
            "downloads": {k: asdict(v) for k, v in self.get_all_downloads().items()},
            "stages": {k: asdict(v) for k, v in self.get_all_stages().items()},
            "phases": {k: asdict(v) for k, v in self.get_all_phases().items()},
        }


def init_default_state(db: StateDB):
    """Initialize database with default state and known completed work."""
    
    # Downloads (Wikipedia already complete)
    datasets = [
        DownloadState("wikipedia", "complete", 
                      "data/raw/phase1/wikipedia.txt", 19578124898,
                      "2025-12-23T16:11:00"),
        DownloadState("pg19", "pending"),
        DownloadState("bookcorpus", "pending"),
        DownloadState("pubmed", "pending"),
        DownloadState("philpapers", "pending"),
    ]
    for ds in datasets:
        db.set_download(ds)
    
    # Pipeline stages
    stages = [
        StageState("download_wikipedia", "complete"),
        StageState("download_pg19", "pending"),
        StageState("download_bookcorpus", "pending"),
        StageState("download_pubmed", "pending"),
        StageState("download_philpapers", "pending"),
        StageState("clean_wikipedia", "pending"),
        StageState("clean_pg19", "pending"),
        StageState("clean_bookcorpus", "pending"),
        StageState("train_tokenizer", "pending"),
        StageState("tokenize_phase1", "pending"),
        StageState("tokenize_phase2", "pending"),
        StageState("tokenize_phase2b", "pending"),
    ]
    for stage in stages:
        db.set_stage(stage)
    
    # Training phases
    phases = [
        PhaseState("1", "pending", total_steps=50000),
        PhaseState("2", "pending", total_steps=12500),
        PhaseState("2b", "pending", total_steps=6250),
    ]
    for phase in phases:
        db.set_phase(phase)
