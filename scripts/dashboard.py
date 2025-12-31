"""Enhanced LLM Training Dashboard with comprehensive monitoring.

Provides:
- Training state with computed fields (ETA, elapsed time)
- System resource metrics (GPU, memory)
- Disk usage for data directories
- Loss history for sparklines
- Health indicators
"""
import os
import sys
import sqlite3
import json
import shutil
import psutil
from pathlib import Path
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.workflows.state_db import StateDB

app = FastAPI(title="LLM Training Dashboard")

DB_PATH = os.getenv("DB_PATH", "data/training_state.db")
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
UI_DIR = Path(__file__).parent.parent / "ui"

# Stage weights for overall progress calculation
STAGE_WEIGHTS = {
    "download": 0.10,
    "clean": 0.05,
    "train_tokenizer": 0.05,
    "tokenize": 0.10,
    "phase_1": 0.40,
    "phase_2": 0.15,
    "phase_2b": 0.10,
    "evaluation": 0.05,
}

# Expected durations for ETA calculation (in seconds)
EXPECTED_DURATIONS = {
    "download": 3600 * 2,      # 2 hours per dataset
    "clean": 1800,             # 30 min per dataset
    "train_tokenizer": 3600,   # 1 hour
    "tokenize": 3600,          # 1 hour per phase
    "phase_1": 3600 * 10,      # 10 hours
    "phase_2": 3600 * 3,       # 3 hours
    "phase_2b": 3600 * 2,      # 2 hours
    "evaluation": 1800,        # 30 min
}


def parse_timestamp(ts: str) -> Optional[datetime]:
    """Parse ISO timestamp string to datetime."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except:
        return None


def format_duration(seconds: float) -> str:
    """Format seconds to human readable duration."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def calculate_elapsed(started_at: str) -> Optional[str]:
    """Calculate elapsed time from start timestamp."""
    start = parse_timestamp(started_at)
    if not start:
        return None
    elapsed = (datetime.now() - start).total_seconds()
    return format_duration(elapsed)


def calculate_eta(current_step: int, total_steps: int, started_at: str) -> Optional[str]:
    """Calculate ETA based on current progress rate."""
    if not started_at or current_step == 0 or total_steps == 0:
        return None
    
    start = parse_timestamp(started_at)
    if not start:
        return None
    
    elapsed = (datetime.now() - start).total_seconds()
    if elapsed < 10:  # Need at least 10 seconds of data
        return None
    
    rate = current_step / elapsed  # steps per second
    remaining_steps = total_steps - current_step
    if rate > 0:
        remaining_seconds = remaining_steps / rate
        return format_duration(remaining_seconds)
    return None


def get_pipeline_stages(state: Dict) -> List[Dict]:
    """Get ordered pipeline stages with status."""
    stages = []
    
    # Downloads
    downloads = state.get("downloads", {})
    dl_complete = sum(1 for d in downloads.values() if d.get("status") == "complete")
    dl_total = len(downloads)
    dl_running = any(d.get("status") == "running" for d in downloads.values())
    stages.append({
        "id": "download",
        "label": "Download",
        "status": "complete" if dl_complete == dl_total else ("running" if dl_running else "pending"),
        "progress": dl_complete / dl_total if dl_total > 0 else 0
    })
    
    # Cleaning
    all_stages = state.get("stages", {})
    clean_stages = {k: v for k, v in all_stages.items() if k.startswith("clean_")}
    clean_complete = sum(1 for s in clean_stages.values() if s.get("status") == "complete")
    clean_total = len(clean_stages) if clean_stages else 5
    clean_running = any(s.get("status") == "running" for s in clean_stages.values())
    stages.append({
        "id": "clean",
        "label": "Clean",
        "status": "complete" if clean_complete == clean_total else ("running" if clean_running else "pending"),
        "progress": clean_complete / clean_total if clean_total > 0 else 0
    })
    
    # Tokenizer
    tokenizer = all_stages.get("train_tokenizer", {})
    stages.append({
        "id": "train_tokenizer",  # Fixed ID to match weights
        "label": "Tokenizer",
        "status": tokenizer.get("status", "pending"),
        "progress": 1.0 if tokenizer.get("status") == "complete" else 0
    })
    
    # Tokenization
    tok_stages = {k: v for k, v in all_stages.items() if k.startswith("tokenize_")}
    tok_complete = sum(1 for s in tok_stages.values() if s.get("status") == "complete")
    tok_total = len(tok_stages) if tok_stages else 3
    tok_running = any(s.get("status") == "running" for s in tok_stages.values())
    stages.append({
        "id": "tokenize",
        "label": "Tokenize",
        "status": "complete" if tok_complete == tok_total else ("running" if tok_running else "pending"),
        "progress": tok_complete / tok_total if tok_total > 0 else 0
    })
    
    # Training phases
    phases = state.get("phases", {})
    for phase_id in ["1", "2", "2b"]:
        phase = phases.get(phase_id, {})
        current = phase.get("current_step", 0)
        total = phase.get("total_steps", 1)
        stages.append({
            "id": f"phase_{phase_id}",
            "label": f"Phase {phase_id}",
            "status": phase.get("status", "pending"),
            "progress": current / total if total > 0 else 0
        })
    
    # Evaluation (after all phases)
    eval_stage = all_stages.get("evaluation", {})
    stages.append({
        "id": "evaluation",
        "label": "Eval",
        "status": eval_stage.get("status", "pending"),
        "progress": 1.0 if eval_stage.get("status") == "complete" else 0
    })
    
    return stages


def calculate_weighted_progress(stages: List[Dict]) -> float:
    """Calculate overall progress with weighted stages."""
    total_weight = 0
    completed_weight = 0
    
    for stage in stages:
        stage_id = stage["id"]
        weight = STAGE_WEIGHTS.get(stage_id, 0.05)
        total_weight += weight
        
        if stage["status"] == "complete":
            completed_weight += weight
        elif stage["status"] == "running":
            # Add partial progress for running stages
            completed_weight += weight * stage.get("progress", 0)
    
    return (completed_weight / total_weight * 100) if total_weight > 0 else 0


# Simple cache for expensive operations
_cache = {
    "disk_usage": {"time": 0, "data": None},
    "gpu_metrics": {"time": 0, "data": None}
}


def detect_health_issues(state: Dict) -> List[Dict]:
    """Detect potential health issues."""
    issues = []
    
    # Check for stalled training
    phases = state.get("phases", {})
    for phase_id, phase in phases.items():
        if phase.get("status") == "running":
            started = parse_timestamp(phase.get("started_at"))
            if started:
                elapsed = (datetime.now() - started).total_seconds()
                current = phase.get("current_step", 0)
                # If running for >5 min with no progress
                if elapsed > 300 and current == 0:
                    issues.append({
                        "level": "warning",
                        "message": f"Phase {phase_id} may be stalled (no progress in {format_duration(elapsed)})"
                    })
    
    # Check memory usage
    try:
        mem = psutil.virtual_memory()
        if mem.percent > 90:
            issues.append({
                "level": "error",
                "message": f"High memory usage: {mem.percent:.1f}%"
            })
        elif mem.percent > 80:
            issues.append({
                "level": "warning",
                "message": f"Memory usage elevated: {mem.percent:.1f}%"
            })
    except:
        pass
    
    # Check disk space
    try:
        disk = shutil.disk_usage(DATA_DIR)
        used_pct = disk.used / disk.total * 100
        if used_pct > 95:
            issues.append({
                "level": "error",
                "message": f"Critical disk space: {100 - used_pct:.1f}% free"
            })
        elif used_pct > 85:
            issues.append({
                "level": "warning",
                "message": f"Low disk space: {100 - used_pct:.1f}% free"
            })
    except:
        pass
    
    return issues


def get_current_activity(state: Dict) -> str:
    """Determine current activity description."""
    # Check training phases first
    phases = state.get("phases", {})
    for phase_id in ["1", "2", "2b"]:
        phase = phases.get(phase_id, {})
        if phase.get("status") == "running":
            return f"Training Phase {phase_id}"
    
    # Check stages
    stages = state.get("stages", {})
    for stage_id, stage in stages.items():
        if stage.get("status") == "running":
            return stage_id.replace("_", " ").title()
    
    # Check downloads
    downloads = state.get("downloads", {})
    for ds_id, ds in downloads.items():
        if ds.get("status") == "running":
            return f"Downloading {ds_id}"
    
    # Check if anything is complete
    all_complete = all(
        d.get("status") == "complete" for d in downloads.values()
    ) and all(
        s.get("status") == "complete" for s in stages.values()
    ) and all(
        p.get("status") == "complete" for p in phases.values()
    )
    
    return "Complete ✓" if all_complete else "Ready"


def get_latest_progress(db_path: str) -> Dict[str, Dict]:
    """Get latest progress message for each running entity from event_log."""
    progress_map = {}
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            # Get latest progress event for each entity
            rows = conn.execute("""
                SELECT entity, message, extra, timestamp
                FROM event_log
                WHERE action = 'progress' AND timestamp > datetime('now', '-5 minutes')
                ORDER BY timestamp DESC
            """).fetchall()
            
            for row in rows:
                entity = row["entity"]
                if entity and entity not in progress_map:
                    extra = json.loads(row["extra"]) if row["extra"] else {}
                    progress_map[entity] = {
                        "message": row["message"],
                        "timestamp": row["timestamp"],
                        "progress": extra.get("progress"),
                        "shards_completed": extra.get("shards_completed"),
                        "documents": extra.get("documents"),
                    }
    except Exception as e:
        pass  # Fail silently, this is enhancement only
    return progress_map

@app.get("/api/state")
async def get_state():
    """Get full training state with computed fields."""
    db = StateDB(DB_PATH)
    raw_state = db.get_full_state()
    
    # Get latest progress for running entities
    latest_progress = get_latest_progress(DB_PATH)
    
    # Enhance downloads with latest progress
    for ds_id, ds in raw_state.get("downloads", {}).items():
        if ds.get("status") == "running":
            progress_info = latest_progress.get(ds_id, {})
            ds["current_progress"] = progress_info.get("message")
            ds["progress_pct"] = progress_info.get("progress")
            ds["shards_done"] = progress_info.get("shards_completed")
            ds["documents_done"] = progress_info.get("documents")
    
    # Enhance phases with computed fields
    for phase_id, phase in raw_state.get("phases", {}).items():
        phase["elapsed"] = calculate_elapsed(phase.get("started_at"))
        phase["eta"] = calculate_eta(
            phase.get("current_step", 0),
            phase.get("total_steps", 0),
            phase.get("started_at")
        )
    
    # Enhance stages with elapsed time and progress
    for stage_id, stage in raw_state.get("stages", {}).items():
        stage["elapsed"] = calculate_elapsed(stage.get("started_at"))
        if stage.get("status") == "running":
            # Extract entity name from stage_id (e.g., download_pubmed -> pubmed)
            entity = stage_id.split("_", 1)[-1] if "_" in stage_id else stage_id
            progress_info = latest_progress.get(entity, {})
            stage["current_progress"] = progress_info.get("message")
            stage["progress_pct"] = progress_info.get("progress")
    
    # Add pipeline stages
    pipeline = get_pipeline_stages(raw_state)
    raw_state["pipeline"] = pipeline
    
    # Add weighted progress
    raw_state["overall_progress"] = calculate_weighted_progress(pipeline)
    
    # Add current activity with details
    raw_state["current_activity"] = get_current_activity(raw_state)
    
    # Enhance current_activity with progress if available
    for ds_id, ds in raw_state.get("downloads", {}).items():
        if ds.get("status") == "running" and ds.get("current_progress"):
            raw_state["current_activity"] = ds["current_progress"]
            break
    
    # Add health issues
    raw_state["health_issues"] = detect_health_issues(raw_state)
    
    # Get current loss from latest running phase
    current_loss = None
    for phase_id in ["1", "2", "2b"]:
        phase = raw_state.get("phases", {}).get(phase_id, {})
        if phase.get("status") == "running" and phase.get("loss"):
            current_loss = phase.get("loss")
            break
    raw_state["current_loss"] = current_loss
    
    # Calculate total ETA
    raw_state["total_eta"] = None
    for stage in pipeline:
        if stage["status"] == "running":
            phase_id = stage["id"].replace("phase_", "")
            phase = raw_state.get("phases", {}).get(phase_id, {})
            if phase.get("eta"):
                raw_state["total_eta"] = phase["eta"]
            break
    
    return raw_state


@app.get("/api/events")
async def get_events(limit: int = 50, category: Optional[str] = None, level: Optional[str] = None):
    """Get events with optional filtering."""
    db = StateDB(DB_PATH)
    with sqlite3.connect(db.db_path) as conn:
        conn.row_factory = sqlite3.Row
        
        query = "SELECT * FROM event_log WHERE 1=1"
        params = []
        
        if category:
            query += " AND category = ?"
            params.append(category)
        
        if level:
            query += " AND level = ?"
            params.append(level)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


@app.get("/api/loss-history")
async def get_loss_history(phase: Optional[str] = None, limit: int = 100):
    """Get loss history for sparkline visualization."""
    db = StateDB(DB_PATH)
    with sqlite3.connect(db.db_path) as conn:
        conn.row_factory = sqlite3.Row
        
        query = """
            SELECT timestamp, entity as phase, new_value
            FROM event_log 
            WHERE category = 'phase' AND action = 'progress'
        """
        params = []
        
        if phase:
            query += " AND entity = ?"
            params.append(phase)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        rows = conn.execute(query, params).fetchall()
        
        history = []
        for row in rows:
            try:
                data = json.loads(row["new_value"]) if row["new_value"] else {}
                history.append({
                    "timestamp": row["timestamp"],
                    "phase": row["phase"],
                    "step": data.get("step", 0),
                    "loss": data.get("loss")
                })
            except:
                continue
        
        return list(reversed(history))  # Chronological order


@app.get("/api/metrics")
async def get_metrics():
    """Get system resource metrics."""
    metrics = {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory": {
            "total": psutil.virtual_memory().total,
            "used": psutil.virtual_memory().used,
            "percent": psutil.virtual_memory().percent
        },
        "gpu": None
    }
    
    # Try to get GPU metrics if available
    try:
        import subprocess
        # Cache GPU metrics for 2 seconds to avoid spamming nvidia-smi
        import time
        now = time.time()
        if now - _cache["gpu_metrics"]["time"] < 2 and _cache["gpu_metrics"]["data"]:
             metrics["gpu"] = _cache["gpu_metrics"]["data"]
        else:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(",")
                if len(parts) >= 3:
                    gpu_data = {
                        "utilization": float(parts[0].strip()),
                        "memory_used": float(parts[1].strip()) * 1024 * 1024,  # MB to bytes
                        "memory_total": float(parts[2].strip()) * 1024 * 1024
                    }
                    metrics["gpu"] = gpu_data
                    _cache["gpu_metrics"] = {"time": now, "data": gpu_data}
    except:
        # GPU not available or nvidia-smi not found
        pass
    
    return metrics


@app.get("/api/disk")
async def get_disk_usage():
    """Get disk usage for data directories. Cached for 60s."""
    import time
    now = time.time()
    if now - _cache["disk_usage"]["time"] < 60 and _cache["disk_usage"]["data"]:
        return _cache["disk_usage"]["data"]

    def get_dir_size_fast(path: Path) -> int:
        """Estimate size without full traversal if possible, or use du."""
        if not path.exists():
            return 0
        try:
            # Try using du command for speed on Unix systems
            import subprocess
            result = subprocess.run(['du', '-sk', str(path)], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return int(result.stdout.split()[0]) * 1024
        except:
            pass
            
        # Fallback to simple traversal (slow for many files)
        total = 0
        try:
            for entry in path.rglob("*"):
                if entry.is_file():
                    total += entry.stat().st_size
        except:
            pass
        return total
    
    usage = {
        "raw": get_dir_size_fast(DATA_DIR / "raw"),
        "cleaned": get_dir_size_fast(DATA_DIR / "cleaned"),
        "tokenized": get_dir_size_fast(DATA_DIR / "tokenized"),
        "checkpoints": get_dir_size_fast(Path("checkpoints")),
        "total_available": shutil.disk_usage(DATA_DIR).free if DATA_DIR.exists() else 0
    }
    
    _cache["disk_usage"] = {"time": now, "data": usage}
    return usage


@app.get("/")
async def read_index():
    return FileResponse(UI_DIR / "index.html")


# Serve UI files
if UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
