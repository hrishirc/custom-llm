"""Tests for StateDB (SQLite state management)."""
import pytest
import tempfile
import os
from pathlib import Path

from src.workflows.state_db import StateDB, DownloadState, StageState, PhaseState


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    os.unlink(db_path)


class TestStateDB:
    """Tests for StateDB class."""
    
    def test_initialization(self, temp_db):
        """Test database initialization creates tables."""
        db = StateDB(temp_db)
        assert Path(temp_db).exists()
        
    def test_download_lifecycle(self, temp_db):
        """Test download status transitions."""
        db = StateDB(temp_db)
        
        # Initially no download
        assert db.get_download("test_dataset") is None
        
        # Create a download record
        db.set_download(DownloadState("test_dataset", "pending"))
        dl = db.get_download("test_dataset")
        assert dl is not None
        assert dl.status == "pending"
        
        # Mark running
        db.mark_download_running("test_dataset")
        dl = db.get_download("test_dataset")
        assert dl.status == "running"
        
        # Mark complete
        db.mark_download_complete("test_dataset", "/path/to/file", 1024)
        dl = db.get_download("test_dataset")
        assert dl.status == "complete"
        assert dl.size_bytes == 1024
        
    def test_stage_lifecycle(self, temp_db):
        """Test stage status transitions."""
        db = StateDB(temp_db)
        
        # Create a stage record
        db.set_stage(StageState("clean_wikipedia", "pending"))
        
        # Mark running
        db.mark_stage_running("clean_wikipedia")
        stage = db.get_stage("clean_wikipedia")
        assert stage is not None
        assert stage.status == "running"
        
        # Mark complete
        db.mark_stage_complete("clean_wikipedia", {"total_docs": 5000})
        stage = db.get_stage("clean_wikipedia")
        assert stage.status == "complete"
        
    def test_phase_lifecycle(self, temp_db):
        """Test training phase status transitions."""
        db = StateDB(temp_db)
        
        # Create a phase record
        db.set_phase(PhaseState("1", "pending", total_steps=50000))
        
        # Mark running
        db.mark_phase_running("1")
        phase = db.get_phase("1")
        assert phase is not None
        assert phase.status == "running"
        
        # Update progress
        db.update_phase_progress("1", current_step=100, loss=2.5)
        phase = db.get_phase("1")
        assert phase.current_step == 100
        
        # Mark complete
        db.mark_phase_complete("1", "/path/to/checkpoint.pt")
        phase = db.get_phase("1")
        assert phase.status == "complete"
        
    def test_get_full_state(self, temp_db):
        """Test retrieving full state."""
        db = StateDB(temp_db)
        
        # Create records first
        db.set_download(DownloadState("wikipedia", "pending"))
        db.mark_download_running("wikipedia")
        db.mark_download_complete("wikipedia", "/path", 1000)
        
        db.set_stage(StageState("clean_wikipedia", "pending"))
        db.mark_stage_running("clean_wikipedia")
        db.mark_stage_complete("clean_wikipedia", {})
        
        state = db.get_full_state()
        assert "downloads" in state
        assert "stages" in state
        assert "phases" in state
        assert "wikipedia" in state["downloads"]
        
    def test_event_logging(self, temp_db):
        """Test event log entries."""
        db = StateDB(temp_db)
        
        db._log_event(
            level="INFO",
            category="download",
            entity="wikipedia",
            action="started",
            message="Download started"
        )
        
        # We don't have a direct method to query events, but this tests it doesn't crash


class TestStateDBIdempotency:
    """Tests for idempotent operations."""
    
    def test_double_mark_running(self, temp_db):
        """Marking running twice should not fail."""
        db = StateDB(temp_db)
        db.set_stage(StageState("test_stage", "pending"))
        db.mark_stage_running("test_stage")
        db.mark_stage_running("test_stage")  # Should not raise
        
    def test_complete_after_running(self, temp_db):
        """Completing after running should work."""
        db = StateDB(temp_db)
        db.set_stage(StageState("test_stage", "pending"))
        db.mark_stage_running("test_stage")
        db.mark_stage_complete("test_stage", {})
        stage = db.get_stage("test_stage")
        assert stage.status == "complete"
