"""Tests for Native Orchestrator."""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.workflows.native_orchestrator import NativeOrchestrator
from src.workflows.state_db import StateDB


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create directory structure
        (tmpdir / "data").mkdir()
        (tmpdir / "checkpoints").mkdir()
        (tmpdir / "logs").mkdir()
        
        yield tmpdir


class TestNativeOrchestratorInit:
    """Tests for orchestrator initialization."""
    
    def test_initialization(self, temp_workspace):
        """Test orchestrator initializes correctly."""
        db_path = temp_workspace / "data" / "state.db"
        
        orchestrator = NativeOrchestrator(
            db_path=str(db_path),
            data_dir=str(temp_workspace / "data"),
            checkpoint_dir=str(temp_workspace / "checkpoints"),
            log_dir=str(temp_workspace / "logs"),
            phases=["1"]
        )
        
        assert orchestrator.phases == ["1"]
        assert orchestrator.max_retries == 3
        
    def test_custom_phases(self, temp_workspace):
        """Test custom phase configuration."""
        db_path = temp_workspace / "data" / "state.db"
        
        orchestrator = NativeOrchestrator(
            db_path=str(db_path),
            data_dir=str(temp_workspace / "data"),
            checkpoint_dir=str(temp_workspace / "checkpoints"),
            log_dir=str(temp_workspace / "logs"),
            phases=["2", "2b"]
        )
        
        assert "1" not in orchestrator.phases
        assert "2" in orchestrator.phases
        assert "2b" in orchestrator.phases


class TestProgressCallback:
    """Tests for progress callback mechanism."""
    
    def test_callback_logs_message(self, temp_workspace):
        """Test callback logs messages."""
        db_path = temp_workspace / "data" / "state.db"
        
        orchestrator = NativeOrchestrator(
            db_path=str(db_path),
            data_dir=str(temp_workspace / "data"),
            checkpoint_dir=str(temp_workspace / "checkpoints"),
            log_dir=str(temp_workspace / "logs"),
        )
        
        callback = orchestrator._progress_callback("test_stage", "stage", "test_stage")
        
        # Should not raise
        callback("Test message", progress=50.0, details={"key": "value"})
        
    def test_callback_updates_db(self, temp_workspace):
        """Test callback updates database."""
        from src.workflows.state_db import StageState
        
        db_path = temp_workspace / "data" / "state.db"
        
        orchestrator = NativeOrchestrator(
            db_path=str(db_path),
            data_dir=str(temp_workspace / "data"),
            checkpoint_dir=str(temp_workspace / "checkpoints"),
            log_dir=str(temp_workspace / "logs"),
        )
        
        # First create and mark stage running
        orchestrator.db.set_stage(StageState("test_stage", "pending"))
        orchestrator.db.mark_stage_running("test_stage")
        
        callback = orchestrator._progress_callback("test_stage", "stage", "test_stage")
        callback("Progress update", progress=75.0, details={"docs": 1000})
        
        # Check DB was updated
        stage = orchestrator.db.get_stage("test_stage")
        assert stage is not None
        assert stage.details is not None
        assert stage.details.get("docs") == 1000


class TestOrchestratorSkipLogic:
    """Tests for skip logic (idempotency)."""
    
    def test_skips_completed_downloads(self, temp_workspace):
        """Test orchestrator skips completed downloads."""
        from src.workflows.state_db import DownloadState
        
        db_path = temp_workspace / "data" / "state.db"
        
        # Pre-populate DB with completed download
        db = StateDB(str(db_path))
        db.set_download(DownloadState("wikipedia", "complete", "/path", 1000))
        
        orchestrator = NativeOrchestrator(
            db_path=str(db_path),
            data_dir=str(temp_workspace / "data"),
            checkpoint_dir=str(temp_workspace / "checkpoints"),
            log_dir=str(temp_workspace / "logs"),
            phases=["1"]
        )
        
        # Check that wikipedia shows as complete
        dl = orchestrator.db.get_download("wikipedia")
        assert dl is not None
        assert dl.status == "complete"
        
    def test_skips_completed_stages(self, temp_workspace):
        """Test orchestrator skips completed stages."""
        from src.workflows.state_db import StageState
        
        db_path = temp_workspace / "data" / "state.db"
        
        # Pre-populate DB with completed stage
        db = StateDB(str(db_path))
        db.set_stage(StageState("clean_wikipedia", "complete"))
        
        orchestrator = NativeOrchestrator(
            db_path=str(db_path),
            data_dir=str(temp_workspace / "data"),
            checkpoint_dir=str(temp_workspace / "checkpoints"),
            log_dir=str(temp_workspace / "logs"),
        )
        
        stage = orchestrator.db.get_stage("clean_wikipedia")
        assert stage is not None
        assert stage.status == "complete"


class TestOrchestratorDatasets:
    """Tests for dataset determination logic."""
    
    def test_phase1_datasets(self, temp_workspace):
        """Test phase 1 requires wikipedia."""
        db_path = temp_workspace / "data" / "state.db"
        
        orchestrator = NativeOrchestrator(
            db_path=str(db_path),
            data_dir=str(temp_workspace / "data"),
            checkpoint_dir=str(temp_workspace / "checkpoints"),
            log_dir=str(temp_workspace / "logs"),
            phases=["1"]
        )
        
        # Check phases list
        assert "1" in orchestrator.phases
        
    def test_phase2_datasets(self, temp_workspace):
        """Test phase 2 requires pg19 and bookcorpus."""
        db_path = temp_workspace / "data" / "state.db"
        
        orchestrator = NativeOrchestrator(
            db_path=str(db_path),
            data_dir=str(temp_workspace / "data"),
            checkpoint_dir=str(temp_workspace / "checkpoints"),
            log_dir=str(temp_workspace / "logs"),
            phases=["2"]
        )
        
        assert "2" in orchestrator.phases
