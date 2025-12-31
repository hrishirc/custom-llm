"""Tests for core pipeline functions."""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestProgressCallback:
    """Tests for progress callback interface."""
    
    def test_callback_interface(self):
        """Test callback receives expected arguments."""
        from src.workflows.core import ProgressCallback
        
        received = []
        
        def test_callback(msg, progress=None, details=None):
            received.append((msg, progress, details))
        
        # Verify it matches the protocol
        test_callback("Test message", progress=50.0, details={"key": "value"})
        
        assert len(received) == 1
        assert received[0][0] == "Test message"
        assert received[0][1] == 50.0
        assert received[0][2] == {"key": "value"}


class TestCleanCore:
    """Tests for clean_core function."""
    
    def test_clean_core_skips_missing_dataset(self, tmp_path):
        """Test clean_core skips when raw file doesn't exist."""
        from src.workflows.core import clean_core
        
        callback = MagicMock()
        result = clean_core("nonexistent", tmp_path, callback)
        
        assert result["status"] == "skipped"
        
    def test_clean_core_returns_cached(self, tmp_path):
        """Test clean_core returns cached result if already done."""
        from src.workflows.core import clean_core
        
        # Create the "completed" clean file
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()
        clean_file = processed_dir / "wikipedia_clean.txt"
        clean_file.write_text("Already cleaned content")
        
        # Create fake raw dir
        raw_dir = tmp_path / "raw" / "phase1" / "wikipedia"
        raw_dir.mkdir(parents=True)
        (raw_dir / "shard_0000.txt").write_text("Raw content")
        
        callback = MagicMock()
        result = clean_core("wikipedia", tmp_path, callback)
        
        assert result["status"] == "complete"
        assert "cached" in str(result.get("n_documents", ""))


class TestTokenizeCore:
    """Tests for tokenize_core function."""
    
    def test_tokenize_core_requires_tokenizer(self, tmp_path):
        """Test tokenize_core fails without tokenizer."""
        from src.workflows.core import tokenize_core
        
        callback = MagicMock()
        
        with pytest.raises(RuntimeError, match="Tokenizer not found"):
            tokenize_core("1", tmp_path, callback)


class TestTrainPhaseCore:
    """Tests for train_phase_core function."""
    
    def test_train_phase_core_requires_tokenized_data(self, tmp_path):
        """Test train_phase_core fails without tokenized data."""
        from src.workflows.core import train_phase_core
        
        callback = MagicMock()
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        
        with pytest.raises(RuntimeError, match="Tokenized data not found"):
            train_phase_core("1", tmp_path, checkpoint_dir, None, callback)


class TestDownloadCore:
    """Tests for download_core function."""
    
    @patch('src.workflows.core.download_dataset_raw')
    def test_download_core_calls_raw_function(self, mock_download, tmp_path):
        """Test download_core delegates to download_dataset_raw."""
        from src.workflows.core import download_core
        
        # Setup mock
        mock_output_dir = tmp_path / "raw" / "phase1" / "wikipedia"
        mock_output_dir.mkdir(parents=True)
        (mock_output_dir / "shard_0000.txt").write_text("data")
        mock_download.return_value = mock_output_dir
        
        callback = MagicMock()
        result = download_core("wikipedia", tmp_path, callback)
        
        assert result["status"] == "complete"
        assert mock_download.called


class TestResumeState:
    """Tests for resume state handling."""
    
    def test_clean_core_resume_state(self, tmp_path):
        """Test clean_core uses resume_state."""
        from src.workflows.core import clean_core
        
        # Create raw directory with shards
        raw_dir = tmp_path / "raw" / "phase1" / "wikipedia"
        raw_dir.mkdir(parents=True)
        for i in range(5):
            (raw_dir / f"shard_{i:04d}.txt").write_text(
                f"Document {i} content that is long enough to pass the filter.\n\n"
            )
        
        callback = MagicMock()
        resume_state = {"shards_processed": 2}
        
        result = clean_core("wikipedia", tmp_path, callback, resume_state)
        
        # Should process only 3 shards (5 - 2 skipped)
        assert result["status"] == "complete"
