"""Regression tests for checkpoint resume functionality.

These tests verify that training can be properly resumed from checkpoints
without losing scheduler state or overwriting metrics.

Historical context:
- Bug discovered 2026-01-03: Scheduler reset on resume caused LR to restart
  warmup instead of continuing from saved step.
- Bug discovered 2026-01-03: MetricsLogger CSV was overwritten on restart.
"""
import pytest
import torch
import tempfile
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.config import ModelConfig, TrainingConfig
from src.model.llm import LLM, create_model
from src.training.trainer import Trainer, TrainerState
from src.training.metrics_logger import MetricsLogger
from src.data.dataloader import CurriculumDataLoader


class TestSchedulerResume:
    """Tests for scheduler state restoration after checkpoint load."""
    
    @pytest.fixture
    def mini_config(self):
        """Minimal config for fast testing."""
        return ModelConfig(
            vocab_size=1000,
            hidden_size=64,
            n_layers=2,
            n_heads=2,
            head_dim=32,
            intermediate_size=128,
            max_seq_len=64,
        )
    
    @pytest.fixture
    def mini_train_config(self):
        """Minimal training config."""
        return TrainingConfig(
            learning_rate=3e-4,
            min_learning_rate=1e-4,
            warmup_ratio=0.1,
            micro_batch_size=2,
            gradient_accumulation_steps=1,
            logging_steps=5,
            save_steps=10,
            compile_model=False,
        )
    
    @pytest.fixture
    def tokenized_data(self, tmp_path):
        """Create minimal tokenized data."""
        data_path = tmp_path / "tokenized" / "phase1.npy"
        data_path.parent.mkdir(parents=True)
        # Create random token data
        tokens = np.random.randint(0, 1000, size=10000, dtype=np.int32)
        tokens.tofile(data_path)
        return data_path
    
    def test_scheduler_state_buffered_when_scheduler_is_none(self, mini_config, mini_train_config, tokenized_data, tmp_path):
        """Scheduler state should be buffered if scheduler doesn't exist yet."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        
        # Create model and trainer
        model = create_model(mini_config)
        data_loader = CurriculumDataLoader(
            data_path=tokenized_data,
            batch_size=mini_train_config.micro_batch_size,
            context_schedule={0.0: 32},
        )
        
        trainer = Trainer(
            model=model,
            train_config=mini_train_config,
            data_loader=data_loader,
            checkpoint_dir=checkpoint_dir,
        )
        
        # Verify scheduler is None before train()
        assert trainer.scheduler is None
        
        # Create a fake checkpoint with scheduler state
        fake_checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": {"last_epoch": 100, "_step_count": 101},
            "state": {
                "global_step": 100,
                "epoch": 0,
                "total_tokens": 0,
                "best_loss": float("inf"),
            },
        }
        ckpt_path = checkpoint_dir / "test_step100.pt"
        torch.save(fake_checkpoint, ckpt_path)
        
        # Load checkpoint (scheduler is still None)
        trainer.load_checkpoint(ckpt_path)
        
        # Verify state was buffered
        assert trainer._pending_scheduler_state is not None
        assert trainer._pending_scheduler_state["last_epoch"] == 100
        assert trainer.state.global_step == 100
    
    def test_scheduler_state_applied_after_train_starts(self, mini_config, mini_train_config, tokenized_data, tmp_path):
        """Buffered scheduler state should be applied when train() creates the scheduler."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        
        # Create first trainer and run a few steps
        model1 = create_model(mini_config)
        data_loader1 = CurriculumDataLoader(
            data_path=tokenized_data,
            batch_size=mini_train_config.micro_batch_size,
            context_schedule={0.0: 32},
        )
        
        trainer1 = Trainer(
            model=model1,
            train_config=mini_train_config,
            data_loader=data_loader1,
            checkpoint_dir=checkpoint_dir,
        )
        
        # Run 15 steps (to get past warmup with total_steps=100)
        trainer1.train(total_steps=15, phase_name="test")
        
        # Save checkpoint
        trainer1.save_checkpoint("test_step15")
        
        # Verify checkpoint was saved with scheduler state
        ckpt = torch.load(checkpoint_dir / "test_step15.pt", map_location="cpu", weights_only=False)
        assert ckpt["scheduler_state_dict"]["last_epoch"] == 15
        
        # Create NEW trainer (simulating restart)
        model2 = create_model(mini_config)
        data_loader2 = CurriculumDataLoader(
            data_path=tokenized_data,
            batch_size=mini_train_config.micro_batch_size,
            context_schedule={0.0: 32},
        )
        
        trainer2 = Trainer(
            model=model2,
            train_config=mini_train_config,
            data_loader=data_loader2,
            checkpoint_dir=checkpoint_dir,
        )
        
        # Load checkpoint BEFORE train()
        trainer2.load_checkpoint(checkpoint_dir / "test_step15.pt")
        
        # Verify pending state is buffered
        assert trainer2._pending_scheduler_state is not None
        assert trainer2.scheduler is None
        
        # Start training (this should apply the buffered state)
        trainer2.train(total_steps=20, phase_name="test")
        
        # Verify scheduler was restored correctly
        # After running 5 more steps (15->20), last_epoch should be 20
        assert trainer2.scheduler.last_epoch == 20
        assert trainer2._pending_scheduler_state is None  # Should be cleared


class TestCSVAppend:
    """Tests for MetricsLogger CSV append behavior."""
    
    def test_csv_appends_on_restart(self, tmp_path):
        """MetricsLogger should append to existing CSV, not overwrite."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        
        # Create first logger and write some entries
        logger1 = MetricsLogger(
            log_dir=log_dir,
            tensorboard=False,
            csv_logging=True,
        )
        
        # Write 5 mock metrics
        for step in range(1, 6):
            from src.training.metrics_logger import TrainingMetrics
            metrics = TrainingMetrics(
                step=step,
                loss=1.0 / step,
                learning_rate=0.001,
            )
            logger1.log(metrics)
        
        logger1.close()
        
        # Count lines (1 header + 5 data)
        csv_path = log_dir / "metrics.csv"
        lines_after_first = len(csv_path.read_text().strip().split("\n"))
        assert lines_after_first == 6  # 1 header + 5 data
        
        # Create SECOND logger (simulating restart)
        logger2 = MetricsLogger(
            log_dir=log_dir,
            tensorboard=False,
            csv_logging=True,
        )
        
        # Write 5 more metrics
        for step in range(6, 11):
            metrics = TrainingMetrics(
                step=step,
                loss=1.0 / step,
                learning_rate=0.001,
            )
            logger2.log(metrics)
        
        logger2.close()
        
        # Count lines (should be 1 header + 10 data = 11)
        lines_after_second = len(csv_path.read_text().strip().split("\n"))
        assert lines_after_second == 11  # 1 header + 10 data (appended, not overwritten)
    
    def test_csv_writes_header_only_once(self, tmp_path):
        """CSV header should only be written once, even after restart."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        
        # Create and close two loggers
        for _ in range(2):
            logger = MetricsLogger(
                log_dir=log_dir,
                tensorboard=False,
                csv_logging=True,
            )
            logger.close()
        
        # Read CSV and count header occurrences
        csv_path = log_dir / "metrics.csv"
        content = csv_path.read_text()
        header_count = content.count("step,timestamp,loss")
        
        assert header_count == 1, f"Header appeared {header_count} times, expected 1"


class TestProgressBarUpdate:
    """Tests for progress bar behavior."""
    
    def test_no_duplicate_pbar_update(self, tmp_path):
        """Training loop should only update progress bar once per step."""
        # This is difficult to test directly without mocking tqdm
        # We verify by checking the train() code structure
        
        # Read trainer.py and count pbar.update calls in train loop
        trainer_path = Path(__file__).parent.parent / "src" / "training" / "trainer.py"
        content = trainer_path.read_text()
        
        # Find the train method and count pbar.update(1) calls
        # After our fix, there should be exactly ONE in the inner while loop
        train_method_start = content.find("def train(")
        train_method_end = content.find("\n    def ", train_method_start + 1)
        train_method = content[train_method_start:train_method_end]
        
        # Count pbar.update(1) calls (our fix removed the duplicate)
        update_calls = train_method.count("pbar.update(1)")
        
        assert update_calls == 1, f"Found {update_calls} pbar.update(1) calls, expected exactly 1"
