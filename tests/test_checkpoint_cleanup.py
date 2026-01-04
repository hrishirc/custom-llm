import shutil
import tempfile
import unittest
from pathlib import Path
from dataclasses import dataclass

# Mocking the Trainer class and Config to test specific method logic without full dependencies
@dataclass
class MockConfig:
    save_steps: int = 50

class MockTrainer:
    def __init__(self, checkpoint_dir, config):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config = config

    # We paste the logic directly or import it if the structure allows. 
    # Since we are testing 'src.training.trainer', let's import the actual class if possible, 
    # but for unit testing business logic often a subclass or mixin is easier if dependencies are heavy.
    # Here, I'll copy the method logic to ensure we are testing the logic *as written* effectively, 
    # or better, rely on the actual method if we can instantiate Trainer easily.
    # Given Trainer has heavy __init__ (model, etc), I will mock the method onto a dummy class 
    # OR better yet, let's just use a dummy class that has the method attached.
    
    def _cleanup_old_checkpoints(self, keep_last: int = 5):
        # This is the EXACT code we just implemented.
        # In a real repo we would import, but to avoid instantiation complexity of the full Trainer
        # we will use this Shim for the test, ensuring it matches 1:1 with the implementation
        # OR we can just monkeypatch the method onto this instance.
        pass

# Actually, let's try to import the Trainer class but mock its __init__ to avoid side effects
from unittest.mock import MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.training.trainer import Trainer
from src.model.config import TrainingConfig

class TestCheckpointCleanup(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.test_dir)
        self.config = TrainingConfig(save_steps=50)
        
        # Create a trainer instance with bypassed init
        self.trainer = Trainer.__new__(Trainer)
        self.trainer.checkpoint_dir = self.checkpoint_dir
        self.trainer.config = self.config
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def create_dummy_checkpoint(self, step):
        p = self.checkpoint_dir / f"phase1_step{step}.pt"
        p.touch()
        
    def test_cleanup_logic_sparse_retention(self):
        # Create checkpoints at 50, 100, 150... up to 1500
        # save_steps = 50
        # We want to verified that it keeps 1000s and last 5
        
        # checkpoints: 50, 100.. 950, 1000, 1050 ... 1500
        steps = range(50, 1501, 50) 
        for s in steps:
            self.create_dummy_checkpoint(s)
            
        # Total files: 30 files (50 to 1500 / 50)
        files = list(self.checkpoint_dir.glob("*.pt"))
        self.assertEqual(len(files), 30)
        
        # Run cleanup
        # Policy: Keep last 5 unconditionally -> 1300, 1350, 1400, 1450, 1500
        # Thinning: Keep only multiples of 1000
        # Candidates for thinning: 50...1250
        # Multiples of 1000 in candidates: 1000
        
        # Expected survivors:
        # Older milestone: 1000
        # Recents: 1300, 1350, 1400, 1450, 1500
        # Total: 6
        
        self.trainer._cleanup_old_checkpoints(keep_last=5)
        
        remaining = list(self.checkpoint_dir.glob("*.pt"))
        remaining_steps = sorted([int(f.stem.split("step")[1]) for f in remaining])
        
        expected_steps = [1000, 1300, 1350, 1400, 1450, 1500]
        
        print(f"Remaining steps: {remaining_steps}")
        print(f"Expected steps: {expected_steps}")
        
        self.assertEqual(remaining_steps, expected_steps)
        
    def test_cleanup_small_number_checkpoints(self):
        # If we have less than keep_last=5, nothing should be deleted
        steps = [50, 100, 150]
        for s in steps:
            self.create_dummy_checkpoint(s)
            
        self.trainer._cleanup_old_checkpoints(keep_last=5)
        
        remaining = list(self.checkpoint_dir.glob("*.pt"))
        self.assertEqual(len(remaining), 3)

if __name__ == "__main__":
    unittest.main()
