import sys
import torch
import unittest
from unittest.mock import MagicMock, call
from pathlib import Path

# Add src to path
sys.path.append("/Users/hrishikesh/Repos/custom-llm")

from src.training.trainer import Trainer, TrainerState
from src.model.config import ModelConfig, TrainingConfig

class TestTrainerFlow(unittest.TestCase):
    def setUp(self):
        self.model = MagicMock()
        param = torch.nn.Parameter(torch.zeros(10, requires_grad=True))
        self.model.parameters.return_value = [param]
        self.model.named_parameters.return_value = [("layer.weight", param)]
        
        self.config = TrainingConfig(
            gradient_accumulation_steps=2,
            logging_steps=1,
            micro_batch_size=2,
            compile_model=False,
            device="cpu"
        )
        self.data_loader = MagicMock()
        self.data_loader.get_loader.return_value = [
            {"input_ids": torch.zeros(1, 1), "labels": torch.zeros(1, 1)}
        ] * 4 # 4 batches
        self.data_loader.get_context_length.return_value = 128
        
        self.trainer = Trainer(
            model=self.model,
            train_config=self.config,
            data_loader=self.data_loader,
            checkpoint_dir="/tmp/ckpt",
            log_dir="/tmp/logs"
        )
        
        
        # Use real optimizer for scheduler compatibility
        self.trainer.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        self.trainer.optimizer.zero_grad = MagicMock(wraps=self.trainer.optimizer.zero_grad)
        self.trainer.optimizer.step = MagicMock(wraps=self.trainer.optimizer.step)
        self.trainer.scheduler = MagicMock()
        self.trainer._create_scheduler = MagicMock() # Prevent overwriting
        self.trainer.scaler = None
        self.trainer.train_step = MagicMock(return_value=1.0)
        self.trainer.metrics_logger = MagicMock()
        self.trainer.save_checkpoint = MagicMock()
        
    def test_training_step_order(self):
        # Run 1 global step (2 micro steps)
        self.trainer.train(total_steps=1)
        
        # Verify order of operations
        manager = MagicMock()
        manager.attach_mock(self.trainer.optimizer.zero_grad, 'zero_grad')
        manager.attach_mock(self.trainer.train_step, 'train_step')
        manager.attach_mock(self.trainer.optimizer.step, 'step')
        manager.attach_mock(self.trainer.metrics_logger.log, 'log')
        manager.attach_mock(self.trainer.scheduler.step, 'scheduler_step')
        
        # Reset trainer state for second run
        self.trainer.state.global_step = 0
        
        # Re-run to capture calls in manager
        self.trainer.train(total_steps=1)
        
        # Expected sequence for 1 global step with accum=2:
        # 1. zero_grad (start of step)
        # 2. train_step (micro 1)
        # 3. train_step (micro 2)
        # 4. step (optimizer update)
        # 5. log (metrics)
        # 6. scheduler_step (end of step)
        
        expected_calls = [
            call.zero_grad(),
            call.train_step(unittest.mock.ANY),
            call.train_step(unittest.mock.ANY),
            call.step(),
            call.log(unittest.mock.ANY),
            call.scheduler_step()
        ]
        
        # Filter calls to only valid ones
        actual_calls = [c for c in manager.mock_calls if c in expected_calls or 
                       (c[0] == 'train_step') or (c[0] == 'log')]
        
        # We need to match the sequence type
        # Simplification: just check relative indices of calls in the list
        call_names = [c[0] for c in manager.mock_calls]
        
        print("\nActual Call Sequence:")
        for context in call_names:
            print(f" - {context}")
            
        try:
            zero_idx = call_names.index('zero_grad')
            step_idx = call_names.index('step')
            log_idx = call_names.index('log')
            sched_idx = call_names.index('scheduler_step')
            
            # Assertions
            self.assertLess(zero_idx, step_idx, "zero_grad must come before step")
            self.assertLess(step_idx, log_idx, "step must come before log (to log updated weights? No, to log gradients that triggered update)")
            # Actually, standard practice: zero -> fwd/bwd -> step -> log -> zero
            # But here we moved zero to start.
            # So: zero -> fwd/bwd -> step -> log -> sched -> [next loop: zero]
            # This means valid gradients exist at 'log'.
            
            self.assertLess(log_idx, sched_idx, "log must come before scheduler step (to log the LR used for the step)")
            
            # Verify accumulation
            train_step_indices = [i for i, n in enumerate(call_names) if n == 'train_step']
            self.assertEqual(len(train_step_indices), 2, "Should have 2 accumulation steps")
            self.assertLess(zero_idx, train_step_indices[0], "zero_grad before first accumulation")
            self.assertLess(train_step_indices[-1], step_idx, "last accumulation before step")
            
            print("Verification Successful: Order of operations is correct.")
            
        except ValueError as e:
            self.fail(f"Missing call in sequence: {e}")

if __name__ == '__main__':
    unittest.main()
