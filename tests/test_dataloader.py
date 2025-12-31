"""Tests for data loading and preprocessing.

Validates:
- Sequence chunking correctness
- Input/target alignment
- Collation with padding
- Context curriculum
"""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.dataloader import (
    TokenizedDataset,
    CurriculumDataLoader,
    collate_fn,
    save_tokenized_data,
)


class TestTokenizedDataset:
    """Test the basic tokenized dataset."""
    
    @pytest.fixture
    def temp_data(self):
        """Create temporary tokenized data file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tokens.npy"
            
            # Create sample tokens (1000 tokens)
            tokens = list(range(1000))
            save_tokenized_data(tokens, path)
            
            yield path
    
    def test_dataset_length(self, temp_data):
        """Dataset length should be correct for given seq_len and stride."""
        seq_len = 64
        dataset = TokenizedDataset(temp_data, seq_len=seq_len)
        
        # With 1000 tokens and seq_len=64, stride=64:
        # (1000 - 64) // 64 + 1 = 15 sequences
        expected = (1000 - seq_len) // seq_len + 1
        
        assert len(dataset) == expected
    
    def test_sequence_content(self, temp_data):
        """Sequences should contain correct token ranges."""
        seq_len = 64
        dataset = TokenizedDataset(temp_data, seq_len=seq_len)
        
        # First sequence should be tokens 0-63
        item = dataset[0]
        expected = torch.arange(64, dtype=torch.long)
        
        assert torch.equal(item["input_ids"], expected)
        assert torch.equal(item["labels"], expected)
    
    def test_sequence_stride(self, temp_data):
        """Sequences should respect stride."""
        seq_len = 64
        stride = 32
        dataset = TokenizedDataset(temp_data, seq_len=seq_len, stride=stride)
        
        # Second sequence should start at position 32
        item = dataset[1]
        expected = torch.arange(32, 96, dtype=torch.long)
        
        assert torch.equal(item["input_ids"], expected)
    
    def test_labels_match_inputs(self, temp_data):
        """Labels should equal input_ids (shifted in model)."""
        dataset = TokenizedDataset(temp_data, seq_len=64)
        
        for i in range(min(5, len(dataset))):
            item = dataset[i]
            assert torch.equal(item["input_ids"], item["labels"])


class TestCollateFn:
    """Test batch collation."""
    
    def test_padding_applied(self):
        """Shorter sequences should be padded."""
        batch = [
            {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([1, 2, 3])},
            {"input_ids": torch.tensor([4, 5, 6, 7, 8]), "labels": torch.tensor([4, 5, 6, 7, 8])},
        ]
        
        collated = collate_fn(batch, pad_token_id=0)
        
        # Both should be length 5
        assert collated["input_ids"].shape == (2, 5)
        assert collated["labels"].shape == (2, 5)
        
        # First sequence should be padded
        assert collated["input_ids"][0, 3] == 0
        assert collated["input_ids"][0, 4] == 0
    
    def test_attention_mask_correct(self):
        """Attention mask should be 1 for real tokens, 0 for padding."""
        batch = [
            {"input_ids": torch.tensor([1, 2]), "labels": torch.tensor([1, 2])},
            {"input_ids": torch.tensor([3, 4, 5, 6]), "labels": torch.tensor([3, 4, 5, 6])},
        ]
        
        collated = collate_fn(batch, pad_token_id=0)
        
        # First sequence: [1, 1, 0, 0]
        assert collated["attention_mask"][0, 0] == 1
        assert collated["attention_mask"][0, 1] == 1
        assert collated["attention_mask"][0, 2] == 0
        assert collated["attention_mask"][0, 3] == 0
        
        # Second sequence: [1, 1, 1, 1]
        assert torch.all(collated["attention_mask"][1] == 1)
    
    def test_no_padding_same_length(self):
        """Same-length sequences should not be padded."""
        batch = [
            {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([1, 2, 3])},
            {"input_ids": torch.tensor([4, 5, 6]), "labels": torch.tensor([4, 5, 6])},
        ]
        
        collated = collate_fn(batch, pad_token_id=0)
        
        assert collated["input_ids"].shape == (2, 3)
        assert torch.all(collated["attention_mask"] == 1)


class TestCurriculumDataLoader:
    """Test context length curriculum."""
    
    @pytest.fixture
    def temp_data(self):
        """Create temporary tokenized data file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tokens.npy"
            tokens = list(range(10000))  # Larger dataset
            save_tokenized_data(tokens, path)
            yield path
    
    def test_seq_len_schedule(self, temp_data):
        """Sequence length should follow schedule based on progress."""
        schedule = {0.0: 64, 0.3: 128, 0.7: 256}
        loader = CurriculumDataLoader(
            temp_data, 
            batch_size=2, 
            context_schedule=schedule
        )
        
        assert loader.get_seq_len_for_progress(0.0) == 64
        assert loader.get_seq_len_for_progress(0.2) == 64
        assert loader.get_seq_len_for_progress(0.3) == 128
        assert loader.get_seq_len_for_progress(0.5) == 128
        assert loader.get_seq_len_for_progress(0.7) == 256
        assert loader.get_seq_len_for_progress(1.0) == 256
    
    def test_loader_updates_with_progress(self, temp_data):
        """Loader should update when progress changes schedule."""
        schedule = {0.0: 64, 0.5: 128}
        loader = CurriculumDataLoader(
            temp_data,
            batch_size=2,
            context_schedule=schedule,
        )
        
        # Get loader at progress 0
        dl1 = loader.get_loader(progress=0.0)
        assert loader._current_seq_len == 64
        
        # Get loader at progress 0.5
        dl2 = loader.get_loader(progress=0.5)
        assert loader._current_seq_len == 128


class TestSaveTokenizedData:
    """Test tokenized data saving."""
    
    def test_save_creates_file(self):
        """save_tokenized_data should create a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tokens.npy"
            tokens = [1, 2, 3, 4, 5]
            
            save_tokenized_data(tokens, path)
            
            assert path.exists()
    
    def test_save_correct_content(self):
        """Saved file should contain correct tokens."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tokens.npy"
            tokens = [10, 20, 30, 40, 50]
            
            save_tokenized_data(tokens, path)
            
            # Read back
            loaded = np.memmap(path, dtype=np.int32, mode="r")
            
            assert list(loaded) == tokens


class TestInputTargetAlignment:
    """Test that inputs and targets are correctly aligned for causal LM."""
    
    def test_shift_alignment(self):
        """Labels should allow for shifted loss computation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tokens.npy"
            
            # Create known sequence
            tokens = list(range(100))
            save_tokenized_data(tokens, path)
            
            dataset = TokenizedDataset(path, seq_len=10)
            item = dataset[0]
            
            input_ids = item["input_ids"]
            labels = item["labels"]
            
            # For causal LM, loss is computed as:
            # predict token[i+1] given token[0:i+1]
            # So input[:-1] predicts labels[1:]
            
            # Verify this alignment makes sense
            assert torch.equal(input_ids[:-1], labels[:-1])  # All but last
            # The model will shift internally


class TestDataLoaderIntegration:
    """Integration tests for data loading with model."""
    
    def test_batch_compatible_with_model(self):
        """Batch from dataloader should be compatible with model."""
        from model.config import ModelConfig
        from model.llm import LLM
        
        config = ModelConfig(
            vocab_size=1000,
            max_seq_len=64,
            n_layers=2,
            hidden_size=32,
            n_heads=2,
            head_dim=16,
            intermediate_size=64,
        )
        model = LLM(config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tokens.npy"
            # Create tokens within vocab range
            tokens = list(np.random.randint(0, 1000, size=1000))
            save_tokenized_data(tokens, path)
            
            dataset = TokenizedDataset(path, seq_len=32)
            
            # Get a batch
            batch = [dataset[i] for i in range(2)]
            collated = collate_fn(batch)
            
            # Forward through model
            outputs = model(
                input_ids=collated["input_ids"],
                attention_mask=collated["attention_mask"],
                labels=collated["labels"],
            )
            
            assert "logits" in outputs
            assert "loss" in outputs
            assert outputs["loss"].item() > 0
