"""Tests for data cleaning utilities."""
import pytest
import tempfile
import os
from pathlib import Path

from src.data.clean import (
    normalize_unicode,
    remove_html,
    remove_urls,
    normalize_whitespace,
    clean_document,
    compute_hash,
    clean_file,
)


class TestTextCleaning:
    """Tests for individual cleaning functions."""
    
    def test_normalize_unicode(self):
        """Test Unicode normalization."""
        # NFKC normalization
        assert normalize_unicode("ﬁ") == "fi"
        assert normalize_unicode("①②③") == "123"
        
    def test_remove_html(self):
        """Test HTML tag removal."""
        assert remove_html("<p>Hello</p>") == "Hello"
        assert remove_html("<div class='test'>Content</div>") == "Content"
        assert remove_html("No tags here") == "No tags here"
        
    def test_remove_urls(self):
        """Test URL removal."""
        assert "http" not in remove_urls("Visit http://example.com for info")
        assert "https" not in remove_urls("Check https://test.org")
        
    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        assert normalize_whitespace("Hello    World") == "Hello World"
        # Multiple newlines are also collapsed to spaces
        result = normalize_whitespace("Line1\n\n\nLine2")
        assert "Line1" in result and "Line2" in result
        
    def test_clean_document_min_length(self):
        """Test minimum length filtering."""
        short = "Too short"
        long = "This is a much longer document that should pass the minimum length filter."
        
        assert clean_document(short, min_length=50) is None
        assert clean_document(long, min_length=50) is not None


class TestHashComputation:
    """Tests for document hashing."""
    
    def test_compute_hash_deterministic(self):
        """Test hash is deterministic."""
        text = "Hello World"
        h1 = compute_hash(text)
        h2 = compute_hash(text)
        assert h1 == h2
        
    def test_compute_hash_different(self):
        """Test different texts produce different hashes."""
        h1 = compute_hash("Hello World")
        h2 = compute_hash("Hello World!")
        assert h1 != h2


class TestCleanFile:
    """Tests for file-level cleaning."""
    
    def test_clean_single_file(self, tmp_path):
        """Test cleaning a single file."""
        # Create input file
        input_file = tmp_path / "input.txt"
        input_file.write_text(
            "This is a valid document with enough content to pass the minimum length filter.\n\n"
            "This is another document that should also be included in the output.\n\n"
            "Short.\n\n"  # Should be filtered out
        )
        
        output_file = tmp_path / "output.txt"
        
        doc_count = clean_file(input_file, output_file, min_length=50)
        
        assert doc_count == 2
        assert output_file.exists()
        
    def test_clean_directory_of_shards(self, tmp_path):
        """Test cleaning a directory of shard files."""
        input_dir = tmp_path / "shards"
        input_dir.mkdir()
        
        # Create shard files
        for i in range(3):
            shard = input_dir / f"shard_{i:04d}.txt"
            shard.write_text(
                f"Document {i} with enough content to pass the minimum length filter.\n\n"
            )
        
        output_file = tmp_path / "output.txt"
        
        doc_count = clean_file(input_dir, output_file, min_length=20)
        
        assert doc_count == 3
        
    def test_deduplicate(self, tmp_path):
        """Test deduplication removes identical documents."""
        input_file = tmp_path / "input.txt"
        input_file.write_text(
            "This is a duplicate document with enough content to pass the filter.\n\n"
            "This is a duplicate document with enough content to pass the filter.\n\n"
            "This is a unique document that should be included separately.\n\n"
        )
        
        output_file = tmp_path / "output.txt"
        
        doc_count = clean_file(input_file, output_file, deduplicate=True, min_length=20)
        
        assert doc_count == 2  # Only 2 unique documents
        
    def test_skip_shards_resume(self, tmp_path):
        """Test shard skipping for resume functionality."""
        input_dir = tmp_path / "shards"
        input_dir.mkdir()
        
        for i in range(5):
            shard = input_dir / f"shard_{i:04d}.txt"
            shard.write_text(f"Document from shard {i} with sufficient length.\n\n")
        
        output_file = tmp_path / "output.txt"
        
        # Skip first 2 shards
        doc_count = clean_file(input_dir, output_file, skip_shards=2, min_length=20)
        
        assert doc_count == 3  # Only processed 3 shards
