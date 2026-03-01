"""Tests for contamination detection module."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from training_data_robo.contamination import (
    BenchmarkIndex,
    ContaminationChecker,
    _build_ngrams,
    _tokenize,
    build_index_from_texts,
)


class TestTokenize:
    def test_basic(self):
        tokens = _tokenize("Hello, World! This is a test.")
        assert tokens == ["hello", "world", "this", "is", "a", "test"]

    def test_empty(self):
        assert _tokenize("") == []


class TestNgrams:
    def test_8grams(self):
        tokens = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
        ngrams = _build_ngrams(tokens, 8)
        assert len(ngrams) == 2  # [a..h] and [b..i]

    def test_too_short(self):
        tokens = ["a", "b", "c"]
        ngrams = _build_ngrams(tokens, 8)
        assert len(ngrams) == 0


class TestBenchmarkIndex:
    def test_add_and_check(self):
        idx = BenchmarkIndex()
        # Add a sentence with enough tokens for 8-grams
        idx.add_text("The quick brown fox jumps over the lazy dog and runs away fast today")
        assert idx.size == 1

        # Check exact match
        result = idx.check_overlap("The quick brown fox jumps over the lazy dog and runs away fast today")
        assert result["8gram_overlaps"] > 0
        assert result["is_contaminated"]

    def test_no_overlap(self):
        idx = BenchmarkIndex()
        idx.add_text("Alpha bravo charlie delta echo foxtrot golf hotel india juliet")
        result = idx.check_overlap("Completely different text with no common phrases at all here today now")
        assert result["8gram_overlaps"] == 0
        assert not result["is_contaminated"]

    def test_partial_overlap(self):
        idx = BenchmarkIndex()
        idx.add_text("Machine learning is a subset of artificial intelligence that focuses on data")
        result = idx.check_overlap("Machine learning is a subset of artificial intelligence that uses data models")
        assert result["8gram_overlaps"] > 0


class TestBuildIndexFromTexts:
    def test_basic(self):
        texts = [
            "The first benchmark question about machine learning algorithms",
            "The second benchmark about natural language processing tasks",
        ]
        idx = build_index_from_texts(texts)
        assert idx.size == 2


class TestContaminationChecker:
    def test_check_dataset_clean(self):
        checker = ContaminationChecker()
        checker.load_custom_texts([
            "Alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo",
        ], name="test_bench")

        examples = [
            {"id": "1", "input_text": "What is X?", "output_text": "Something completely different and unique here today"},
        ]
        report = checker.check_dataset(examples)
        assert report["total_examples"] == 1
        assert report["contaminated_count"] == 0
        assert report["contamination_rate"] == 0.0

    def test_check_dataset_contaminated(self):
        # Create a benchmark text
        bench_text = "The quick brown fox jumps over the lazy dog and then runs away very fast"
        checker = ContaminationChecker()
        checker.load_custom_texts([bench_text], name="test_bench")

        # Use the exact same text as training data
        examples = [
            {"id": "1", "input_text": "", "output_text": bench_text},
        ]
        report = checker.check_dataset(examples)
        assert report["contaminated_count"] == 1
        assert report["contamination_rate"] == 1.0

    def test_load_benchmark_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"question": "What is the capital of France and what is it known for historically"}) + "\n")
            f.write(json.dumps({"question": "How does photosynthesis work in plants and what are the main stages"}) + "\n")
            tmp_path = Path(f.name)

        try:
            checker = ContaminationChecker()
            checker.load_benchmark_file(tmp_path, name="test")
            assert checker.index.size == 2
        finally:
            tmp_path.unlink()

    def test_empty_benchmark(self):
        checker = ContaminationChecker()
        examples = [{"id": "1", "output_text": "Some text"}]
        report = checker.check_dataset(examples)
        assert report["contaminated_count"] == 0
