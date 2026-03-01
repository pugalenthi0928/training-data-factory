"""Tests for benchmark utility functions (no model loading required)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from benchmark import _compute_exact_match, _paired_bootstrap


class TestExactMatch:
    def test_perfect(self):
        preds = ["hello", "world"]
        refs = ["hello", "world"]
        assert _compute_exact_match(preds, refs) == 1.0

    def test_none(self):
        preds = ["hello", "world"]
        refs = ["goodbye", "earth"]
        assert _compute_exact_match(preds, refs) == 0.0

    def test_partial(self):
        preds = ["hello", "world"]
        refs = ["hello", "earth"]
        assert _compute_exact_match(preds, refs) == 0.5

    def test_whitespace_trim(self):
        preds = ["  hello  "]
        refs = ["hello"]
        assert _compute_exact_match(preds, refs) == 1.0

    def test_empty(self):
        assert _compute_exact_match([], []) == 0.0


class TestPairedBootstrap:
    def test_equal_scores(self):
        scores_a = [0.5] * 20
        scores_b = [0.5] * 20
        result = _paired_bootstrap(scores_a, scores_b)
        assert not result["significant"]

    def test_clearly_better(self):
        scores_a = [0.2] * 50
        scores_b = [0.8] * 50
        result = _paired_bootstrap(scores_a, scores_b)
        assert result["observed_delta"] > 0
        # With this dramatic difference, should be significant
        assert result["p_value"] < 0.05

    def test_empty(self):
        result = _paired_bootstrap([], [])
        assert result["p_value"] == 1.0
        assert not result["significant"]
