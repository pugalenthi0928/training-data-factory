"""Tests for benchmark utility functions (no model loading required)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from benchmark import (
    _compute_exact_match,
    _paired_bootstrap_ci,
    _paired_randomization_test,
)  # noqa: E402


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


class TestPairedRandomization:
    def test_equal_scores(self):
        scores_a = [0.5] * 20
        scores_b = [0.5] * 20
        result = _paired_randomization_test(scores_a, scores_b)
        assert not result["significant"]
        assert result["method"] == "paired_randomization_one_sided"

    def test_clearly_better(self):
        scores_a = [0.2] * 50
        scores_b = [0.8] * 50
        result = _paired_randomization_test(scores_a, scores_b)
        assert result["observed_delta"] > 0
        assert result["p_value"] < 0.05
        assert result["p_value"] >= 1 / 1001

    def test_empty(self):
        result = _paired_randomization_test([], [])
        assert result["p_value"] == 1.0
        assert not result["significant"]

    def test_mismatched_pairs_are_rejected(self):
        with pytest.raises(ValueError, match="equal length"):
            _paired_randomization_test([0.1], [0.1, 0.2])


class TestPairedBootstrapInterval:
    def test_constant_improvement_has_exact_interval(self):
        result = _paired_bootstrap_ci([0.1] * 20, [0.4] * 20)
        assert result["method"] == "paired_percentile_bootstrap"
        assert result["lower"] == pytest.approx(0.3)
        assert result["upper"] == pytest.approx(0.3)

    def test_empty_interval_is_explicit(self):
        result = _paired_bootstrap_ci([], [])
        assert result["lower"] is None
        assert result["upper"] is None
