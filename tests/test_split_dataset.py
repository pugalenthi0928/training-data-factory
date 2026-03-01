"""Tests for train/test split utility."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from split_dataset import stratified_split


def _make_rows(n: int = 20) -> list:
    tasks = ["qa_v1", "summary_v1", "cot_v1"]
    return [{"id": str(i), "task_name": tasks[i % len(tasks)]} for i in range(n)]


class TestStratifiedSplit:
    def test_basic_split(self):
        rows = _make_rows(30)
        train, test = stratified_split(rows, test_fraction=0.2)
        assert len(train) + len(test) == 30
        assert len(test) >= 3  # at least 1 per group

    def test_proportions(self):
        rows = _make_rows(100)
        train, test = stratified_split(rows, test_fraction=0.2)
        # Should be roughly 80/20
        assert 70 <= len(train) <= 85
        assert 15 <= len(test) <= 30

    def test_stratification(self):
        rows = _make_rows(30)  # 10 each of 3 tasks
        train, test = stratified_split(rows, test_fraction=0.2)
        # Each task should be in both train and test
        train_tasks = set(r["task_name"] for r in train)
        test_tasks = set(r["task_name"] for r in test)
        assert len(train_tasks) == 3
        assert len(test_tasks) == 3

    def test_deterministic(self):
        rows = _make_rows(20)
        train1, test1 = stratified_split(rows, seed=42)
        train2, test2 = stratified_split(rows, seed=42)
        assert [r["id"] for r in train1] == [r["id"] for r in train2]
        assert [r["id"] for r in test1] == [r["id"] for r in test2]

    def test_different_seeds(self):
        rows = _make_rows(20)
        train1, _ = stratified_split(rows, seed=42)
        train2, _ = stratified_split(rows, seed=99)
        # Different seeds should (almost certainly) give different orders
        ids1 = [r["id"] for r in train1]
        ids2 = [r["id"] for r in train2]
        assert ids1 != ids2

    def test_very_small_groups(self):
        rows = [
            {"id": "1", "task_name": "rare_task"},
            {"id": "2", "task_name": "rare_task"},
            {"id": "3", "task_name": "common"},
            {"id": "4", "task_name": "common"},
            {"id": "5", "task_name": "common"},
            {"id": "6", "task_name": "common"},
        ]
        train, test = stratified_split(rows, test_fraction=0.3)
        # rare_task has only 2 examples → should all go to train
        assert len(train) + len(test) == 6
