"""Tests for train/test split utility."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from split_dataset import assert_no_identifier_overlap, stratified_split


def _make_rows(n_sources: int = 10) -> list:
    tasks = ["qa_v1", "summary_v1", "cot_v1"]
    rows = []
    for source_index in range(n_sources):
        for task_index, task in enumerate(tasks):
            rows.append(
                {
                    "id": f"{source_index}-{task_index}",
                    "task_name": task,
                    "document_id": f"doc-{source_index}",
                    "chunk_id": f"chunk-{source_index}-{task_index}",
                }
            )
    return rows


class TestStratifiedSplit:
    def test_basic_split(self):
        rows = _make_rows(10)
        train, test = stratified_split(rows, test_fraction=0.2)
        assert len(train) + len(test) == 30
        assert len(test) == 6

    def test_proportions(self):
        rows = _make_rows(100)
        train, test = stratified_split(rows, test_fraction=0.2)
        assert len(train) == 240
        assert len(test) == 60

    def test_stratification(self):
        rows = _make_rows(10)
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

    def test_source_documents_never_cross_partitions(self):
        train, test = stratified_split(_make_rows(12), test_fraction=0.25)
        train_sources = {row["document_id"] for row in train}
        test_sources = {row["document_id"] for row in test}
        assert train_sources.isdisjoint(test_sources)

    def test_uneven_sources_choose_closest_row_fraction(self):
        rows = []
        for source_id, count in {"large-a": 40, "large-b": 40, "tiny": 4}.items():
            rows.extend(
                {
                    "id": f"{source_id}-{index}",
                    "document_id": source_id,
                    "chunk_id": f"{source_id}-chunk-{index}",
                }
                for index in range(count)
            )

        _, test = stratified_split(rows, test_fraction=0.2, seed=3)
        assert len(test) == 4

    def test_missing_source_provenance_fails_closed(self):
        rows = _make_rows(3)
        del rows[0]["document_id"]
        with pytest.raises(ValueError, match="missing required provenance"):
            stratified_split(rows)

    def test_single_source_cannot_be_split_safely(self):
        with pytest.raises(ValueError, match="At least two unique"):
            stratified_split(_make_rows(1))

    def test_invalid_fraction_is_rejected(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            stratified_split(_make_rows(3), test_fraction=1.0)

    def test_chunk_overlap_is_detected(self):
        train = [{"document_id": "doc-a", "chunk_id": "shared"}]
        test = [{"document_id": "doc-b", "chunk_id": "shared"}]
        with pytest.raises(ValueError, match="Unsafe split"):
            assert_no_identifier_overlap(train, test, "chunk_id")
