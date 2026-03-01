"""Tests for consolidated JSONL I/O utilities."""
from __future__ import annotations

import json
from pathlib import Path

from training_data_robo.io import count_jsonl_rows, iter_jsonl, load_jsonl, write_jsonl


class TestLoadJsonl:
    def test_loads_valid_file(self, tmp_path: Path) -> None:
        path = tmp_path / "data.jsonl"
        path.write_text(
            '{"a": 1}\n{"b": 2}\n', encoding="utf-8"
        )
        records = load_jsonl(path)
        assert len(records) == 2
        assert records[0] == {"a": 1}
        assert records[1] == {"b": 2}

    def test_skips_empty_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "data.jsonl"
        path.write_text('{"a": 1}\n\n\n{"b": 2}\n', encoding="utf-8")
        records = load_jsonl(path)
        assert len(records) == 2

    def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "data.jsonl"
        path.write_text('{"a": 1}\nnot json\n{"b": 2}\n', encoding="utf-8")
        records = load_jsonl(path)
        assert len(records) == 2

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "data.jsonl"
        path.write_text("", encoding="utf-8")
        assert load_jsonl(path) == []

    def test_whitespace_only_file(self, tmp_path: Path) -> None:
        path = tmp_path / "data.jsonl"
        path.write_text("   \n  \n", encoding="utf-8")
        assert load_jsonl(path) == []


class TestWriteJsonl:
    def test_writes_records(self, tmp_path: Path) -> None:
        path = tmp_path / "out.jsonl"
        records = [{"x": 1}, {"y": "hello"}]
        write_jsonl(path, records)
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"x": 1}
        assert json.loads(lines[1]) == {"y": "hello"}

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "sub" / "dir" / "out.jsonl"
        write_jsonl(path, [{"a": 1}])
        assert path.exists()

    def test_empty_records(self, tmp_path: Path) -> None:
        path = tmp_path / "out.jsonl"
        write_jsonl(path, [])
        assert path.read_text(encoding="utf-8") == ""

    def test_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "roundtrip.jsonl"
        original = [{"id": i, "text": f"example {i}"} for i in range(10)]
        write_jsonl(path, original)
        loaded = load_jsonl(path)
        assert loaded == original

    def test_unicode_handling(self, tmp_path: Path) -> None:
        path = tmp_path / "unicode.jsonl"
        records = [{"text": "Hello \u4e16\u754c"}, {"text": "\u00e9\u00e8\u00ea"}]
        write_jsonl(path, records)
        loaded = load_jsonl(path)
        assert loaded[0]["text"] == "Hello \u4e16\u754c"


class TestIterJsonl:
    def test_iterates_lazily(self, tmp_path: Path) -> None:
        path = tmp_path / "data.jsonl"
        path.write_text('{"a": 1}\n{"b": 2}\n{"c": 3}\n', encoding="utf-8")
        results = list(iter_jsonl(path))
        assert len(results) == 3
        assert results[0] == {"a": 1}

    def test_skips_bad_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "data.jsonl"
        path.write_text('{"ok": true}\nbad line\n{"ok": true}\n', encoding="utf-8")
        results = list(iter_jsonl(path))
        assert len(results) == 2


class TestCountJsonlRows:
    def test_counts_rows(self, tmp_path: Path) -> None:
        path = tmp_path / "data.jsonl"
        path.write_text('{"a": 1}\n{"b": 2}\n{"c": 3}\n', encoding="utf-8")
        assert count_jsonl_rows(path) == 3

    def test_skips_empty_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "data.jsonl"
        path.write_text('{"a": 1}\n\n{"b": 2}\n\n', encoding="utf-8")
        assert count_jsonl_rows(path) == 2

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "data.jsonl"
        path.write_text("", encoding="utf-8")
        assert count_jsonl_rows(path) == 0
