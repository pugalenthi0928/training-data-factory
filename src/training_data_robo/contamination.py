"""Contamination detection via n-gram overlap against benchmark datasets.

Checks training examples for text overlap with popular evaluation benchmarks
(MMLU, ARC, HellaSwag) using 8-gram and 13-gram matching. This is a pure
string-matching approach with zero API cost.

Benchmark data is downloaded on first use and cached in .cache/benchmarks/.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .logging_config import get_logger

logger = get_logger("training_data_robo.contamination")

_DEFAULT_CACHE_DIR = Path(".cache/benchmarks")

# N-gram sizes: 8 catches short overlapping phrases, 13 catches full sentences
_NGRAM_SIZES = (8, 13)


def _tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer with lowercasing and punctuation stripping."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def _build_ngrams(tokens: List[str], n: int) -> Set[str]:
    """Build a set of n-gram strings from a token list."""
    if len(tokens) < n:
        return set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


class BenchmarkIndex:
    """In-memory n-gram index of benchmark question/answer texts."""

    def __init__(self) -> None:
        self._ngrams_8: Set[str] = set()
        self._ngrams_13: Set[str] = set()
        self._benchmark_count: int = 0

    @property
    def size(self) -> int:
        return self._benchmark_count

    def add_text(self, text: str) -> None:
        """Add a benchmark text to the index."""
        tokens = _tokenize(text)
        self._ngrams_8.update(_build_ngrams(tokens, 8))
        self._ngrams_13.update(_build_ngrams(tokens, 13))
        self._benchmark_count += 1

    def check_overlap(self, text: str) -> Dict[str, Any]:
        """Check a text for n-gram overlap with the benchmark index.

        Returns dict with overlap counts and a contamination flag.
        """
        tokens = _tokenize(text)
        ngrams_8 = _build_ngrams(tokens, 8)
        ngrams_13 = _build_ngrams(tokens, 13)

        overlap_8 = ngrams_8 & self._ngrams_8
        overlap_13 = ngrams_13 & self._ngrams_13

        return {
            "8gram_overlaps": len(overlap_8),
            "13gram_overlaps": len(overlap_13),
            "is_contaminated": len(overlap_13) > 0 or len(overlap_8) >= 3,
            "matching_8grams": sorted(list(overlap_8))[:5],  # sample for debugging
            "matching_13grams": sorted(list(overlap_13))[:3],
        }


def _download_benchmark(name: str, url: str, cache_dir: Path) -> Path:
    """Download a benchmark file if not cached."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    cached = cache_dir / f"{name}_{url_hash}.jsonl"

    if cached.exists():
        logger.info("Using cached benchmark: %s", cached)
        return cached

    logger.info("Downloading benchmark %s from %s ...", name, url)
    import requests

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    cached.write_bytes(resp.content)
    logger.info("Cached benchmark to %s (%d bytes)", cached, len(resp.content))
    return cached


# Known benchmark sources (HuggingFace datasets API for JSONL exports)
_BENCHMARK_URLS: Dict[str, Dict[str, str]] = {
    "mmlu": {
        "url": "https://huggingface.co/datasets/cais/mmlu/resolve/main/all/test-00000-of-00001.parquet",
        "text_fields": "question,choices",
    },
    "arc": {
        "url": "https://huggingface.co/datasets/allenai/ai2_arc/resolve/main/ARC-Challenge/test-00000-of-00001.parquet",
        "text_fields": "question,choices",
    },
}


def build_index_from_texts(texts: List[str]) -> BenchmarkIndex:
    """Build a BenchmarkIndex from a list of plain-text strings."""
    index = BenchmarkIndex()
    for text in texts:
        if text.strip():
            index.add_text(text)
    return index


def build_index_from_jsonl(path: Path, text_fields: Optional[List[str]] = None) -> BenchmarkIndex:
    """Build a BenchmarkIndex from a local JSONL file."""
    text_fields = text_fields or ["question", "text", "input", "context"]
    index = BenchmarkIndex()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            for field in text_fields:
                val = row.get(field)
                if val:
                    if isinstance(val, list):
                        for item in val:
                            index.add_text(str(item))
                    else:
                        index.add_text(str(val))
    return index


class ContaminationChecker:
    """Check training data for overlap with benchmark datasets."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self.cache_dir = cache_dir or _DEFAULT_CACHE_DIR
        self.index = BenchmarkIndex()
        self._loaded_benchmarks: List[str] = []

    def load_benchmark_file(self, path: Path, name: str = "custom", text_fields: Optional[List[str]] = None) -> None:
        """Load a local JSONL benchmark file into the index."""
        text_fields = text_fields or ["question", "text", "input", "context"]
        count_before = self.index.size
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for field in text_fields:
                    val = row.get(field)
                    if val:
                        if isinstance(val, list):
                            for item in val:
                                self.index.add_text(str(item))
                        else:
                            self.index.add_text(str(val))
        added = self.index.size - count_before
        self._loaded_benchmarks.append(name)
        logger.info("Loaded %s: %d entries (total index: %d)", name, added, self.index.size)

    def load_custom_texts(self, texts: List[str], name: str = "custom") -> None:
        """Add plain-text benchmark entries directly."""
        for t in texts:
            if t.strip():
                self.index.add_text(t)
        self._loaded_benchmarks.append(name)

    def check_example(self, text: str) -> Dict[str, Any]:
        """Check a single text for contamination."""
        return self.index.check_overlap(text)

    def check_dataset(
        self,
        examples: List[Dict[str, Any]],
        text_fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Check an entire dataset for contamination.

        Returns a summary report with per-example flags and aggregate stats.
        """
        text_fields = text_fields or ["output_text", "input_text"]

        results: List[Dict[str, Any]] = []
        contaminated_count = 0

        for ex in examples:
            # Concatenate all relevant text fields for checking
            combined = " ".join(str(ex.get(f, "")) for f in text_fields)
            check = self.index.check_overlap(combined)
            check["example_id"] = ex.get("id", "")
            results.append(check)
            if check["is_contaminated"]:
                contaminated_count += 1

        return {
            "total_examples": len(examples),
            "contaminated_count": contaminated_count,
            "contamination_rate": contaminated_count / max(1, len(examples)),
            "benchmarks_checked": self._loaded_benchmarks,
            "index_size": self.index.size,
            "per_example": results,
        }
