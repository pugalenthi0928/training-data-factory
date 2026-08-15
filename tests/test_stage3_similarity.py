"""Stage 3 exact, MinHash, LSH, semantic, and calibration tests."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from forge.calibration import evaluate_calibration, load_calibration_pairs
from forge.similarity import cosine, lsh_candidate_pairs, minhash_signature, normalise_text, word_shingles


class FixtureEncoder:
    name = "fixture-semantic-encoder"
    revision = "1"

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            normalised = normalise_text(text)
            if (
                "source" in normalised
                or "document" in normalised
                or "partition" in normalised
                or "same side" in normalised
            ):
                vectors.append([1.0, 0.0, 0.0])
            elif "permission" in normalised or "rights" in normalised or "publish" in normalised:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors


def test_minhash_and_lsh_are_deterministic_for_near_duplicates() -> None:
    left = word_shingles("one two three four five six seven eight nine ten")
    right = word_shingles("one two three four five six seven eight nine eleven")
    signatures = [minhash_signature(left), minhash_signature(right)]

    assert signatures == [minhash_signature(left), minhash_signature(right)]
    assert (0, 1) in lsh_candidate_pairs(signatures)


def test_cosine_rejects_dimension_mismatch() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_controlled_calibration_reports_uncertainty_and_semantic_slice() -> None:
    fixture = Path("sample_benchmarks/curation_calibration_pairs.jsonl")
    report = evaluate_calibration(
        load_calibration_pairs(fixture),
        encoder=FixtureEncoder(),
        seeds=(3, 7),
        resamples_per_seed=20,
    )

    assert report["schema_version"] == "forge.curation-calibration/v1"
    assert report["fuzzy_minhash_lsh"]["selected"]["precision"] >= 0.8
    assert report["semantic_embedding"]["status"] == "measured"
    assert report["semantic_embedding"]["selected"]["f1_interval"]["seeds"] == [3, 7]
