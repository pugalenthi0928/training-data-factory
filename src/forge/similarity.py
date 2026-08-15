"""Deterministic lexical similarity and optional semantic embedding adapters."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Protocol, Sequence


class EmbeddingEncoder(Protocol):
    """Small adapter boundary used by dedupe and contamination controls."""

    name: str
    revision: str

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class SentenceTransformerEncoder:
    """Optional local sentence-transformers adapter loaded only when requested."""

    def __init__(self, model_name: str, revision: str = "main") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "semantic backend 'sentence_transformers' requires `pip install -e '.[semantic]'`"
            ) from exc
        self.name = model_name
        self.revision = revision
        self._model = SentenceTransformer(model_name, revision=revision)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return [[float(value) for value in vector] for vector in vectors]


class OpenAIEmbeddingEncoder:
    """OpenAI embeddings adapter for live candidate runs."""

    def __init__(self, model_name: str) -> None:
        from openai import OpenAI

        self.name = model_name
        self.revision = "provider-managed"
        self._client = OpenAI()

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.name, input=list(texts))
        ordered = sorted(response.data, key=lambda item: item.index)
        return [[float(value) for value in item.embedding] for item in ordered]


def build_encoder(backend: str, model_name: str, revision: str = "main") -> EmbeddingEncoder | None:
    """Construct a semantic encoder without making disabled mode import heavy packages."""
    if backend == "disabled":
        return None
    if backend == "sentence_transformers":
        return SentenceTransformerEncoder(model_name, revision)
    if backend == "openai":
        return OpenAIEmbeddingEncoder(model_name)
    raise ValueError(f"Unsupported semantic backend: {backend!r}")


def normalise_text(text: str) -> str:
    """Return a stable, conservative text form for exact comparison."""
    return " ".join("".join(character.lower() if character.isalnum() else " " for character in text).split())


def word_shingles(text: str, size: int = 5) -> set[str]:
    tokens = normalise_text(text).split()
    if not tokens:
        return set()
    if len(tokens) < size:
        return {" ".join(tokens)}
    return {" ".join(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding vectors must have the same dimension")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def _hash64(value: str, seed: int) -> int:
    payload = seed.to_bytes(4, byteorder="big", signed=False) + value.encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), byteorder="big")


def minhash_signature(shingles: set[str], permutations: int = 64) -> tuple[int, ...]:
    """Create a deterministic MinHash signature for a set of word shingles."""
    if permutations < 8:
        raise ValueError("MinHash requires at least 8 permutations")
    if not shingles:
        return tuple(0 for _ in range(permutations))
    return tuple(min(_hash64(shingle, seed) for shingle in shingles) for seed in range(permutations))


def lsh_candidate_pairs(signatures: Sequence[tuple[int, ...]], bands: int = 16) -> set[tuple[int, int]]:
    """Return candidate pairs that share at least one MinHash LSH band."""
    if not signatures:
        return set()
    width = len(signatures[0])
    if width == 0 or width % bands:
        raise ValueError("Signature length must be divisible by the number of LSH bands")
    if any(len(signature) != width for signature in signatures):
        raise ValueError("All MinHash signatures must have equal length")
    rows = width // bands
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    for index, signature in enumerate(signatures):
        for band in range(bands):
            start = band * rows
            buckets[(band, signature[start : start + rows])].append(index)
    pairs: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for offset, left in enumerate(members):
            for right in members[offset + 1 :]:
                pairs.add((min(left, right), max(left, right)))
    return pairs


@dataclass(frozen=True)
class SimilarityEvidence:
    exact: bool
    fuzzy_jaccard: float
    semantic_cosine: float | None

    def reason_codes(self, *, fuzzy_threshold: float, semantic_threshold: float) -> list[str]:
        reasons: list[str] = []
        if self.exact:
            reasons.append("duplicate.exact")
        if self.fuzzy_jaccard >= fuzzy_threshold:
            reasons.append("duplicate.fuzzy_minhash")
        if self.semantic_cosine is not None and self.semantic_cosine >= semantic_threshold:
            reasons.append("duplicate.semantic_embedding")
        return reasons


def pair_evidence(
    left: str,
    right: str,
    *,
    shingle_size: int = 5,
    left_embedding: Sequence[float] | None = None,
    right_embedding: Sequence[float] | None = None,
) -> SimilarityEvidence:
    semantic = None
    if left_embedding is not None and right_embedding is not None:
        semantic = cosine(left_embedding, right_embedding)
    return SimilarityEvidence(
        exact=normalise_text(left) == normalise_text(right),
        fuzzy_jaccard=jaccard(word_shingles(left, shingle_size), word_shingles(right, shingle_size)),
        semantic_cosine=semantic,
    )


def embedding_metadata(encoder: EmbeddingEncoder | None) -> dict[str, Any]:
    if encoder is None:
        return {"backend": "disabled", "model": None, "revision": None}
    return {"backend": type(encoder).__name__, "model": encoder.name, "revision": encoder.revision}
