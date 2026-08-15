"""Canonical Python implementations for Forge's curation stages."""

from __future__ import annotations

import asyncio
import json
import os
import random
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping

from training_data_robo.ai_client import BaseLLMClient, DummyLLMClient, OpenAILLMClient
from training_data_robo.chunking import simple_chunk_document
from training_data_robo.cli import build_task_templates_from_names
from training_data_robo.contamination import ContaminationChecker
from training_data_robo.difficulty import calibrate_batch
from training_data_robo.judge import DummyJudge, LLMJudge
from training_data_robo.models import Document, DocumentSource
from training_data_robo.selector import select_examples
from training_data_robo.sources import UnifiedLoader
from training_data_robo.tasks import TaskManager

from .contracts import (
    ContaminationConfig,
    DedupeConfig,
    DifficultyConfig,
    GenerationConfig,
    IngestConfig,
    JsonValue,
    JudgmentConfig,
    ProfileConfig,
    QualityConfig,
    RecordGovernanceConfig,
    SelectionConfig,
    SourceGovernanceConfig,
    SplitConfig,
    sha256_file,
)
from .governance import audit_decision, govern_documents, govern_records
from .pipeline import StageContext
from .profiling import build_dataset_profile, write_profile
from .similarity import (
    build_encoder,
    cosine,
    lsh_candidate_pairs,
    minhash_signature,
    normalise_text,
    word_shingles,
)

_REFUSAL_PATTERNS = (
    "as an ai language model",
    "i am an ai language model",
    "i'm an ai language model",
    "i cannot",
    "i can't",
    "i am unable to",
    "i'm unable to",
    "cannot provide",
    "cannot answer",
    "sorry, but i",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record at {path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
            count += 1
    return count


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _run_async(coroutine: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    raise RuntimeError("Forge's synchronous pipeline API must run outside an active asyncio event loop")


def run_ingest(context: StageContext, config: IngestConfig) -> Mapping[str, JsonValue]:
    loader = UnifiedLoader()
    documents = _run_async(loader.load_documents([Path(source) for source in config.sources]))
    if not documents:
        raise ValueError("No supported source documents were loaded")

    rows = []
    for document in sorted(documents, key=lambda item: (item.id, item.path or "", item.url or "")):
        rows.append(
            {
                "id": document.id,
                "title": document.title,
                "content": document.content,
                "source": document.source.value,
                "path": document.path,
                "url": document.url,
                "metadata": document.metadata,
            }
        )
    _write_jsonl(context.path("documents.jsonl"), rows)
    return {"documents": len(rows), "declared_sources": len(config.sources)}


def run_source_governance(context: StageContext, config: SourceGovernanceConfig) -> Mapping[str, JsonValue]:
    documents = _read_jsonl(context.path(config.input_path))
    if not documents:
        raise ValueError("Source governance received no documents")
    source_manifest = Path(config.source_manifest).expanduser().resolve() if config.source_manifest else None
    kept, rejected, report = govern_documents(
        documents,
        source_manifest=source_manifest,
        required_use=config.required_use,
        pii_action=config.pii_action,
        allow_unknown_rights=config.allow_unknown_rights,
    )
    _write_jsonl(context.path(config.output_path), kept)
    _write_jsonl(context.path(config.rejected_path), rejected)
    _write_json(context.path(config.report_path), report)
    if report["status"] != "passed":
        raise ValueError(
            "Source governance gate failed: "
            f"kept={report['kept_documents']}, unknown_rights={report['unknown_rights']}, "
            f"disallowed_rights={report['disallowed_rights']}"
        )
    return {
        "input_documents": len(documents),
        "kept_documents": len(kept),
        "rejected_documents": len(rejected),
        "unknown_rights": int(report["unknown_rights"]),
        "pii_findings": int(report["pii_findings"]),
    }


def _document_from_row(row: Mapping[str, Any]) -> Document:
    source_value = str(row.get("source", "text"))
    try:
        source = DocumentSource(source_value)
    except ValueError as exc:
        raise ValueError(f"Unknown document source type: {source_value!r}") from exc
    metadata = row.get("metadata")
    return Document(
        id=str(row["id"]),
        title=str(row.get("title", "Untitled")),
        content=str(row.get("content", "")),
        source=source,
        path=str(row["path"]) if row.get("path") is not None else None,
        url=str(row["url"]) if row.get("url") is not None else None,
        metadata=dict(metadata) if isinstance(metadata, dict) else {},
    )


async def _generate(config: GenerationConfig, documents: list[Document]) -> list[dict[str, Any]]:
    templates = build_task_templates_from_names(list(config.tasks))
    llm_client: BaseLLMClient
    if config.dry_run:
        llm_client = DummyLLMClient()
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for a live generation run")
        llm_client = OpenAILLMClient(model=config.model, api_key=api_key)

    chunks = []
    for document in sorted(documents, key=lambda item: item.id):
        chunks.extend(simple_chunk_document(document, max_chars=config.max_chars, overlap=config.overlap))
    chunks.sort(key=lambda item: (item.document_id, item.index, item.id))
    if not chunks:
        raise ValueError("Source documents did not produce any chunks")

    manager = TaskManager(llm_client)
    examples = await manager.run_tasks_on_chunks(chunks, templates, max_examples=config.max_examples)
    if not examples:
        raise ValueError("Generation produced no examples")

    return [
        {
            "id": example.id,
            "task_name": example.task_name,
            "task_type": example.task_type.value,
            "input_text": example.input_text,
            "output_text": example.output_text,
            "document_id": example.document_id,
            "chunk_id": example.chunk_id,
            "model_name": example.model_name,
            "task_version": example.task_version,
            "temperature": example.temperature,
            "metadata": example.metadata,
        }
        for example in examples
    ]


def run_generation(context: StageContext, config: GenerationConfig) -> Mapping[str, JsonValue]:
    documents = [_document_from_row(row) for row in _read_jsonl(context.path(config.input_path))]
    rows = _run_async(_generate(config, documents))
    _write_jsonl(context.path("raw_dataset.jsonl"), rows)
    return {
        "documents": len(documents),
        "examples": len(rows),
        "tasks": list(config.tasks),
        "generation_mode": "deterministic_dummy" if config.dry_run else "model",
    }


def _normalise_row(row: dict[str, Any]) -> dict[str, Any]:
    if "input_text" not in row:
        row["input_text"] = next((str(row[key]) for key in ("input", "context", "question") if key in row), "")
    if "output_text" not in row:
        row["output_text"] = next((str(row[key]) for key in ("output", "answer") if key in row), "")
    return row


def _quality_score(row: Mapping[str, Any]) -> tuple[list[str], float]:
    flags: list[str] = []
    output = str(row.get("output_text", "") or "")
    answer = str(row.get("answer", "") or "")
    context = str(row.get("context", "") or "")
    task_name = str(row.get("task_name", "") or "").lower()
    task_type = str(row.get("task_type", "") or "").lower()
    score = 1.0

    if not output.strip():
        flags.append("empty_output")
        score -= 0.5
    minimum = 0
    if "summary" in task_name:
        minimum = 80
    elif "key_points" in task_name or "keypoints" in task_name:
        minimum = 60
    elif "title" in task_name or task_type == "qa" or "qa" in task_name:
        minimum = 10
    if minimum and len(output) < minimum:
        flags.append("short_output")
        score -= 0.2
    if any(pattern in output.lower() for pattern in _REFUSAL_PATTERNS):
        flags.append("possible_refusal")
        score -= 0.3

    tokens = output.split()
    if tokens and len(tokens) >= 20 and Counter(tokens).most_common(1)[0][1] / len(tokens) > 0.5:
        flags.append("repetitive_output")
        score -= 0.2
    if answer.strip() and context.strip():
        answer_tokens = set(answer.lower().split())
        overlap = len(answer_tokens & set(context.lower().split()))
        if answer_tokens and overlap < max(1, int(0.2 * len(answer_tokens))):
            flags.append("weak_grounding")
            score -= 0.2
    return sorted(set(flags)), max(0.0, min(1.0, float(score)))


def run_quality(context: StageContext, config: QualityConfig) -> Mapping[str, JsonValue]:
    rows = _read_jsonl(context.path(config.input_path))
    if not rows:
        raise ValueError("Quality stage received an empty dataset")
    flag_counts: Counter[str] = Counter()
    for row in rows:
        _normalise_row(row)
        flags, score = _quality_score(row)
        row["quality_flags"] = flags
        row["quality_score"] = score
        flag_counts.update(flags)
    _write_jsonl(context.path(config.output_path), rows)
    return {"examples": len(rows), "flag_counts": dict(flag_counts)}


def run_record_governance(context: StageContext, config: RecordGovernanceConfig) -> Mapping[str, JsonValue]:
    records = _read_jsonl(context.path(config.input_path))
    if not records:
        raise ValueError("Record governance received no examples")
    kept, rejected, report = govern_records(records, pii_action=config.pii_action, text_fields=config.text_fields)
    _write_jsonl(context.path(config.output_path), kept)
    _write_jsonl(context.path(config.rejected_path), rejected)
    _write_json(context.path(config.report_path), report)
    if report["status"] != "passed":
        raise ValueError("Record governance rejected every example")
    return {
        "input_examples": len(records),
        "kept_examples": len(kept),
        "rejected_examples": len(rejected),
        "redacted_examples": int(report["redacted_records"]),
        "pii_findings": int(report["pii_findings"]),
    }


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _record_text(row: Mapping[str, Any], fields: Iterable[str]) -> str:
    return "\n".join(str(row.get(field, "")) for field in fields)


def run_dedupe(context: StageContext, config: DedupeConfig) -> Mapping[str, JsonValue]:
    rows = _read_jsonl(context.path(config.input_path))
    if not rows:
        raise ValueError("Dedupe stage received an empty dataset")
    if not 0.0 <= config.fuzzy_threshold <= 1.0 or not 0.0 <= config.semantic_threshold <= 1.0:
        raise ValueError("Dedupe thresholds must be between zero and one")
    texts = [_record_text(row, config.text_fields) for row in rows]
    shingles = [word_shingles(text, config.shingle_size) for text in texts]
    signatures = [minhash_signature(value, config.minhash_permutations) for value in shingles]
    candidate_pairs = lsh_candidate_pairs(signatures, config.lsh_bands)
    exact_groups: dict[str, list[int]] = {}
    for index, text in enumerate(texts):
        exact_groups.setdefault(normalise_text(text), []).append(index)
    for group in exact_groups.values():
        candidate_pairs.update(combinations(group, 2))

    encoder = build_encoder(config.semantic_backend, config.semantic_model, config.semantic_revision)
    embeddings: list[list[float]] | None = None
    if encoder is not None:
        if len(rows) > config.semantic_max_records:
            raise ValueError(
                f"Semantic dedupe is limited to {config.semantic_max_records} records in the local Stage 3 adapter"
            )
        embeddings = encoder.encode(texts)
        if len(embeddings) != len(rows):
            raise ValueError("Semantic encoder returned the wrong number of embeddings")

    union = _UnionFind(len(rows))
    pair_evidence: dict[tuple[int, int], dict[str, Any]] = {}
    reason_counts: Counter[str] = Counter()
    pairs_to_score = set(candidate_pairs)
    if embeddings is not None:
        pairs_to_score.update(combinations(range(len(rows)), 2))
    for left, right in sorted(pairs_to_score):
        exact = normalise_text(texts[left]) == normalise_text(texts[right])
        fuzzy = (
            len(shingles[left] & shingles[right]) / len(shingles[left] | shingles[right])
            if shingles[left] | shingles[right]
            else 1.0
        )
        semantic = cosine(embeddings[left], embeddings[right]) if embeddings is not None else None
        reasons = []
        if exact:
            reasons.append("duplicate.exact")
        if (left, right) in candidate_pairs and fuzzy >= config.fuzzy_threshold:
            reasons.append("duplicate.fuzzy_minhash")
        if semantic is not None and semantic >= config.semantic_threshold:
            reasons.append("duplicate.semantic_embedding")
        if reasons:
            union.union(left, right)
            reason_counts.update(reasons)
            pair_evidence[(left, right)] = {
                "reason_codes": reasons,
                "fuzzy_jaccard": round(fuzzy, 6),
                "semantic_cosine": round(semantic, 6) if semantic is not None else None,
            }

    clusters: dict[int, list[int]] = {}
    for index in range(len(rows)):
        clusters.setdefault(union.find(index), []).append(index)
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for members in sorted(clusters.values(), key=lambda values: min(values)):
        representative = min(
            members,
            key=lambda index: (-float(rows[index].get("quality_score", 0.0)), str(rows[index].get("id", index))),
        )
        kept_row = rows[representative]
        audit_decision(
            kept_row,
            control="curation.dedupe",
            outcome="kept",
            reason_codes=("dedupe.representative",) if len(members) > 1 else ("dedupe.unique",),
            evidence={"cluster_size": len(members)},
        )
        kept.append(kept_row)
        for member in members:
            if member == representative:
                continue
            rejected_row = rows[member]
            pair = (min(member, representative), max(member, representative))
            evidence = pair_evidence.get(pair, {"reason_codes": ["duplicate.transitive_cluster"]})
            rejected_row["duplicate_of"] = str(kept_row.get("id", representative))
            audit_decision(
                rejected_row,
                control="curation.dedupe",
                outcome="rejected",
                reason_codes=evidence["reason_codes"],
                evidence={key: value for key, value in evidence.items() if key != "reason_codes"},
            )
            rejected.append(rejected_row)
    _write_jsonl(context.path(config.output_path), kept)
    _write_jsonl(context.path(config.rejected_path), rejected)
    report: dict[str, Any] = {
        "schema_version": "forge.dedupe-report/v1",
        "status": "passed",
        "detectors": [
            "normalised_exact",
            "minhash_lsh_jaccard",
            *(("embedding_cosine",) if encoder is not None else ()),
        ],
        "thresholds": {
            "fuzzy_jaccard": config.fuzzy_threshold,
            "semantic_cosine": config.semantic_threshold if encoder is not None else None,
        },
        "semantic_model": ({"name": encoder.name, "revision": encoder.revision} if encoder is not None else None),
        "input_examples": len(rows),
        "kept_examples": len(kept),
        "dropped_examples": len(rows) - len(kept),
        "clusters": len(clusters),
        "lsh_candidate_pairs": len(candidate_pairs),
        "scored_pairs": len(pairs_to_score),
        "reason_counts": dict(sorted(reason_counts.items())),
    }
    _write_json(context.path(config.report_path), report)
    return report


async def _judge(rows: list[dict[str, Any]], config: JudgmentConfig) -> list[dict[str, Any]]:
    judge = DummyJudge() if config.dry_run else LLMJudge(model=config.model, max_concurrent=config.max_concurrent)
    results = await judge.judge_batch(rows)
    result_map = {result.example_id: result for result in results}
    for row in rows:
        result = result_map.get(str(row.get("id", "")))
        if result is not None:
            row["judge_scores"] = result.to_dict()["verdicts"]
            row["judge_avg_score"] = result.avg_score
    return rows


def run_judgment(context: StageContext, config: JudgmentConfig) -> Mapping[str, JsonValue]:
    rows = _read_jsonl(context.path(config.input_path))
    if not rows:
        raise ValueError("Judgment stage received an empty dataset")
    judged = _run_async(_judge(rows, config))
    _write_jsonl(context.path(config.output_path), judged)
    scores = [float(row.get("judge_avg_score", 0.0)) for row in judged]
    return {
        "examples": len(judged),
        "average_score": round(sum(scores) / len(scores), 4),
        "judge_mode": "deterministic_dummy" if config.dry_run else "model",
    }


def run_contamination(context: StageContext, config: ContaminationConfig) -> Mapping[str, JsonValue]:
    rows = _read_jsonl(context.path(config.input_path))
    if not rows:
        raise ValueError("Contamination stage received an empty dataset")
    checker = ContaminationChecker()
    benchmark_entries: list[dict[str, str]] = []
    for benchmark in config.benchmark_paths:
        path = Path(benchmark).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Benchmark file not found: {path}")
        checker.load_benchmark_file(path, name=path.stem)
        for line_number, record in enumerate(_read_jsonl(path), start=1):
            values: list[str] = []
            for field in ("text", "question", "input", "context", "prompt", "answer", "output"):
                value = record.get(field)
                if isinstance(value, list):
                    values.extend(str(item) for item in value if str(item).strip())
                elif value is not None and str(value).strip():
                    values.append(str(value))
            if values:
                benchmark_entries.append(
                    {"id": f"{path.name}:{line_number}", "text": "\n".join(values), "source": path.name}
                )
    if checker.index.size == 0:
        raise ValueError("No usable benchmark text was loaded")
    if not benchmark_entries:
        raise ValueError("Benchmark files contained no supported text fields")
    comparisons = len(rows) * len(benchmark_entries)
    if comparisons > config.semantic_max_comparisons and config.semantic_backend != "disabled":
        raise ValueError(
            f"Semantic contamination would require {comparisons} comparisons; "
            f"the local Stage 3 limit is {config.semantic_max_comparisons}"
        )
    lexical_report = checker.check_dataset(rows, text_fields=list(config.text_fields))
    row_texts = [_record_text(row, config.text_fields) for row in rows]
    benchmark_texts = [entry["text"] for entry in benchmark_entries]
    row_shingles = [word_shingles(text, config.shingle_size) for text in row_texts]
    benchmark_shingles = [word_shingles(text, config.shingle_size) for text in benchmark_texts]
    encoder = build_encoder(config.semantic_backend, config.semantic_model, config.semantic_revision)
    row_embeddings: list[list[float]] | None = None
    benchmark_embeddings: list[list[float]] | None = None
    if encoder is not None:
        all_embeddings = encoder.encode([*row_texts, *benchmark_texts])
        if len(all_embeddings) != len(row_texts) + len(benchmark_texts):
            raise ValueError("Semantic encoder returned the wrong number of contamination embeddings")
        row_embeddings = all_embeddings[: len(row_texts)]
        benchmark_embeddings = all_embeddings[len(row_texts) :]

    per_example = []
    contaminated = 0
    reason_counts: Counter[str] = Counter()
    lexical_by_id = {str(item.get("example_id", "")): item for item in lexical_report.get("per_example", [])}
    for row_index, row in enumerate(rows):
        lexical = lexical_by_id.get(str(row.get("id", "")), {})
        best_fuzzy = (0.0, 0)
        best_semantic: tuple[float, int] | None = None
        for benchmark_index in range(len(benchmark_entries)):
            union = row_shingles[row_index] | benchmark_shingles[benchmark_index]
            fuzzy = len(row_shingles[row_index] & benchmark_shingles[benchmark_index]) / len(union) if union else 1.0
            if fuzzy > best_fuzzy[0]:
                best_fuzzy = (fuzzy, benchmark_index)
            if row_embeddings is not None and benchmark_embeddings is not None:
                semantic = cosine(row_embeddings[row_index], benchmark_embeddings[benchmark_index])
                if best_semantic is None or semantic > best_semantic[0]:
                    best_semantic = (semantic, benchmark_index)
        reasons: list[str] = []
        if bool(lexical.get("is_contaminated")):
            reasons.append("contamination.lexical_ngram")
        if best_fuzzy[0] >= config.fuzzy_threshold:
            reasons.append("contamination.fuzzy_jaccard")
        if best_semantic is not None and best_semantic[0] >= config.semantic_threshold:
            reasons.append("contamination.semantic_embedding")
        if reasons:
            contaminated += 1
            reason_counts.update(reasons)
        match_index = (
            best_semantic[1]
            if best_semantic is not None and best_semantic[0] >= config.semantic_threshold
            else best_fuzzy[1]
        )
        per_example.append(
            {
                "example_id": row.get("id", ""),
                "is_contaminated": bool(reasons),
                "reason_codes": reasons,
                "lexical_8gram_overlaps": int(lexical.get("8gram_overlaps", 0)),
                "max_fuzzy_jaccard": round(best_fuzzy[0], 6),
                "max_semantic_cosine": round(best_semantic[0], 6) if best_semantic is not None else None,
                "closest_benchmark_id": benchmark_entries[match_index]["id"],
            }
        )
    report: dict[str, Any] = {
        "schema_version": "forge.contamination-report/v2",
        "status": "passed" if contaminated == 0 else "failed",
        "total_examples": len(rows),
        "contaminated_count": contaminated,
        "contamination_rate": contaminated / len(rows),
        "benchmarks_checked": list(lexical_report["benchmarks_checked"]),
        "benchmark_records": len(benchmark_entries),
        "comparisons": comparisons,
        "detectors": [
            "lexical_8gram",
            "fuzzy_shingle_jaccard",
            *(("semantic_embedding",) if encoder is not None else ()),
        ],
        "thresholds": {
            "fuzzy_jaccard": config.fuzzy_threshold,
            "semantic_cosine": config.semantic_threshold if encoder is not None else None,
        },
        "semantic_model": ({"name": encoder.name, "revision": encoder.revision} if encoder is not None else None),
        "reason_counts": dict(sorted(reason_counts.items())),
        "per_example": per_example,
    }
    _write_json(context.path(config.output_path), report)
    if config.fail_on_contamination and contaminated:
        raise ValueError(f"Contamination gate failed with {contaminated} flagged examples")
    return {
        "examples": len(rows),
        "flagged_examples": contaminated,
        "benchmarks": list(report["benchmarks_checked"]),
        "detectors": list(report["detectors"]),
    }


def run_difficulty(context: StageContext, config: DifficultyConfig) -> Mapping[str, JsonValue]:
    rows = _read_jsonl(context.path(config.input_path))
    if not rows:
        raise ValueError("Difficulty stage received an empty dataset")
    summary = calibrate_batch(rows)
    _write_jsonl(context.path(config.output_path), rows)
    return {"examples": len(rows), "distribution": dict(summary["distribution"])}


def run_selection(context: StageContext, config: SelectionConfig) -> Mapping[str, JsonValue]:
    rows = _read_jsonl(context.path(config.input_path))
    if not rows:
        raise ValueError("Selection stage received an empty dataset")
    if config.count is None:
        selected = rows
    else:
        if config.count < 1:
            raise ValueError("Selection count must be positive")
        selected = select_examples(rows, min(config.count, len(rows)), strategy=config.strategy)
    _write_jsonl(context.path(config.output_path), selected)
    return {"input_examples": len(rows), "selected_examples": len(selected), "strategy": config.strategy}


def _identifier_values(rows: Iterable[Mapping[str, Any]], field: str) -> set[str]:
    return {str(row[field]).strip() for row in rows if row.get(field) is not None and str(row[field]).strip()}


def _assert_no_overlap(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    field: str,
    *,
    allow_missing: bool = False,
) -> None:
    if not allow_missing:
        missing = [row for row in [*train, *test] if row.get(field) is None or not str(row[field]).strip()]
        if missing:
            raise ValueError(f"Split contains {len(missing)} rows without required field {field!r}")
    overlap = _identifier_values(train, field) & _identifier_values(test, field)
    if overlap:
        raise ValueError(f"Unsafe split: {len(overlap)} {field!r} values occur in both partitions")


def _choose_test_sources(
    source_groups: dict[str, list[dict[str, Any]]], test_fraction: float, rng: random.Random
) -> set[str]:
    source_ids = sorted(source_groups)
    rng.shuffle(source_ids)
    target_rows = sum(len(group) for group in source_groups.values()) * test_fraction
    target_sources = len(source_ids) * test_fraction
    candidates: list[set[str]] = [{source_id} for source_id in source_ids]
    for _ in range(min(128, max(16, len(source_ids) * 4))):
        order = list(source_ids)
        rng.shuffle(order)
        chosen: set[str] = set()
        chosen_rows = 0
        for source_id in order:
            if len(chosen) >= len(source_ids) - 1:
                break
            group_size = len(source_groups[source_id])
            if not chosen or abs(chosen_rows + group_size - target_rows) < abs(chosen_rows - target_rows):
                chosen.add(source_id)
                chosen_rows += group_size
        candidates.append(chosen)

    def score(candidate: set[str]) -> tuple[float, float]:
        rows = sum(len(source_groups[source_id]) for source_id in candidate)
        return abs(rows - target_rows), abs(len(candidate) - target_sources)

    return min(candidates, key=score)


def _source_safe_split(
    rows: list[dict[str, Any]], config: SplitConfig
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < config.test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    source_groups: dict[str, list[dict[str, Any]]] = {}
    missing: list[int] = []
    for index, row in enumerate(rows):
        source_id = str(row.get(config.source_field, "")).strip()
        if not source_id:
            missing.append(index)
        else:
            source_groups.setdefault(source_id, []).append(row)
    if missing:
        raise ValueError(f"{len(missing)} rows are missing required provenance field {config.source_field!r}")
    if len(source_groups) < 2:
        raise ValueError(f"At least two unique {config.source_field!r} values are required for a safe split")

    rng = random.Random(config.seed)
    test_sources = _choose_test_sources(source_groups, config.test_fraction, rng)
    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for source_id, group in source_groups.items():
        (test if source_id in test_sources else train).extend(group)
    rng.shuffle(train)
    rng.shuffle(test)
    _assert_no_overlap(train, test, config.source_field)
    _assert_no_overlap(train, test, "chunk_id", allow_missing=True)
    return train, test


def run_split(context: StageContext, config: SplitConfig) -> Mapping[str, JsonValue]:
    rows = _read_jsonl(context.path(config.input_path))
    if not rows:
        raise ValueError("Split stage received an empty dataset")
    train, test = _source_safe_split(rows, config)
    train_path = context.path(config.train_path)
    test_path = context.path(config.test_path)
    _write_jsonl(train_path, train)
    _write_jsonl(test_path, test)
    train_sources = _identifier_values(train, config.source_field)
    test_sources = _identifier_values(test, config.source_field)
    manifest = {
        "schema_version": 1,
        "method": "source_grouped",
        "seed": config.seed,
        "test_fraction": config.test_fraction,
        "source_field": config.source_field,
        "stratify_field": config.stratify_field,
        "total": len(rows),
        "train": len(train),
        "test": len(test),
        "achieved_test_fraction": round(len(test) / len(rows), 6),
        "train_sources": len(train_sources),
        "test_sources": len(test_sources),
        "source_overlap": sorted(train_sources & test_sources),
        "train_distribution": dict(Counter(str(row.get(config.stratify_field, "unknown")) for row in train)),
        "test_distribution": dict(Counter(str(row.get(config.stratify_field, "unknown")) for row in test)),
        "artifacts": {
            "input": {"path": config.input_path, "sha256": sha256_file(context.path(config.input_path))},
            "train": {"path": config.train_path, "sha256": sha256_file(train_path)},
            "test": {"path": config.test_path, "sha256": sha256_file(test_path)},
        },
    }
    _write_json(context.path(config.manifest_path), manifest)
    return {
        "train_examples": len(train),
        "test_examples": len(test),
        "train_sources": len(train_sources),
        "test_sources": len(test_sources),
        "source_overlap": 0,
    }


def run_profile(context: StageContext, config: ProfileConfig) -> Mapping[str, JsonValue]:
    train = _read_jsonl(context.path(config.train_path))
    test = _read_jsonl(context.path(config.test_path))
    if not train or not test:
        raise ValueError("Dataset profile requires non-empty train and test partitions")

    def read_json(path: str) -> dict[str, Any]:
        value = json.loads(context.path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Profile input {path!r} must be a JSON object")
        return value

    artifact_paths = {
        "train": context.path(config.train_path),
        "test": context.path(config.test_path),
        "source_governance": context.path(config.source_governance_path),
        "record_governance": context.path(config.record_governance_path),
        "dedupe_report": context.path(config.dedupe_report_path),
        "contamination_report": context.path(config.contamination_path),
    }
    profile = build_dataset_profile(
        train=train,
        test=test,
        source_governance=read_json(config.source_governance_path),
        record_governance=read_json(config.record_governance_path),
        dedupe=read_json(config.dedupe_report_path),
        contamination=read_json(config.contamination_path),
        rejected_sources=_read_jsonl(context.path(config.rejected_sources_path)),
        rejected_records=_read_jsonl(context.path(config.rejected_records_path)),
        dedupe_rejections=_read_jsonl(context.path(config.dedupe_rejections_path)),
        artifact_paths=artifact_paths,
    )
    write_profile(context.path(config.output_path), profile)
    return {
        "records": len(train) + len(test),
        "sources": int(profile["records"]["sources"]),
        "rejections": int(profile["curation"]["rejections"]["total"]),
    }
