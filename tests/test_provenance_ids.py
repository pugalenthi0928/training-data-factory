"""Tests for stable document and chunk provenance identifiers."""

from training_data_robo.models import Document, TextChunk


def test_document_id_is_stable_for_identical_content() -> None:
    first = Document.from_file("/tmp/first.txt", "Same source text")
    second = Document.from_file("/different/path.txt", "Same  source\ntext")

    assert first.id == second.id
    assert first.id.startswith("doc_")


def test_document_id_changes_when_content_changes() -> None:
    first = Document.from_text("Version one")
    second = Document.from_text("Version two")

    assert first.id != second.id


def test_chunk_id_is_stable_and_source_scoped() -> None:
    document = Document.from_text("A source document")
    first = TextChunk.from_document(document, "Chunk content", index=0)
    second = TextChunk.from_document(document, "Chunk  content", index=0)
    different_index = TextChunk.from_document(document, "Chunk content", index=1)

    assert first.id == second.id
    assert first.id.startswith("chunk_")
    assert first.id != different_index.id
