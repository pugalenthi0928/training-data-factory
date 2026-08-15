"""Tests for structure-aware chunking."""

from __future__ import annotations

from training_data_robo.chunking import (
    _classify_block,
    _detect_section,
    _split_into_sections,
    simple_chunk_document,
    structure_aware_chunk_document,
)
from training_data_robo.models import ChunkType, Document


def _make_doc(content: str, title: str = "test") -> Document:
    return Document.from_text(content=content, title=title)


# ---------------------------------------------------------------------------
# Block classification
# ---------------------------------------------------------------------------


class TestClassifyBlock:
    def test_prose(self):
        text = "This is a regular paragraph of prose text about machine learning."
        assert _classify_block(text) == ChunkType.PROSE

    def test_list_bullets(self):
        text = "- Item one\n- Item two\n- Item three\n- Item four"
        assert _classify_block(text) == ChunkType.LIST

    def test_list_numbered(self):
        text = "1. First point\n2. Second point\n3. Third point"
        assert _classify_block(text) == ChunkType.LIST

    def test_table(self):
        text = "| Col A | Col B |\n|-------|-------|\n| val1  | val2  |\n| val3  | val4  |"
        assert _classify_block(text) == ChunkType.TABLE

    def test_mixed(self):
        text = "Some prose here.\n- item one\n- item two\nMore prose follows after the list."
        assert _classify_block(text) == ChunkType.MIXED

    def test_empty(self):
        assert _classify_block("") == ChunkType.PROSE
        assert _classify_block("   \n  \n  ") == ChunkType.PROSE


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------


class TestDetectSection:
    def test_h1(self):
        title, level = _detect_section("# Introduction\nSome text here.")
        assert title == "Introduction"
        assert level == 1

    def test_h3(self):
        title, level = _detect_section("### Methodology\nDetails follow.")
        assert title == "Methodology"
        assert level == 3

    def test_no_heading(self):
        title, level = _detect_section("Just some plain text without any headings.")
        assert title is None
        assert level is None


# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------


class TestSplitIntoSections:
    def test_single_section(self):
        text = "# Title\n\nBody text here."
        sections = _split_into_sections(text)
        assert len(sections) == 1
        assert sections[0]["section_title"] == "Title"
        assert sections[0]["section_level"] == 1

    def test_preamble_plus_sections(self):
        text = "Preamble text.\n\n# Section One\n\nBody one.\n\n## Section Two\n\nBody two."
        sections = _split_into_sections(text)
        assert len(sections) == 3
        assert sections[0]["section_title"] is None  # preamble
        assert sections[1]["section_title"] == "Section One"
        assert sections[2]["section_title"] == "Section Two"
        assert sections[2]["section_level"] == 2

    def test_no_headings(self):
        text = "Just plain text.\n\nAnother paragraph."
        sections = _split_into_sections(text)
        assert len(sections) == 1
        assert sections[0]["section_title"] is None


# ---------------------------------------------------------------------------
# Structure-aware chunker
# ---------------------------------------------------------------------------


class TestStructureAwareChunker:
    def test_basic_prose(self):
        content = "# Introduction\n\n" + ("Hello world. " * 50)
        doc = _make_doc(content)
        chunks = structure_aware_chunk_document(doc, max_chars=300)
        assert len(chunks) >= 1
        for c in chunks:
            assert "chunk_type" in c.metadata
            assert c.metadata["section_title"] == "Introduction"

    def test_list_chunk_detected(self):
        content = "# Items\n\n- Apple\n- Banana\n- Cherry\n- Date\n- Elderberry"
        doc = _make_doc(content)
        chunks = structure_aware_chunk_document(doc, max_chars=2000)
        assert len(chunks) == 1
        assert chunks[0].metadata["chunk_type"] == "list"

    def test_multiple_sections(self):
        content = "# Intro\n\nIntro text here.\n\n## Methods\n\nMethods text here.\n\n## Results\n\nResults text here."
        doc = _make_doc(content)
        chunks = structure_aware_chunk_document(doc, max_chars=2000)
        titles = [c.metadata["section_title"] for c in chunks]
        assert "Intro" in titles
        assert "Methods" in titles
        assert "Results" in titles

    def test_respects_max_chars(self):
        # Use actual paragraphs separated by \n\n so the chunker can split
        paras = "\n\n".join(f"Paragraph {i} has some content here." for i in range(30))
        long_section = "# Data\n\n" + paras
        doc = _make_doc(long_section)
        chunks = structure_aware_chunk_document(doc, max_chars=300)
        assert len(chunks) >= 2
        for c in chunks:
            # Allow slack for overlap
            assert len(c.text) <= 600

    def test_empty_document(self):
        doc = _make_doc("")
        chunks = structure_aware_chunk_document(doc)
        assert chunks == []


# ---------------------------------------------------------------------------
# Simple chunker still works (regression)
# ---------------------------------------------------------------------------


class TestSimpleChunkerRegression:
    def test_basic(self):
        doc = _make_doc("Para one.\n\nPara two.\n\nPara three.")
        chunks = simple_chunk_document(doc, max_chars=2000)
        assert len(chunks) == 1
        assert "Para one" in chunks[0].text

    def test_splits_on_size(self):
        doc = _make_doc("A" * 500 + "\n\n" + "B" * 500)
        chunks = simple_chunk_document(doc, max_chars=600)
        assert len(chunks) >= 2
