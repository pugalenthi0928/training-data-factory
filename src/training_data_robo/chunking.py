from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .models import ChunkType, Document, TextChunk

# ---------------------------------------------------------------------------
# Structure detection helpers
# ---------------------------------------------------------------------------

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_LIST_ITEM_RE = re.compile(r"^[\s]*[-*+]\s|^\s*\d+[.)]\s", re.MULTILINE)
_TABLE_ROW_RE = re.compile(r"^\|.+\|$", re.MULTILINE)


def _classify_block(text: str) -> ChunkType:
    """Classify a text block as prose, list, table, or mixed."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ChunkType.PROSE

    list_lines = sum(1 for ln in lines if _LIST_ITEM_RE.match(ln))
    table_lines = sum(1 for ln in lines if _TABLE_ROW_RE.match(ln))
    total = len(lines)

    list_frac = list_lines / total
    table_frac = table_lines / total

    if table_frac > 0.5:
        return ChunkType.TABLE
    if list_frac > 0.5:
        return ChunkType.LIST
    if list_frac > 0.2 or table_frac > 0.2:
        return ChunkType.MIXED
    return ChunkType.PROSE


def _detect_section(text: str) -> Tuple[Optional[str], Optional[int]]:
    """Return (section_title, heading_level) from the first heading found, or (None, None)."""
    m = _HEADER_RE.search(text)
    if m:
        return m.group(2).strip(), len(m.group(1))
    return None, None


# ---------------------------------------------------------------------------
# Section-aware splitter
# ---------------------------------------------------------------------------


def _split_into_sections(text: str) -> List[Dict]:
    """Split text on markdown headings, preserving section metadata.

    Returns a list of dicts: {text, section_title, section_level}.
    Non-headed text at the top becomes section_title=None.
    """
    parts = _HEADER_RE.split(text)
    # re.split with groups: [pre_text, hashes1, title1, body1, hashes2, title2, body2, ...]
    sections: List[Dict] = []

    # First element is text before any heading
    preamble = parts[0].strip()
    if preamble:
        sections.append(
            {
                "text": preamble,
                "section_title": None,
                "section_level": None,
            }
        )

    # Iterate over (hashes, title, body) triples
    i = 1
    while i + 2 <= len(parts):
        hashes = parts[i]
        title = parts[i + 1].strip()
        body = parts[i + 2].strip() if i + 2 < len(parts) else ""
        full_text = f"{'#' * len(hashes)} {title}\n\n{body}".strip()
        sections.append(
            {
                "text": full_text,
                "section_title": title,
                "section_level": len(hashes),
            }
        )
        i += 3

    return sections if sections else [{"text": text, "section_title": None, "section_level": None}]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simple_chunk_document(
    document: Document,
    max_chars: int = 800,
    overlap: int = 100,
) -> List[TextChunk]:
    """
    Very simple paragraph-aware chunker.

    - Splits document into paragraphs on blank lines.
    - Packs paragraphs together until max_chars is reached.
    - Adds a bit of overlap between chunks for context.
    """

    text = document.content
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: List[TextChunk] = []
    current: str = ""
    idx: int = 0

    for para in paragraphs:
        # +2 to account for the "\n\n" we'll add when joining
        candidate = (current + "\n\n" + para) if current else para
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(
                    TextChunk.from_document(
                        document=document,
                        text=current,
                        index=idx,
                    )
                )
                idx += 1

                # overlap: take the last `overlap` characters from previous chunk
                overlap_text: str = current[-overlap:] if overlap > 0 else ""
                current = (overlap_text + "\n\n" + para).strip()
            else:
                # Single paragraph longer than max_chars → hard split
                start = 0
                para_text = para
                while start < len(para_text):
                    end = start + max_chars
                    piece = para_text[start:end]
                    chunks.append(
                        TextChunk.from_document(
                            document=document,
                            text=piece,
                            index=idx,
                        )
                    )
                    idx += 1
                    start = end

                current = ""

    # Flush the last chunk
    if current:
        chunks.append(
            TextChunk.from_document(
                document=document,
                text=current,
                index=idx,
            )
        )

    return chunks


def structure_aware_chunk_document(
    document: Document,
    max_chars: int = 800,
    overlap: int = 100,
) -> List[TextChunk]:
    """
    Structure-aware chunker that detects markdown headings, lists, and tables.

    Each TextChunk.metadata is enriched with:
      - section_title: heading text (or None)
      - section_level: heading depth 1-6 (or None)
      - chunk_type: 'prose' | 'list' | 'table' | 'mixed'

    Strategy:
      1. Split the document on markdown headings into logical sections.
      2. Within each section, pack paragraphs up to max_chars (like simple_chunk_document).
      3. Classify each resulting chunk by content type.
    """
    sections = _split_into_sections(document.content)

    chunks: List[TextChunk] = []
    idx = 0

    for section in sections:
        sec_text = section["text"]
        sec_title = section["section_title"]
        sec_level = section["section_level"]

        paragraphs = [p.strip() for p in sec_text.split("\n\n") if p.strip()]

        current = ""
        for para in paragraphs:
            candidate = (current + "\n\n" + para) if current else para
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunk_type = _classify_block(current)
                    chunks.append(
                        TextChunk.from_document(
                            document=document,
                            text=current,
                            index=idx,
                            metadata={
                                "section_title": sec_title,
                                "section_level": sec_level,
                                "chunk_type": chunk_type.value,
                            },
                        )
                    )
                    idx += 1

                    overlap_text = current[-overlap:] if overlap > 0 else ""
                    current = (overlap_text + "\n\n" + para).strip()
                else:
                    # Single paragraph longer than max_chars → hard split
                    start = 0
                    while start < len(para):
                        end = start + max_chars
                        piece = para[start:end]
                        chunk_type = _classify_block(piece)
                        chunks.append(
                            TextChunk.from_document(
                                document=document,
                                text=piece,
                                index=idx,
                                metadata={
                                    "section_title": sec_title,
                                    "section_level": sec_level,
                                    "chunk_type": chunk_type.value,
                                },
                            )
                        )
                        idx += 1
                        start = end
                    current = ""

        # Flush remaining text for this section
        if current:
            chunk_type = _classify_block(current)
            chunks.append(
                TextChunk.from_document(
                    document=document,
                    text=current,
                    index=idx,
                    metadata={
                        "section_title": sec_title,
                        "section_level": sec_level,
                        "chunk_type": chunk_type.value,
                    },
                )
            )
            idx += 1

    return chunks
