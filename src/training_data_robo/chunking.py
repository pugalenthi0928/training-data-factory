from __future__ import annotations

from typing import List, Optional

from .models import Document, TextChunk


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
