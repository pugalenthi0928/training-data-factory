from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Union

from pypdf import PdfReader

from ..models import Document
from .base import BaseLoader


class PdfFileLoader(BaseLoader):
    """
    Loader for PDF files using pypdf.
    """

    async def load_documents(
        self,
        sources: Iterable[Union[str, Path]],
    ) -> List[Document]:
        docs: List[Document] = []

        for src in sources:
            path = Path(str(src))

            if path.is_dir():
                for file_path in path.rglob("*.pdf"):
                    docs.append(self._load_single_pdf(file_path))
            elif path.is_file() and path.suffix.lower() == ".pdf":
                docs.append(self._load_single_pdf(path))

        return docs

    def _load_single_pdf(self, path: Path) -> Document:
        reader = PdfReader(str(path))
        pages_text = []

        for page in reader.pages:
            text = page.extract_text() or ""
            pages_text.append(text)

        full_text = "\n\n".join(pages_text)

        metadata = {
            "num_pages": len(reader.pages),
        }

        return Document.from_file(
            path=str(path),
            content=full_text,
            title=path.name,
            metadata=metadata,
        )
