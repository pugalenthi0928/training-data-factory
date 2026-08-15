from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Union

from .models import Document


class BaseLoader:
    """Abstract loader interface."""

    async def load_documents(self, sources: Iterable[Union[str, Path]]) -> List[Document]:
        """
        Given a list of sources (files / folders / raw text),
        return a list of normalized Document objects.
        """
        raise NotImplementedError


class SimpleFileAndTextLoader(BaseLoader):
    """
    Minimal loader that supports:
      - .txt files
      - directories of .txt files (scans recursively)
      - raw text strings (fallback)
    """

    def __init__(self, encoding: str = "utf-8") -> None:
        self.encoding = encoding

    async def load_documents(self, sources: Iterable[Union[str, Path]]) -> List[Document]:
        docs: List[Document] = []

        for src in sources:
            path = Path(str(src))

            if path.exists():
                if path.is_dir():
                    # Recursively load all .txt files in the folder
                    for file_path in path.rglob("*.txt"):
                        text = file_path.read_text(encoding=self.encoding)
                        docs.append(Document.from_file(str(file_path), content=text))
                elif path.is_file():
                    # Single file
                    text = path.read_text(encoding=self.encoding)
                    docs.append(Document.from_file(str(path), content=text))
            else:
                # Fallback: treat the source as raw text
                docs.append(Document.from_text(content=str(src)))

        return docs
