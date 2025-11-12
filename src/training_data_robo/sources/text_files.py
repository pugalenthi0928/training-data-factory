from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Union

from ..models import Document
from .base import BaseLoader


class TextFileLoader(BaseLoader):
    """
    Loader for plain text files (.txt).
    """

    def __init__(self, encoding: str = "utf-8") -> None:
        self.encoding = encoding

    async def load_documents(
        self,
        sources: Iterable[Union[str, Path]],
    ) -> List[Document]:
        docs: List[Document] = []

        for src in sources:
            path = Path(str(src))

            if path.is_dir():
                for file_path in path.rglob("*.txt"):
                    text = file_path.read_text(encoding=self.encoding)
                    docs.append(
                        Document.from_file(
                            path=str(file_path),
                            content=text,
                            title=file_path.name,
                        )
                    )
            elif path.is_file() and path.suffix.lower() == ".txt":
                text = path.read_text(encoding=self.encoding)
                docs.append(
                    Document.from_file(
                        path=str(path),
                        content=text,
                        title=path.name,
                    )
                )

        return docs
