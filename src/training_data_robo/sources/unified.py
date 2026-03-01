from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Union

from ..models import Document
from .base import BaseLoader
from .pdf_files import PdfFileLoader
from .text_files import TextFileLoader
from .web_pages import WebPageLoader


class UnifiedLoader(BaseLoader):
    """
    Unified loader that can:
      - load .txt files (and folders of .txt)
      - load .pdf files (and folders of .pdf)
      - load web pages from http(s) URLs
      - treat everything else as raw text
    """

    def __init__(self, encoding: str = "utf-8") -> None:
        self.text_loader = TextFileLoader(encoding=encoding)
        self.pdf_loader = PdfFileLoader()
        self.web_loader = WebPageLoader()

    async def load_documents(
        self,
        sources: Iterable[Union[str, Path]],
    ) -> List[Document]:
        docs: List[Document] = []
        text_sources: List[Union[str, Path]] = []
        pdf_sources: List[Union[str, Path]] = []
        url_sources: List[str] = []
        raw_texts: List[str] = []

        for src in sources:
            s = str(src)

            # URL?
            if s.startswith("http://") or s.startswith("https://"):
                url_sources.append(s)
                continue

            path = Path(s)
            if path.exists():
                if path.is_dir():
                    # let file loaders recurse & filter by extension
                    text_sources.append(path)
                    pdf_sources.append(path)
                elif path.is_file():
                    if path.suffix.lower() == ".pdf":
                        pdf_sources.append(path)
                    else:
                        text_sources.append(path)
            else:
                # Non-existing path and not a URL: treat as raw text
                raw_texts.append(s)

        if text_sources:
            docs.extend(await self.text_loader.load_documents(text_sources))

        if pdf_sources:
            docs.extend(await self.pdf_loader.load_documents(pdf_sources))

        if url_sources:
            docs.extend(await self.web_loader.load_documents(url_sources))

        for text in raw_texts:
            docs.append(Document.from_text(content=text))

        return docs
