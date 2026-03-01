from __future__ import annotations

import re
from typing import Iterable, List

import requests

from ..models import Document
from .base import BaseLoader


class WebPageLoader(BaseLoader):
    """
    Very simple web page loader:
      - fetches each URL with requests
      - strips HTML tags with a regex
      - wraps result as a Document
    """

    def __init__(self, timeout: int = 10) -> None:
        self.timeout = timeout

    async def load_documents(
        self,
        sources: Iterable[str],
    ) -> List[Document]:
        docs: List[Document] = []
        for url in sources:
            try:
                resp = requests.get(url, timeout=self.timeout)
                resp.raise_for_status()
                html = resp.text
                text = self._strip_html(html)
                docs.append(
                    Document.from_url(
                        url=url,
                        content=text,
                        title=url,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                # In a real system, you'd log this more carefully
                print(f"[WebPageLoader] Failed to load {url}: {exc}")
        return docs

    def _strip_html(self, html: str) -> str:
        # SUPER naive HTML tag stripper; good enough for learning.
        text = re.sub(r"<script.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        # collapse whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()
