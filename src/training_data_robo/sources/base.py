from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Union

from ..models import Document


class BaseLoader:
    """Abstract loader interface for specific source types."""

    async def load_documents(
        self,
        sources: Iterable[Union[str, Path]],
    ) -> List[Document]:
        raise NotImplementedError
