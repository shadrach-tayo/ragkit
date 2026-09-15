"""Vector store protocols and backends."""

from typing import Protocol

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class VectorStoreLike(Protocol):
    """Minimal document store interface used by the RAG pipeline."""

    def add_documents(self, documents: list[Document]) -> None:
        """Persist documents in the store."""
        ...

    def as_retriever(self, k: int = 4) -> BaseRetriever:
        """Return a retriever limited to ``k`` results."""
        ...
