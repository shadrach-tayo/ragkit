"""Postgres + pgvector store."""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from ragkit.config import get_settings
from ragkit.embeddings import voyage_embeddings

logger = logging.getLogger(__name__)


class PostgresVectorStoreManager:
    """Vector table backed by langchain-postgres / pgvector."""

    def __init__(
        self,
        index_name: str,
        *,
        embedding_dim: int = 1024,
        chunk_size: int | None = None,
        chunk_overlap: int = 30,
        embeddings: Embeddings | None = None,
        overwrite_existing: bool = False,
        database_url: str | None = None,
    ) -> None:
        """Connect to Postgres and open (or create) the named vector table."""
        try:
            from langchain_postgres import PGEngine, PGVectorStore
        except ImportError as exc:
            raise ImportError("Install ragkit[postgres] to use pgvector") from exc

        url = database_url or get_settings().database_url
        if not url:
            raise ValueError("DATABASE_URL is not set")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index_name = index_name
        self.engine = PGEngine.from_connection_string(url=url)
        if chunk_size:
            self.engine.init_vectorstore_table(
                table_name=index_name,
                vector_size=embedding_dim,
                overwrite_existing=overwrite_existing,
            )
        self.vector_store = PGVectorStore.create_sync(
            engine=self.engine,
            table_name=index_name,
            embedding_service=embeddings or voyage_embeddings(),
        )

    def add_documents(self, documents: list[Document]) -> None:
        """Split and persist documents."""
        splits = self._split_documents(documents)
        self.vector_store.add_documents(splits)
        logger.info("Added %s chunks to vector store %s", len(splits), self.index_name)

    def as_retriever(self, k: int = 4) -> BaseRetriever:
        """Return a similarity retriever limited to ``k`` results."""
        return self.vector_store.as_retriever(search_kwargs={"k": k})

    def _split_documents(self, documents: list[Document]) -> list[Document]:
        if self.chunk_size is None:
            raise ValueError("chunk_size is required to split documents")
        try:
            from langchain_text_splitters import CharacterTextSplitter
        except ImportError as exc:
            raise ImportError("Install ragkit[postgres] to split documents") from exc
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return splitter.split_documents(documents)
