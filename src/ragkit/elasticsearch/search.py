"""Elasticsearch BM25 + dense kNN search with reciprocal rank fusion."""

from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ragkit.config import get_settings
from ragkit.embeddings import voyage_embeddings

logger = logging.getLogger(__name__)

_EMBED_BATCH = 32


def reciprocal_rank_fuse(
    result_lists: list[list[dict[str, Any]]],
    *,
    rank_constant: int = 60,
) -> list[dict[str, Any]]:
    """Merge ranked hit lists with reciprocal rank fusion."""
    scores: dict[str, float] = {}
    docs: dict[str, dict[str, Any]] = {}
    for hits in result_lists:
        for rank, hit in enumerate(hits, start=1):
            doc_id = hit["_id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rank_constant + rank)
            docs[doc_id] = hit
    ranked_ids = sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)
    fused = []
    for doc_id in ranked_ids:
        hit = dict(docs[doc_id])
        hit["_score"] = scores[doc_id]
        fused.append(hit)
    return fused


class Search:
    """Index and query Elasticsearch with BM25 plus the shared dense embedder."""

    def __init__(
        self,
        chunk_size: int,
        embedding_dims: int,
        embeddings: Embeddings | None = None,
        chunk_overlap: int = 30,
        elasticsearch_url: str | None = None,
    ) -> None:
        """Connect to Elasticsearch and store the shared embedding model."""
        try:
            from elasticsearch import Elasticsearch
        except ImportError as exc:
            raise ImportError("Install ragkit[elasticsearch] for hybrid search") from exc

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_dims = embedding_dims
        self.embeddings = embeddings or voyage_embeddings()
        self.es = Elasticsearch(elasticsearch_url or get_settings().elasticsearch_url)
        logger.info("Connected to Elasticsearch %s", self.es.info().body.get("cluster_name"))

    def get_mapping(self, index: str) -> Any:
        """Return the Elasticsearch mapping for an index."""
        return self.es.indices.get_mapping(index=index)

    def create_index(self, index_name: str = "default-index") -> None:
        """Create or replace an index with BM25 text fields and dense vectors."""
        self.es.indices.delete(index=index_name, ignore_unavailable=True)
        self.es.indices.create(
            index=index_name,
            mappings={
                "properties": {
                    "content": {"type": "text"},
                    "source": {"type": "keyword"},
                    "page": {"type": "integer"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": self.embedding_dims,
                        "index": True,
                        "similarity": "cosine",
                    },
                }
            },
        )

    def get_embedding(self, text: str) -> list[float]:
        """Embed a query or document with the shared embedding model."""
        return self.embeddings.embed_query(text)

    def insert_document(self, index_name: str, document: list[Document]) -> None:
        """Insert new document to index."""
        splits = self._split_documents(document)
        for split, vector in zip(
            splits, self._embed_texts([s.page_content for s in splits]), strict=True
        ):
            self.es.index(index=index_name, body=self._to_es_doc(split, vector))

    def insert_documents(self, index_name: str, documents: list[Document]) -> Any:
        """Bulk-insert chunked documents with batched embeddings."""
        operations = []
        splits = self._split_documents(documents)
        vectors = self._embed_texts([document.page_content for document in splits])
        for document, vector in zip(splits, vectors, strict=True):
            operations.append({"index": {"_index": index_name}})
            operations.append(self._to_es_doc(document, vector))
        return self.es.bulk(operations=operations)

    def reindex(self, index_name: str, documents: list[Document]) -> Any:
        """Refresh an index."""
        self.create_index(index_name=index_name)
        return self.insert_documents(index_name=index_name, documents=documents)

    def search(self, index_name: str, **query_args: Any) -> Any:
        """Run search query on index."""
        return self.es.search(index=index_name, **query_args)

    def hybrid_search(
        self,
        index_name: str,
        question: str,
        *,
        size: int = 10,
        from_: int = 0,
        rank_constant: int = 60,
        window_size: int | None = None,
    ) -> dict[str, Any]:
        """Fuse BM25 and kNN rankings with reciprocal rank fusion."""
        window = max(window_size or 50, size)
        query_vector = self.get_embedding(question)
        bm25 = self.search(
            index_name=index_name,
            query={"multi_match": {"query": question, "fields": ["content"]}},
            size=window,
        )
        knn = self.search(
            index_name=index_name,
            knn={
                "field": "embedding",
                "query_vector": query_vector,
                "k": window,
                "num_candidates": max(50, window * 2),
            },
            size=window,
        )
        fused = reciprocal_rank_fuse(
            [bm25["hits"]["hits"], knn["hits"]["hits"]],
            rank_constant=rank_constant,
        )
        page = fused[from_ : from_ + size]
        return {
            "hits": {
                "hits": page,
                "total": {"value": len(fused), "relation": "eq"},
            }
        }

    def retrieve_document(self, index: str, id: str) -> Any:
        """Retrieve a document by id from an index."""
        return self.es.get(index=index, id=id)

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches, retrying Voyage/provider rate limits."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH):
            batch = texts[start : start + _EMBED_BATCH]
            last_exc: BaseException | None = None
            for attempt in range(4):
                try:
                    vectors.extend(self.embeddings.embed_documents(batch))
                    last_exc = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if "429" not in str(exc) and "rate" not in str(exc).lower():
                        raise
                    time.sleep(2**attempt)
            if last_exc is not None:
                raise last_exc
        return vectors

    def _to_es_doc(self, doc: Document, vector: list[float]) -> dict[str, Any]:
        return {
            "content": doc.page_content,
            "embedding": vector,
            **doc.metadata,
        }

    def _split_documents(self, documents: list[Document]) -> list[Document]:
        try:
            from langchain_text_splitters import CharacterTextSplitter
        except ImportError as exc:
            raise ImportError("Install ragkit[elasticsearch] to split documents") from exc
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return splitter.split_documents(documents)
