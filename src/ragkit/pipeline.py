"""Domain-agnostic RAG pipeline with configurable retrieval and generation."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pydantic import SecretStr

from ragkit.config import RagConfig, RetrievalStrategy, get_settings
from ragkit.embeddings import voyage_embeddings
from ragkit.result import RetrievalResult
from ragkit.tracing import traced

logger = logging.getLogger(__name__)


class RagPipeline:
    """Retrieve and optionally generate answers from an index.

    The pipeline is corpus-agnostic. Pass documents as LangChain ``Document``
    objects and configure embeddings, chunking, strategy, and reranking.
    """

    def __init__(
        self,
        config: RagConfig | None = None,
        *,
        embeddings: Embeddings | None = None,
        search_client: Any | None = None,
    ):
        """Store config and optional injected clients."""
        self.config = config or RagConfig()
        self._embeddings = embeddings
        self._search_client = search_client
        self._stores: dict[str, Any] = {}
        self._cohere: Any | None = None
        self._llm: Any | None = None

    @property
    def embeddings(self) -> Embeddings:
        """Return the vector-store embedding model."""
        if self._embeddings is None:
            self._embeddings = voyage_embeddings(model=self.config.embedding_model)
        return self._embeddings

    @property
    def search_client(self) -> Any:
        """Return the Elasticsearch client used for hybrid retrieval."""
        if self._search_client is None:
            from ragkit.elasticsearch.search import Search

            self._search_client = Search(
                self.config.chunk_size,
                self.config.embedding_dim,
                embeddings=self.embeddings,
                chunk_overlap=self.config.chunk_overlap,
            )
        return self._search_client

    @traced(name="rag.retrieve")
    def retrieve(
        self,
        question: str,
        *,
        index_name: str | None = None,
        strategy: RetrievalStrategy | None = None,
        top_k: int | None = None,
        from_: int = 0,
        rerank: bool | None = None,
        rerank_top_n: int | None = None,
    ) -> RetrievalResult:
        """Fetch ranked chunks for a question using the selected strategy."""
        index = index_name or self.config.index_name
        chosen = strategy or self.config.strategy
        k = top_k or self.config.top_k
        should_rerank = self.config.rerank if rerank is None else rerank
        previous_top_n = self.config.rerank_top_n
        if rerank_top_n is not None:
            self.config.rerank_top_n = rerank_top_n
        try:
            result = self._retrieve_by_strategy(question, index, k, from_, chosen)
            if should_rerank and result.docs:
                result = self._apply_rerank(result, question)
            return result
        finally:
            self.config.rerank_top_n = previous_top_n

    @traced(name="rag.aretrieve")
    async def aretrieve(
        self,
        question: str,
        *,
        index_name: str | None = None,
        strategy: RetrievalStrategy | None = None,
        top_k: int | None = None,
        from_: int = 0,
        rerank: bool | None = None,
        rerank_top_n: int | None = None,
    ) -> RetrievalResult:
        """Fetch ranked chunks without blocking the caller’s event loop on sync drivers."""
        index = index_name or self.config.index_name
        chosen = strategy or self.config.strategy
        k = top_k or self.config.top_k
        should_rerank = self.config.rerank if rerank is None else rerank
        previous_top_n = self.config.rerank_top_n
        if rerank_top_n is not None:
            self.config.rerank_top_n = rerank_top_n
        try:
            if chosen == "vector":
                result = await self._aretrieve_vector(question, index, k)
            elif chosen == "hybrid":
                result = await asyncio.to_thread(
                    self._retrieve_hybrid, question, index, k, from_
                )
            elif chosen == "ensemble":
                result = await self._aretrieve_ensemble(question, index, k, from_)
            else:
                raise ValueError(f"Unknown retrieval strategy: {chosen}")
            if should_rerank and result.docs:
                result = await asyncio.to_thread(self._apply_rerank, result, question)
            return result
        finally:
            self.config.rerank_top_n = previous_top_n

    def warmup(self, index_name: str | None = None) -> None:
        """Open the vector store so later retrieves skip engine setup."""
        self._vector_store(index_name or self.config.index_name)

    @traced(name="rag.generate")
    def generate(
        self,
        question: str,
        *,
        index_name: str | None = None,
        strategy: RetrievalStrategy | None = None,
        top_k: int | None = None,
        rerank: bool | None = None,
    ) -> dict[str, Any]:
        """Answer a question from retrieved context."""
        result = self.retrieve(
            question,
            index_name=index_name,
            strategy=strategy or self.config.strategy,
            top_k=top_k,
            rerank=rerank,
        )
        retrieval_context = _dedupe_texts(result.docs)
        context = "\n\n".join(retrieval_context)
        llm = self._get_llm()
        instructions = f"""{self.config.system_prompt}

        <context>
        {context}
        </context>"""
        response = llm.invoke(
            [
                {"role": "system", "content": instructions},
                {"role": "user", "content": question},
            ]
        )
        return {
            "question": question,
            "content": response.content,
            "retrieval_context": retrieval_context,
            "docs": result.metadata,
            "reranks": result.rerank,
            "es_docs": [
                {
                    "source": (hit.get("_source") or {}).get("source"),
                    "page": (hit.get("_source") or {}).get("page"),
                    "score": hit.get("_score"),
                }
                for hit in result.es_hits
            ],
            "strategy": result.strategy,
        }

    def ingest(
        self,
        documents: list[Document],
        *,
        index_name: str | None = None,
        targets: tuple[str, ...] = ("vector", "hybrid"),
    ) -> None:
        """Split and write documents into the configured stores."""
        index = index_name or self.config.index_name
        if "vector" in targets:
            from ragkit.vector_store.postgres import PostgresVectorStoreManager

            store = PostgresVectorStoreManager(
                index,
                embedding_dim=self.config.embedding_dim,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                embeddings=self.embeddings,
                overwrite_existing=True,
            )
            store.add_documents(documents)
            self._stores[index] = store
        if "hybrid" in targets:
            client = self.search_client
            client.create_index(index)
            client.insert_documents(index, documents)

    def _retrieve_by_strategy(
        self,
        question: str,
        index: str,
        top_k: int,
        from_: int,
        chosen: RetrievalStrategy,
    ) -> RetrievalResult:
        if chosen == "vector":
            return self._retrieve_vector(question, index, top_k)
        if chosen == "hybrid":
            return self._retrieve_hybrid(question, index, top_k, from_)
        if chosen == "ensemble":
            return self._retrieve_ensemble(question, index, top_k, from_)
        raise ValueError(f"Unknown retrieval strategy: {chosen}")

    def _vector_store(self, index_name: str) -> Any:
        """Return a cached Postgres/pgvector store for ``index_name``."""
        from ragkit.vector_store.postgres import PostgresVectorStoreManager

        store = self._stores.get(index_name)
        if store is None:
            store = PostgresVectorStoreManager(
                index_name,
                embedding_dim=self.config.embedding_dim,
                embeddings=self.embeddings,
            )
            self._stores[index_name] = store
        return store

    def _retrieve_vector(
        self, question: str, index_name: str, top_k: int
    ) -> RetrievalResult:
        """Retrieve the top-k dense neighbors from Postgres/pgvector."""
        retriever = self._vector_store(index_name).as_retriever(top_k)
        retrieval = retriever.invoke(question)
        docs = [doc.page_content for doc in retrieval]
        metadata = [dict(doc.metadata) for doc in retrieval]
        return RetrievalResult(
            docs=docs,
            metadata=metadata,
            strategy="vector",
            total=len(docs),
        )

    async def _aretrieve_vector(
        self, question: str, index_name: str, top_k: int
    ) -> RetrievalResult:
        """Async variant of dense vector retrieval."""
        retriever = self._vector_store(index_name).as_retriever(top_k)
        retrieval = await retriever.ainvoke(question)
        docs = [doc.page_content for doc in retrieval]
        metadata = [dict(doc.metadata) for doc in retrieval]
        return RetrievalResult(
            docs=docs,
            metadata=metadata,
            strategy="vector",
            total=len(docs),
        )

    def _retrieve_hybrid(
        self, question: str, index_name: str, top_k: int, from_: int
    ) -> RetrievalResult:
        """Retrieve BM25 + Voyage kNN hits fused with reciprocal rank fusion."""
        payload = self.search_client.hybrid_search(
            index_name=index_name,
            question=question,
            size=top_k,
            from_=from_,
            rank_constant=self.config.rrf_rank_constant,
        )
        hits = list(payload["hits"]["hits"])
        docs = [_es_hit_content(hit) for hit in hits]
        metadata = [
            {
                "source": (hit.get("_source") or {}).get("source"),
                "page": (hit.get("_source") or {}).get("page"),
            }
            for hit in hits
        ]
        return RetrievalResult(
            docs=docs,
            metadata=metadata,
            es_hits=hits,
            total=payload["hits"]["total"],
            strategy="hybrid",
        )

    def _retrieve_ensemble(
        self, question: str, index_name: str, top_k: int, from_: int
    ) -> RetrievalResult:
        """Concatenate vector and hybrid hits, dropping near-duplicate texts."""
        vector = self._retrieve_vector(question, index_name, top_k)
        try:
            hybrid = self._retrieve_hybrid(question, index_name, top_k, from_)
        except Exception:  # noqa: BLE001
            logger.exception("Hybrid retrieval failed; using vector results only")
            return vector
        docs = _dedupe_texts([*vector.docs, *hybrid.docs])
        metadata = [*vector.metadata, *hybrid.metadata]
        return RetrievalResult(
            docs=docs,
            metadata=metadata,
            rerank=vector.rerank,
            es_hits=hybrid.es_hits,
            total=hybrid.total,
            strategy="ensemble",
        )

    async def _aretrieve_ensemble(
        self, question: str, index_name: str, top_k: int, from_: int
    ) -> RetrievalResult:
        """Async ensemble retrieval; falls back to vector if hybrid fails."""
        vector = await self._aretrieve_vector(question, index_name, top_k)
        try:
            hybrid = await asyncio.to_thread(
                self._retrieve_hybrid, question, index_name, top_k, from_
            )
        except Exception:  # noqa: BLE001
            logger.exception("Hybrid retrieval failed; using vector results only")
            return vector
        docs = _dedupe_texts([*vector.docs, *hybrid.docs])
        metadata = [*vector.metadata, *hybrid.metadata]
        return RetrievalResult(
            docs=docs,
            metadata=metadata,
            rerank=vector.rerank,
            es_hits=hybrid.es_hits,
            total=hybrid.total,
            strategy="ensemble",
        )

    def rerank_documents(
        self,
        docs: list[str],
        question: str,
        *,
        top_n: int | None = None,
    ) -> RetrievalResult:
        """Rerank an arbitrary document list with the configured reranker."""
        previous = self.config.rerank_top_n
        if top_n is not None:
            self.config.rerank_top_n = top_n
        try:
            return self._apply_rerank(RetrievalResult(docs=docs), question)
        finally:
            self.config.rerank_top_n = previous

    def _apply_rerank(self, result: RetrievalResult, question: str) -> RetrievalResult:
        """Reorder docs and metadata together with the configured Cohere model."""
        client = self._get_cohere()
        response = client.rerank(
            model=self.config.rerank_model,
            query=question,
            documents=result.docs,
            top_n=min(self.config.rerank_top_n, len(result.docs)),
        )
        ranked_docs = [result.docs[item.index] for item in response.results]
        ranked_meta = [
            result.metadata[item.index] if item.index < len(result.metadata) else {}
            for item in response.results
        ]
        return RetrievalResult(
            docs=ranked_docs,
            metadata=ranked_meta,
            rerank=[item.dict() for item in response.results],
            es_hits=result.es_hits,
            total=result.total,
            strategy=result.strategy,
        )

    def _get_cohere(self) -> Any:
        """Lazily construct the Cohere client used for reranking."""
        if self._cohere is None:
            try:
                import cohere
            except ImportError as exc:
                raise ImportError("Install ragkit[rerank] to rerank results") from exc
            self._cohere = cohere.ClientV2(api_key=get_settings().cohere_api_key)
        return self._cohere

    def _get_llm(self) -> Any:
        """Lazily construct the chat model used for ``generate``."""
        if self._llm is None:
            try:
                from langchain_openai import ChatOpenAI
            except ImportError as exc:
                raise ImportError("Install ragkit[openai] to call generate()") from exc
            kwargs: dict[str, Any] = {
                "model": self.config.llm_model,
                "temperature": self.config.llm_temperature,
            }
            if self.config.llm_base_url:
                kwargs["base_url"] = self.config.llm_base_url
            api_key = self.config.llm_api_key or get_settings().openai_api_key
            if api_key:
                kwargs["api_key"] = SecretStr(api_key)
            self._llm = ChatOpenAI(**kwargs)
        return self._llm


def _es_hit_content(hit: dict[str, Any]) -> str:
    """Return the document text stored on an Elasticsearch hit."""
    return (hit.get("_source") or {}).get("content") or ""


def _dedupe_texts(texts: list[str]) -> list[str]:
    """Drop near-duplicate passages while preserving first-seen order."""
    seen: set[str] = set()
    unique: list[str] = []
    for text in texts:
        key = " ".join(text.split())[:240]
        if not text or key in seen:
            continue
        seen.add(key)
        unique.append(text)
    return unique
