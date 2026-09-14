"""Normalized retrieval output used by apps and generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalResult:
    """Ranked chunks plus optional Elasticsearch hits and rerank rows."""

    docs: list[str] = field(default_factory=list)
    metadata: list[dict[str, Any]] = field(default_factory=list)
    rerank: list[dict[str, Any]] | None = None
    es_hits: list[dict[str, Any]] = field(default_factory=list)
    total: Any = None
    strategy: str = "vector"

    def as_vector_payload(self) -> dict[str, Any]:
        """Return the shape expected by existing vector_rag callers."""
        return {
            "docs": self.docs,
            "original_docs": self.metadata,
            "rerank": self.rerank,
        }

    def as_es_payload(self) -> dict[str, Any]:
        """Return the shape expected by existing search callers."""
        return {"results": self.es_hits, "total": self.total}
