"""Embedding factories. Nothing is constructed at import time."""

from __future__ import annotations

import logging

from langchain_core.embeddings import Embeddings

from ragkit.config import get_settings

logger = logging.getLogger(__name__)


def voyage_embeddings(*, model: str = "voyage-3.5", api_key: str | None = None) -> Embeddings:
    """Build a Voyage embedding client. Requires ``ragkit[voyage]``."""
    try:
        from langchain_voyageai import VoyageAIEmbeddings
        from pydantic import SecretStr
    except ImportError as exc:
        raise ImportError("Install ragkit[voyage] to use Voyage embeddings") from exc

    key = api_key if api_key is not None else get_settings().voyage_api_key or ""
    return VoyageAIEmbeddings(api_key=SecretStr(key), model=model)


def preload_voyage_tokenizer(embeddings: Embeddings | None = None) -> None:
    """Download and cache the Voyage tokenizer so later embeds skip Hugging Face I/O."""
    model = embeddings or voyage_embeddings()
    try:
        client = getattr(model, "_client", None)
        name = getattr(model, "model", None)
        if client is None or name is None:
            return
        client.tokenizer(name)
    except Exception:  # noqa: BLE001
        logger.warning("Could not preload Voyage tokenizer", exc_info=True)
