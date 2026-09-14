"""Process-wide request id, readable from API handlers and RAG code."""

from __future__ import annotations

from contextvars import ContextVar, Token

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def get_request_id() -> str | None:
    """Return the id bound to the current request, if any."""
    return _request_id.get()


def set_request_id(request_id: str) -> Token:
    """Bind a request id for this task and return a reset token."""
    return _request_id.set(request_id)


def reset_request_id(token: Token) -> None:
    """Restore the previous request id after a request finishes."""
    _request_id.reset(token)
