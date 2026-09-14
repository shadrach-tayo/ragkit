"""Optional Braintrust tracing. No-op unless ``ragkit[tracing]`` is installed."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import cache
from typing import Any, TypeVar

from ragkit.request_id import get_request_id

F = TypeVar("F", bound=Callable[..., Any])


def traced(*, name: str) -> Callable[[F], F]:
    """Decorate a function with Braintrust when the extra is installed."""
    try:
        from braintrust import traced as braintrust_traced
    except ImportError:
        def decorator(fn: F) -> F:
            return fn

        return decorator
    return braintrust_traced(name=name)


@cache
def setup_braintrust() -> None:
    """Start a Braintrust logger. Skip when no API key or extra is missing."""
    try:
        import braintrust
    except ImportError:
        return
    from ragkit.config import get_settings

    settings = get_settings()
    if not settings.braintrust_api_key:
        return
    braintrust.init_logger(
        api_key=settings.braintrust_api_key,
        project=settings.braintrust_project,
        project_id=settings.braintrust_project_id,
        async_flush=False,
    )


def flush_braintrust() -> None:
    """Send buffered spans to Braintrust when the extra is installed."""
    try:
        import braintrust
    except ImportError:
        return
    braintrust.flush()  # type: ignore[no-untyped-call]


def _span_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Copy kwargs and stamp the current request id onto span metadata."""
    request_id = get_request_id()
    if not request_id:
        return kwargs
    metadata = dict(kwargs.get("metadata") or {})
    metadata.setdefault("request_id", request_id)
    return {**kwargs, "metadata": metadata}


@contextmanager
def trace_span(name: str, **kwargs: Any) -> Iterator[Any]:
    """Open a parent span, or yield None when Braintrust is not installed."""
    try:
        import braintrust
    except ImportError:
        yield None
        return
    with braintrust.start_span(name=name, **_span_kwargs(kwargs)) as span:
        yield span


def instrument_providers() -> None:
    """Patch LLM providers. Call before constructing chat clients if desired."""
    try:
        import braintrust
    except ImportError:
        return
    braintrust.auto_instrument()
