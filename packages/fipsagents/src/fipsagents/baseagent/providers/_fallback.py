"""Fallback provider — try providers in order on retriable failures.

Wraps a primary provider and a list of fallbacks. On connection errors,
timeouts, or HTTP 5xx responses the next provider in the chain is tried.
Non-retriable errors (4xx) raise immediately.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from fipsagents.baseagent.llm import LLMError
from fipsagents.baseagent.providers._base import LLMProvider

logger = logging.getLogger(__name__)


def _is_retriable(exc: LLMError) -> bool:
    """Decide whether a failed LLM call should trigger a fallback.

    Retriable: connection errors, timeouts, HTTP 5xx.
    Non-retriable: HTTP 4xx (bad request, auth, not found).
    """
    cause = exc.__cause__
    if cause is None:
        return True

    cls_name = type(cause).__name__

    if "ConnectionError" in cls_name or "TimeoutError" in cls_name:
        return True

    status = getattr(cause, "status_code", None)
    if isinstance(status, int):
        return status >= 500

    if "Connection" in cls_name or "Timeout" in cls_name:
        return True

    return True


class FallbackProvider(LLMProvider):
    """Provider that tries a chain of backends in order.

    Parameters
    ----------
    primary:
        The preferred provider.
    fallbacks:
        Providers to try in order if the primary (or earlier fallback)
        fails with a retriable error.
    """

    def __init__(
        self,
        primary: LLMProvider,
        fallbacks: list[LLMProvider],
    ) -> None:
        self._primary = primary
        self._fallbacks = fallbacks
        self._chain = [primary, *fallbacks]

    @property
    def provider_name(self) -> str:
        return f"{self._primary.provider_name}+{len(self._fallbacks)} fallback(s)"

    def _build_kwargs(self, **overrides: Any) -> dict[str, Any]:
        return self._primary._build_kwargs(**overrides)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        last_error: LLMError | None = None
        for i, provider in enumerate(self._chain):
            try:
                result = await provider.complete(
                    messages, tools=tools, **kwargs
                )
                if i > 0:
                    logger.info(
                        "Fallback succeeded — served by %s (attempt %d/%d)",
                        provider.provider_name,
                        i + 1,
                        len(self._chain),
                    )
                return result
            except LLMError as exc:
                last_error = exc
                if not _is_retriable(exc):
                    raise
                remaining = len(self._chain) - i - 1
                if remaining > 0:
                    logger.warning(
                        "Provider %s failed (%s) — trying next fallback "
                        "(%d remaining)",
                        provider.provider_name,
                        exc,
                        remaining,
                    )
                else:
                    logger.error(
                        "All providers exhausted — last failure from %s: %s",
                        provider.provider_name,
                        exc,
                    )
        raise last_error  # type: ignore[misc]

    async def stream_raw(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        last_error: LLMError | None = None
        for i, provider in enumerate(self._chain):
            try:
                stream = provider.stream_raw(
                    messages, tools=tools, **kwargs
                )
                first_chunk = await stream.__anext__()
            except (LLMError, StopAsyncIteration) as exc:
                if isinstance(exc, StopAsyncIteration):
                    exc = LLMError("Provider returned empty stream")
                last_error = exc  # type: ignore[assignment]
                if not _is_retriable(exc):  # type: ignore[arg-type]
                    raise exc
                remaining = len(self._chain) - i - 1
                if remaining > 0:
                    logger.warning(
                        "Provider %s failed during stream start (%s) — "
                        "trying next fallback (%d remaining)",
                        provider.provider_name,
                        exc,
                        remaining,
                    )
                    continue
                else:
                    logger.error(
                        "All providers exhausted — last failure from %s: %s",
                        provider.provider_name,
                        exc,
                    )
                    raise exc
            else:
                if i > 0:
                    logger.info(
                        "Fallback stream succeeded — served by %s "
                        "(attempt %d/%d)",
                        provider.provider_name,
                        i + 1,
                        len(self._chain),
                    )
                yield first_chunk
                async for chunk in stream:
                    yield chunk
                return

        raise last_error  # type: ignore[misc]

    async def complete_json(
        self,
        messages: list[dict[str, Any]],
        schema: Any,
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        last_error: LLMError | None = None
        for i, provider in enumerate(self._chain):
            try:
                result = await provider.complete_json(
                    messages, schema, tools=tools, **kwargs
                )
                if i > 0:
                    logger.info(
                        "Fallback (JSON) succeeded — served by %s "
                        "(attempt %d/%d)",
                        provider.provider_name,
                        i + 1,
                        len(self._chain),
                    )
                return result
            except LLMError as exc:
                last_error = exc
                if not _is_retriable(exc):
                    raise
                remaining = len(self._chain) - i - 1
                if remaining > 0:
                    logger.warning(
                        "Provider %s failed (JSON) (%s) — trying next "
                        "fallback (%d remaining)",
                        provider.provider_name,
                        exc,
                        remaining,
                    )
        raise last_error  # type: ignore[misc]
