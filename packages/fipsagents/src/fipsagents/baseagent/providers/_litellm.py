"""LiteLLM provider — 100+ LLM providers via one dependency.

LiteLLM returns OpenAI-compatible ``ModelResponse`` objects, so no
chunk normalization is needed.  Model names use LiteLLM's
``provider/model`` convention (e.g. ``anthropic/claude-sonnet-4-20250514``,
``bedrock/anthropic.claude-3-sonnet``).
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

try:
    import litellm
except ImportError as _exc:
    raise ImportError(
        "The litellm provider requires the litellm package. "
        "Install it: pip install fipsagents[litellm]"
    ) from _exc

from fipsagents.baseagent.config import LLMConfig
from fipsagents.baseagent.llm import LLMError
from fipsagents.baseagent.providers._base import LLMProvider

logger = logging.getLogger(__name__)

litellm.suppress_debug_info = True


class LiteLLMProvider(LLMProvider):
    """Provider backed by ``litellm.acompletion``.

    LiteLLM handles provider routing via model-name prefixes
    (``anthropic/…``, ``bedrock/…``, ``azure/…``, etc.) and reads
    provider-specific API keys from environment variables automatically.
    """

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        if config.endpoint:
            litellm.api_base = config.endpoint

    def _build_kwargs(self, **overrides: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._config.name,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
        }
        kwargs.update(overrides)
        return kwargs

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        call_kwargs = self._build_kwargs(**kwargs)
        call_kwargs["messages"] = messages
        if tools is not None:
            call_kwargs["tools"] = tools
        try:
            return await litellm.acompletion(**call_kwargs)
        except Exception as exc:
            raise LLMError(
                f"LLM call failed ({type(exc).__name__}): {exc}"
            ) from exc

    async def stream_raw(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        call_kwargs = self._build_kwargs(**kwargs)
        call_kwargs["messages"] = messages
        call_kwargs["stream"] = True
        call_kwargs.setdefault("stream_options", {"include_usage": True})
        if tools is not None:
            call_kwargs["tools"] = tools
        try:
            response = await litellm.acompletion(**call_kwargs)
        except Exception as exc:
            raise LLMError(
                f"LLM streaming call failed ({type(exc).__name__}): {exc}"
            ) from exc
        try:
            async for chunk in response:
                yield chunk
        except Exception as exc:
            raise LLMError(
                f"Error during streaming iteration ({type(exc).__name__}): {exc}"
            ) from exc
