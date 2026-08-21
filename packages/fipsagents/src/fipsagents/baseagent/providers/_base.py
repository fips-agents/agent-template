"""Abstract base class for LLM provider backends.

Each provider normalizes its output to OpenAI-compatible shapes so that
``astep_stream()`` in ``agent.py`` works unchanged regardless of which
backend is active.  The three concrete implementations — OpenAI, LiteLLM,
Anthropic — live in sibling modules.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from pydantic import BaseModel

from fipsagents.baseagent.config import LLMConfig

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Backend-specific LLM communication.

    Subclasses must implement :meth:`complete` and :meth:`stream_raw`.
    Both return shapes compatible with the OpenAI Python SDK so that
    ``LLMClient`` and ``astep_stream()`` need no per-provider branching.

    Parameters
    ----------
    config:
        The ``LLMConfig`` from ``agent.yaml``.
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    @property
    def provider_name(self) -> str:
        """Short identifier for logging (e.g. ``"openai"``, ``"litellm"``)."""
        return self._config.provider

    def _build_kwargs(self, **overrides: Any) -> dict[str, Any]:
        """Base kwargs shared by every call: model, temperature, max_tokens.

        Subclasses should override to apply provider-specific model-name
        normalization before calling ``super()._build_kwargs()``.
        """
        kwargs: dict[str, Any] = {
            "model": self._config.name,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
        }
        kwargs.update(overrides)
        return kwargs

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Non-streaming chat completion.

        Must return an object with ``choices[0].message.content`` and
        ``choices[0].message.tool_calls`` attributes — i.e. something
        ``ModelResponse(raw)`` can wrap.
        """

    @abstractmethod
    async def stream_raw(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Streaming chat completion yielding OpenAI-compatible chunks.

        Each chunk must support ``chunk.choices[0].delta`` with
        ``content``, ``reasoning_content``, ``tool_calls``, ``role``,
        and the choice-level ``finish_reason``.  The terminal chunk
        should carry ``chunk.usage`` when available.

        ``astep_stream()`` consumes these chunks directly.
        """

    async def complete_json(
        self,
        messages: list[dict[str, Any]],
        schema: type[BaseModel] | dict[str, Any],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Structured-output completion.

        Default implementation adds ``response_format`` to kwargs and
        delegates to :meth:`complete`.  Providers with native structured
        output can override.
        """
        from fipsagents.baseagent.llm import _schema_to_response_format

        kwargs["response_format"] = _schema_to_response_format(schema)
        return await self.complete(messages, tools=tools, **kwargs)
