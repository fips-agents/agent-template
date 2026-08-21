"""OpenAI provider — direct ``AsyncOpenAI`` client.

Works with any OpenAI-compatible endpoint: OpenAI itself, vLLM,
LlamaStack, llm-d, and the adapter sidecar.
"""

from __future__ import annotations

import logging
import os
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from fipsagents.baseagent.config import LLMConfig
from fipsagents.baseagent.llm import LLMError
from fipsagents.baseagent.providers._base import LLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """Provider backed by the OpenAI Python SDK.

    Parameters
    ----------
    config:
        ``LLMConfig`` from ``agent.yaml``.
    """

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._client = AsyncOpenAI(
            base_url=config.endpoint or None,
            api_key=os.environ.get("OPENAI_API_KEY", "not-required"),
        )

    def _build_kwargs(self, **overrides: Any) -> dict[str, Any]:
        model_name = self._config.name
        if "/" in model_name:
            prefix = model_name.split("/", 1)[0].lower()
            if prefix in ("openai", "vllm", "llamastack"):
                model_name = model_name.split("/", 1)[1]
        kwargs: dict[str, Any] = {
            "model": model_name,
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
            return await self._client.chat.completions.create(**call_kwargs)
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
            response = await self._client.chat.completions.create(**call_kwargs)
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
