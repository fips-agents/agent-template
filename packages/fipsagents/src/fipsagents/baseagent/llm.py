"""LLM client for BaseAgent — async wrappers around litellm.

Provides four calling patterns:

- ``call_model`` — standard chat completion, returns string
- ``call_model_json`` — structured output via Pydantic or dict schema
- ``call_model_stream`` — async iterator of content chunks
- ``call_model_validated`` — call + validate + retry with backoff

All methods are async.  All LLM communication goes through litellm so
that switching providers is a configuration change, not a code change.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Callable, TypeVar

import litellm
from pydantic import BaseModel

from fipsagents.baseagent.config import LLMConfig

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """Raised when an LLM call fails.

    Wraps the underlying litellm/provider exception so callers only need
    to catch one type.
    """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _schema_to_response_format(
    schema: type[BaseModel] | dict[str, Any],
) -> dict[str, Any]:
    """Convert a Pydantic model or raw JSON schema dict into the
    ``response_format`` value expected by litellm for structured output.
    """
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        json_schema = schema.model_json_schema()
        name = schema.__name__
    elif isinstance(schema, dict):
        json_schema = schema
        name = schema.get("title", "response")
    else:
        raise LLMError(
            f"schema must be a Pydantic model class or a JSON-schema dict, "
            f"got {type(schema).__name__}"
        )
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": json_schema,
        },
    }


def _parse_json_response(
    content: str,
    schema: type[BaseModel] | dict[str, Any],
) -> BaseModel | dict[str, Any]:
    """Parse a JSON string into the target type.

    Returns a Pydantic model instance when *schema* is a model class,
    otherwise a plain dict.
    """
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMError(f"Model returned invalid JSON: {exc}") from exc

    if isinstance(schema, type) and issubclass(schema, BaseModel):
        try:
            return schema.model_validate(data)
        except Exception as exc:
            raise LLMError(
                f"Model output failed schema validation: {exc}"
            ) from exc

    return data


# ---------------------------------------------------------------------------
# Response wrapper
# ---------------------------------------------------------------------------


class ModelResponse:
    """Thin wrapper around a litellm response for convenient access.

    Attributes
    ----------
    content:
        The text content of the first choice, or ``None`` if the model
        returned only tool calls.
    tool_calls:
        List of tool-call dicts from the response, or ``None``.
    raw:
        The full litellm ``ModelResponse`` object for advanced use.
    """

    __slots__ = ("content", "tool_calls", "raw")

    def __init__(self, raw: Any) -> None:
        self.raw = raw
        message = raw.choices[0].message
        # litellm responses expose content via attribute access.
        self.content: str | None = getattr(message, "content", None) or None
        # Normalise tool_calls — litellm may return a list or None.
        tc = getattr(message, "tool_calls", None)
        self.tool_calls: list[Any] | None = list(tc) if tc else None

    def __str__(self) -> str:
        return self.content or ""


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------


class LLMClient:
    """Async LLM client backed by litellm.

    Parameters
    ----------
    config:
        An ``LLMConfig`` instance (from ``agent.yaml``).  Provides model
        name, endpoint URL, temperature, and max_tokens.
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    # -- internal helpers ---------------------------------------------------

    def _base_kwargs(self, **overrides: Any) -> dict[str, Any]:
        """Build the kwargs dict that every litellm call starts from."""
        kwargs: dict[str, Any] = {
            "model": self._config.name,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
        }
        if self._config.endpoint:
            kwargs["api_base"] = self._config.endpoint
        kwargs.update(overrides)
        return kwargs

    async def _acompletion(self, **kwargs: Any) -> Any:
        """Call ``litellm.acompletion`` and translate exceptions."""
        try:
            return await litellm.acompletion(**kwargs)
        except Exception as exc:
            raise LLMError(
                f"LLM call failed ({type(exc).__name__}): {exc}"
            ) from exc

    # -- public API ---------------------------------------------------------

    async def call_model(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Standard chat completion.

        Parameters
        ----------
        messages:
            OpenAI-format message list.
        tools:
            Optional list of tool schemas for function calling.
        **kwargs:
            Extra keyword arguments forwarded to ``litellm.acompletion``.

        Returns
        -------
        ModelResponse:
            Wrapper with ``.content``, ``.tool_calls``, and ``.raw``.
        """
        call_kwargs = self._base_kwargs(**kwargs)
        call_kwargs["messages"] = messages
        if tools is not None:
            call_kwargs["tools"] = tools
        raw = await self._acompletion(**call_kwargs)
        return ModelResponse(raw)

    async def call_model_json(
        self,
        messages: list[dict[str, Any]],
        schema: type[BaseModel] | dict[str, Any],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> BaseModel | dict[str, Any]:
        """Structured-output completion.

        Requests JSON conforming to *schema* and returns a parsed object.
        When *schema* is a Pydantic model class the return value is an
        instance of that class.  When it is a raw JSON-schema dict the
        return value is a plain dict.

        Parameters
        ----------
        messages:
            OpenAI-format message list.
        schema:
            A Pydantic model class **or** a JSON-schema dict.
        tools:
            Optional tool schemas for function calling.
        **kwargs:
            Extra keyword arguments forwarded to ``litellm.acompletion``.
        """
        response_format = _schema_to_response_format(schema)
        call_kwargs = self._base_kwargs(**kwargs)
        call_kwargs["messages"] = messages
        call_kwargs["response_format"] = response_format
        if tools is not None:
            call_kwargs["tools"] = tools
        raw = await self._acompletion(**call_kwargs)
        content = raw.choices[0].message.content
        if content is None:
            raise LLMError(
                "Model returned no content in structured-output mode"
            )
        return _parse_json_response(content, schema)

    async def call_model_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Streaming chat completion (content-only).

        Yields content-delta strings as they arrive from the provider.
        Discards reasoning, tool calls, and other delta fields. Use
        ``call_model_stream_raw`` if you need the full chunk.

        Parameters
        ----------
        messages:
            OpenAI-format message list.
        tools:
            Optional tool schemas for function calling.
        **kwargs:
            Extra keyword arguments forwarded to ``litellm.acompletion``.
        """
        async for chunk in self.call_model_stream_raw(
            messages, tools=tools, **kwargs
        ):
            try:
                delta = chunk.choices[0].delta
            except (AttributeError, IndexError):
                continue
            content = getattr(delta, "content", None)
            if content:
                yield content

    async def call_model_stream_raw(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Streaming chat completion (raw chunks).

        Yields the full litellm chunk for each delta. Callers can
        inspect ``chunk.choices[0].delta`` for ``content``, ``role``,
        ``tool_calls``, ``reasoning_content``, and other provider fields.
        Used by ``BaseAgent.astep_stream`` to drive rich streaming with
        thinking, tool execution, and response phases preserved.

        Parameters
        ----------
        messages:
            OpenAI-format message list.
        tools:
            Optional tool schemas for function calling.
        **kwargs:
            Extra keyword arguments forwarded to ``litellm.acompletion``.
        """
        call_kwargs = self._base_kwargs(**kwargs)
        call_kwargs["messages"] = messages
        call_kwargs["stream"] = True
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

    async def call_model_validated(
        self,
        messages: list[dict[str, Any]],
        validator_fn: Callable[[ModelResponse], T],
        *,
        max_retries: int = 3,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> T:
        """Call the model, pass the result to a validator, and retry on failure.

        Calls ``call_model`` and feeds the ``ModelResponse`` to
        *validator_fn*.  If the validator raises, the call is retried with
        exponential backoff until *max_retries* attempts are exhausted.

        Parameters
        ----------
        messages:
            OpenAI-format message list.
        validator_fn:
            A callable that receives the ``ModelResponse`` and returns the
            validated result.  Should raise any exception to signal that
            the response is invalid and the call should be retried.
        max_retries:
            Maximum number of retry attempts after the initial call.
            Defaults to 3 (so up to 4 total attempts).
        tools:
            Optional tool schemas for function calling.
        **kwargs:
            Extra keyword arguments forwarded to ``call_model``.

        Returns
        -------
        T:
            Whatever ``validator_fn`` returns on success.

        Raises
        ------
        LLMError:
            If all attempts are exhausted without a valid response.
        """
        last_error: Exception | None = None
        total_attempts = 1 + max_retries

        for attempt in range(total_attempts):
            response = await self.call_model(
                messages, tools=tools, **kwargs
            )
            try:
                return validator_fn(response)
            except Exception as exc:
                last_error = exc
                if attempt < total_attempts - 1:
                    delay = (2 ** attempt) * 1.0  # 1s, 2s, 4s, ...
                    logger.warning(
                        "Validation failed (attempt %d/%d): %s — "
                        "retrying in %.1fs",
                        attempt + 1,
                        total_attempts,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)

        raise LLMError(
            f"Validation failed after {total_attempts} attempts. "
            f"Last error: {last_error}"
        ) from last_error
