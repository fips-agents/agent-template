"""Anthropic provider — direct ``AsyncAnthropic`` client.

Enables Anthropic-specific features that LiteLLM cannot pass through:
extended thinking (``reasoning_content``), prompt caching
(``cache_control``), and native system-message handling.

Normalizes Anthropic streaming events into OpenAI-compatible chunk
objects so ``astep_stream()`` works unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from types import SimpleNamespace
from typing import Any, AsyncIterator

try:
    import anthropic
except ImportError as _exc:
    raise ImportError(
        "The anthropic provider requires the anthropic SDK. "
        "Install it: pip install fipsagents[anthropic]"
    ) from _exc

from fipsagents.baseagent.config import LLMConfig
from fipsagents.baseagent.llm import LLMError
from fipsagents.baseagent.providers._base import LLMProvider

logger = logging.getLogger(__name__)

_ANTHROPIC_TO_OPENAI_STOP: dict[str, str] = {
    "end_turn": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "stop_sequence": "stop",
}


def _make_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


# ---------------------------------------------------------------------------
# Message translation: OpenAI dict format → Anthropic
# ---------------------------------------------------------------------------


def _build_anthropic_messages(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Convert OpenAI-format message dicts to ``(system, anthropic_msgs)``."""
    system_parts: list[str] = []
    result: list[dict[str, Any]] = []
    tool_result_buffer: list[dict[str, Any]] = []

    def _flush_tool_results() -> None:
        if tool_result_buffer:
            result.append({"role": "user", "content": list(tool_result_buffer)})
            tool_result_buffer.clear()

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content")

        if role == "system":
            if content:
                system_parts.append(
                    content if isinstance(content, str) else str(content)
                )
            continue

        if role == "tool":
            tool_result_buffer.append({
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": content or "",
            })
            continue

        _flush_tool_results()

        if role == "assistant":
            content_blocks: list[dict[str, Any]] = []
            if content and isinstance(content, str) and content.strip():
                content_blocks.append({"type": "text", "text": content})
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    try:
                        tool_input = json.loads(fn.get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        tool_input = {}
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "input": tool_input,
                    })
            if content_blocks:
                result.append({"role": "assistant", "content": content_blocks})
        else:
            blocks: list[dict[str, Any]]
            if isinstance(content, str):
                blocks = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                blocks = content
            else:
                blocks = [{"type": "text", "text": str(content or "")}]
            result.append({"role": "user", "content": blocks})

    _flush_tool_results()

    merged: list[dict[str, Any]] = []
    for entry in result:
        if merged and merged[-1]["role"] == entry["role"]:
            prev = merged[-1]["content"]
            curr = entry["content"]
            if isinstance(prev, list) and isinstance(curr, list):
                prev.extend(curr)
            else:
                merged.append(entry)
        else:
            merged.append(entry)

    return "\n\n".join(system_parts), merged


def _translate_tools(
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """Convert OpenAI tool definitions to Anthropic format."""
    if not tools:
        return None
    result = []
    for t in tools:
        fn = t.get("function", {})
        result.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters") or {
                "type": "object",
                "properties": {},
            },
        })
    return result


# ---------------------------------------------------------------------------
# Response normalization: Anthropic → OpenAI-compatible objects
# ---------------------------------------------------------------------------


def _anthropic_response_to_openai(response: Any, model: str) -> Any:
    """Wrap an Anthropic ``Message`` as an OpenAI-compatible object."""
    content_parts: list[str] = []
    tool_calls: list[Any] = []

    for block in response.content:
        if block.type == "text":
            content_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(SimpleNamespace(
                id=block.id,
                type="function",
                function=SimpleNamespace(
                    name=block.name,
                    arguments=json.dumps(block.input),
                ),
            ))

    finish_reason = _ANTHROPIC_TO_OPENAI_STOP.get(
        response.stop_reason, response.stop_reason
    )

    message = SimpleNamespace(
        content="\n".join(content_parts) if content_parts else None,
        tool_calls=tool_calls or None,
        role="assistant",
    )
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(
        prompt_tokens=response.usage.input_tokens,
        completion_tokens=response.usage.output_tokens,
        total_tokens=response.usage.input_tokens + response.usage.output_tokens,
    )
    return SimpleNamespace(
        id=_make_completion_id(),
        choices=[choice],
        usage=usage,
        model=model,
    )


# ---------------------------------------------------------------------------
# Streaming normalization: Anthropic events → OpenAI-compatible chunks
# ---------------------------------------------------------------------------


async def _stream_as_openai_chunks(
    stream: Any, model: str
) -> AsyncIterator[Any]:
    """Translate an Anthropic event stream into OpenAI-compatible chunk objects.

    Each yielded object has ``choices[0].delta`` with ``content``,
    ``reasoning_content``, ``tool_calls``, ``role``, and the choice-level
    ``finish_reason`` — matching what ``astep_stream()`` expects.
    """
    completion_id = _make_completion_id()
    tool_call_index = 0
    block_to_tool_index: dict[int, int] = {}
    input_tokens = 0
    output_tokens = 0
    sent_role = False

    def _chunk(
        delta: dict[str, Any],
        finish_reason: str | None = None,
        usage: Any = None,
    ) -> Any:
        delta_ns = SimpleNamespace(**delta)
        choice = SimpleNamespace(delta=delta_ns, finish_reason=finish_reason)
        obj = SimpleNamespace(
            id=completion_id,
            choices=[choice],
            model=model,
            usage=usage,
        )
        return obj

    try:
        async for event in stream:
            if not sent_role:
                yield _chunk({"role": "assistant", "content": None})
                sent_role = True

            etype = event.type

            if etype == "message_start":
                input_tokens = event.message.usage.input_tokens

            elif etype == "content_block_start":
                block_type = event.content_block.type
                if block_type == "tool_use":
                    block_to_tool_index[event.index] = tool_call_index
                    tc = SimpleNamespace(
                        index=tool_call_index,
                        id=event.content_block.id,
                        type="function",
                        function=SimpleNamespace(
                            name=event.content_block.name,
                            arguments="",
                        ),
                    )
                    yield _chunk({"tool_calls": [tc]})
                    tool_call_index += 1

            elif etype == "content_block_delta":
                delta_type = event.delta.type
                if delta_type == "text_delta":
                    yield _chunk({"content": event.delta.text})
                elif delta_type == "thinking_delta":
                    yield _chunk({"reasoning_content": event.delta.thinking})
                elif delta_type == "input_json_delta":
                    idx = block_to_tool_index[event.index]
                    tc = SimpleNamespace(
                        index=idx,
                        id=None,
                        function=SimpleNamespace(
                            name=None,
                            arguments=event.delta.partial_json,
                        ),
                    )
                    yield _chunk({"tool_calls": [tc]})

            elif etype == "message_delta":
                finish_reason = _ANTHROPIC_TO_OPENAI_STOP.get(
                    event.delta.stop_reason, event.delta.stop_reason
                )
                output_tokens = event.usage.output_tokens
                yield _chunk({}, finish_reason=finish_reason)

                usage_ns = SimpleNamespace(
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                )
                yield _chunk({}, usage=usage_ns)

    except Exception as exc:
        raise LLMError(
            f"Error during Anthropic streaming ({type(exc).__name__}): {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


class AnthropicProvider(LLMProvider):
    """Provider backed by the Anthropic Python SDK.

    Parameters
    ----------
    config:
        ``LLMConfig`` from ``agent.yaml``.
    """

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if config.endpoint:
            kwargs["base_url"] = config.endpoint
        self._client = anthropic.AsyncAnthropic(**kwargs)

    def _build_kwargs(self, **overrides: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._config.name,
            "max_tokens": self._config.max_tokens,
        }
        if self._config.temperature is not None:
            kwargs["temperature"] = self._config.temperature
        kwargs.update(overrides)
        return kwargs

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        system_prompt, anthropic_msgs = _build_anthropic_messages(messages)
        call_kwargs = self._build_kwargs(**kwargs)
        call_kwargs["messages"] = anthropic_msgs
        if system_prompt:
            call_kwargs["system"] = system_prompt

        anthropic_tools = _translate_tools(tools)
        if anthropic_tools:
            call_kwargs["tools"] = anthropic_tools

        call_kwargs.pop("stream", None)
        call_kwargs.pop("stream_options", None)
        call_kwargs.pop("response_format", None)

        try:
            response = await self._client.messages.create(**call_kwargs)
        except Exception as exc:
            raise LLMError(
                f"LLM call failed ({type(exc).__name__}): {exc}"
            ) from exc

        return _anthropic_response_to_openai(response, self._config.name)

    async def stream_raw(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        system_prompt, anthropic_msgs = _build_anthropic_messages(messages)
        call_kwargs = self._build_kwargs(**kwargs)
        call_kwargs["messages"] = anthropic_msgs
        if system_prompt:
            call_kwargs["system"] = system_prompt

        anthropic_tools = _translate_tools(tools)
        if anthropic_tools:
            call_kwargs["tools"] = anthropic_tools

        call_kwargs.pop("stream", None)
        call_kwargs.pop("stream_options", None)
        call_kwargs.pop("response_format", None)

        try:
            async with self._client.messages.stream(**call_kwargs) as stream:
                async for chunk in _stream_as_openai_chunks(
                    stream, self._config.name
                ):
                    yield chunk
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(
                f"LLM streaming call failed ({type(exc).__name__}): {exc}"
            ) from exc

    async def complete_json(
        self,
        messages: list[dict[str, Any]],
        schema: Any,
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Structured output via Anthropic's tool-use pattern.

        Anthropic doesn't support ``response_format`` like OpenAI.
        For structured output, we add a tool with the schema and force
        the model to use it, then extract the result.
        """
        from fipsagents.baseagent.llm import _schema_to_response_format

        rf = _schema_to_response_format(schema)
        json_schema = rf["json_schema"]["schema"]
        schema_name = rf["json_schema"]["name"]

        extraction_tool: dict[str, Any] = {
            "name": schema_name,
            "description": f"Return the result as structured {schema_name}",
            "input_schema": json_schema,
        }

        system_prompt, anthropic_msgs = _build_anthropic_messages(messages)
        call_kwargs = self._build_kwargs(**kwargs)
        call_kwargs["messages"] = anthropic_msgs
        if system_prompt:
            call_kwargs["system"] = system_prompt

        all_tools = list(_translate_tools(tools) or [])
        all_tools.append(extraction_tool)
        call_kwargs["tools"] = all_tools
        call_kwargs["tool_choice"] = {"type": "tool", "name": schema_name}

        call_kwargs.pop("stream", None)
        call_kwargs.pop("stream_options", None)
        call_kwargs.pop("response_format", None)

        try:
            response = await self._client.messages.create(**call_kwargs)
        except Exception as exc:
            raise LLMError(
                f"LLM call failed ({type(exc).__name__}): {exc}"
            ) from exc

        for block in response.content:
            if block.type == "tool_use" and block.name == schema_name:
                content = json.dumps(block.input)
                message = SimpleNamespace(
                    content=content,
                    tool_calls=None,
                    role="assistant",
                )
                choice = SimpleNamespace(
                    message=message,
                    finish_reason="stop",
                )
                return SimpleNamespace(
                    id=_make_completion_id(),
                    choices=[choice],
                    usage=SimpleNamespace(
                        prompt_tokens=response.usage.input_tokens,
                        completion_tokens=response.usage.output_tokens,
                        total_tokens=(
                            response.usage.input_tokens
                            + response.usage.output_tokens
                        ),
                    ),
                    model=self._config.name,
                )

        raise LLMError(
            "Anthropic model did not return structured output via tool use"
        )
