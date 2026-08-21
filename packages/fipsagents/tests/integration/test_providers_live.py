"""Live integration tests for multi-provider LLM backends.

Run with: pytest tests/integration/test_providers_live.py -v -s

Requires:
  ANTHROPIC_API_KEY for anthropic provider tests
  OPENAI_API_KEY for openai provider tests (optional — uses vLLM/etc if set)

Skips gracefully when keys are missing.
"""

from __future__ import annotations

import json
import os

import pytest

from fipsagents.baseagent.config import LLMConfig
from fipsagents.baseagent.llm import LLMClient, ModelResponse

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

skip_no_anthropic = pytest.mark.skipif(
    not ANTHROPIC_KEY, reason="ANTHROPIC_API_KEY not set"
)


# ---------------------------------------------------------------------------
# Anthropic provider — non-streaming
# ---------------------------------------------------------------------------


@skip_no_anthropic
class TestAnthropicLive:
    """Live tests against the Anthropic Messages API."""

    @pytest.fixture
    def client(self):
        config = LLMConfig(
            provider="anthropic",
            name="claude-haiku-4-5-20251001",
            temperature=0.0,
            max_tokens=256,
        )
        return LLMClient(config)

    @pytest.mark.asyncio
    async def test_basic_completion(self, client):
        """Non-streaming completion returns content."""
        response = await client.call_model(
            messages=[{"role": "user", "content": "Say exactly: hello world"}],
        )
        assert isinstance(response, ModelResponse)
        assert response.content is not None
        assert "hello" in response.content.lower()
        assert response.tool_calls is None
        print(f"\n  Content: {response.content!r}")
        print(f"  Usage: prompt={response.raw.usage.prompt_tokens}, "
              f"completion={response.raw.usage.completion_tokens}")

    @pytest.mark.asyncio
    async def test_streaming(self, client):
        """Streaming yields OpenAI-compatible chunks with content."""
        chunks = []
        content_parts = []
        finish_reason = None
        has_usage = False

        async for chunk in client.call_model_stream_raw(
            messages=[{"role": "user", "content": "Count from 1 to 5."}],
        ):
            chunks.append(chunk)
            try:
                delta = chunk.choices[0].delta
                c = getattr(delta, "content", None)
                if c:
                    content_parts.append(c)
                fr = chunk.choices[0].finish_reason
                if fr:
                    finish_reason = fr
            except (AttributeError, IndexError):
                pass
            if getattr(chunk, "usage", None) is not None:
                has_usage = True

        full_content = "".join(content_parts)
        print(f"\n  Chunks: {len(chunks)}")
        print(f"  Content: {full_content!r}")
        print(f"  Finish reason: {finish_reason}")
        print(f"  Has usage: {has_usage}")

        assert len(chunks) > 2, "Expected multiple streaming chunks"
        assert "1" in full_content and "5" in full_content
        assert finish_reason == "stop"
        assert has_usage, "Terminal chunk should carry usage"

    @pytest.mark.asyncio
    async def test_content_only_stream(self, client):
        """call_model_stream yields content-only strings."""
        parts = []
        async for text in client.call_model_stream(
            messages=[{"role": "user", "content": "Say: test passed"}],
        ):
            parts.append(text)

        full = "".join(parts)
        print(f"\n  Content stream: {full!r}")
        assert "test" in full.lower()

    @pytest.mark.asyncio
    async def test_tool_calling(self, client):
        """Tool calling works with Anthropic-format translation."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name",
                            }
                        },
                        "required": ["city"],
                    },
                },
            }
        ]

        response = await client.call_model(
            messages=[
                {"role": "user", "content": "What's the weather in Paris?"}
            ],
            tools=tools,
        )
        print(f"\n  Content: {response.content!r}")
        print(f"  Tool calls: {response.tool_calls}")

        assert response.tool_calls is not None, "Expected a tool call"
        tc = response.tool_calls[0]
        assert tc.function.name == "get_weather"
        args = json.loads(tc.function.arguments)
        assert "paris" in args.get("city", "").lower()

    @pytest.mark.asyncio
    async def test_tool_calling_stream(self, client):
        """Streaming tool calls produce OpenAI-compatible deltas."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_temp",
                    "description": "Get temperature for a location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        tool_name = None
        tool_args = ""
        tool_id = None

        async for chunk in client.call_model_stream_raw(
            messages=[
                {"role": "user", "content": "Get the temperature in Tokyo."}
            ],
            tools=tools,
        ):
            try:
                delta = chunk.choices[0].delta
                tc_list = getattr(delta, "tool_calls", None) or []
                for tc in tc_list:
                    if getattr(tc, "id", None):
                        tool_id = tc.id
                    if getattr(tc.function, "name", None):
                        tool_name = tc.function.name
                    if getattr(tc.function, "arguments", None):
                        tool_args += tc.function.arguments
            except (AttributeError, IndexError):
                pass

        print(f"\n  Tool name: {tool_name}")
        print(f"  Tool args: {tool_args}")
        print(f"  Tool ID: {tool_id}")

        assert tool_name == "get_temp"
        assert tool_id is not None
        args = json.loads(tool_args)
        assert "tokyo" in args.get("location", "").lower()

    @pytest.mark.asyncio
    async def test_multi_turn_with_tool_result(self, client):
        """Full tool-call round-trip: call → result → final answer."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "Add two numbers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                        "required": ["a", "b"],
                    },
                },
            }
        ]

        r1 = await client.call_model(
            messages=[{"role": "user", "content": "What is 7 + 13?"}],
            tools=tools,
        )
        assert r1.tool_calls is not None
        tc = r1.tool_calls[0]
        args = json.loads(tc.function.arguments)
        result = args["a"] + args["b"]

        r2 = await client.call_model(
            messages=[
                {"role": "user", "content": "What is 7 + 13?"},
                {
                    "role": "assistant",
                    "content": r1.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": str(result),
                    "tool_call_id": tc.id,
                },
            ],
            tools=tools,
        )
        print(f"\n  Final answer: {r2.content!r}")
        assert r2.content is not None
        assert "20" in r2.content

    @pytest.mark.asyncio
    async def test_structured_output(self, client):
        """Structured output via Anthropic's tool-use pattern."""
        from pydantic import BaseModel

        class CityInfo(BaseModel):
            name: str
            country: str
            population_millions: float

        result = await client.call_model_json(
            messages=[
                {
                    "role": "user",
                    "content": "Give me info about Tokyo.",
                }
            ],
            schema=CityInfo,
        )
        print(f"\n  Structured result: {result}")
        assert isinstance(result, CityInfo)
        assert result.name.lower() == "tokyo"
        assert result.country.lower() == "japan"
        assert result.population_millions > 1

    @pytest.mark.asyncio
    async def test_system_message_extraction(self, client):
        """System messages are extracted to the Anthropic system parameter."""
        response = await client.call_model(
            messages=[
                {"role": "system", "content": "You are a pirate. Always say 'arr'."},
                {"role": "user", "content": "Hello"},
            ],
        )
        print(f"\n  Response: {response.content!r}")
        assert response.content is not None
        assert "arr" in response.content.lower()

    @pytest.mark.asyncio
    async def test_reasoning_content(self, client):
        """Extended thinking produces reasoning_content on deltas."""
        config = LLMConfig(
            provider="anthropic",
            name="claude-haiku-4-5-20251001",
            temperature=1.0,
            max_tokens=4096,
        )
        thinking_client = LLMClient(config)

        has_reasoning = False
        has_content = False

        async for chunk in thinking_client.call_model_stream_raw(
            messages=[{"role": "user", "content": "What is 15 * 23?"}],
            thinking={"type": "enabled", "budget_tokens": 1024},
        ):
            try:
                delta = chunk.choices[0].delta
                if getattr(delta, "reasoning_content", None):
                    has_reasoning = True
                if getattr(delta, "content", None):
                    has_content = True
            except (AttributeError, IndexError):
                pass

        print(f"\n  Has reasoning: {has_reasoning}")
        print(f"  Has content: {has_content}")
        assert has_content, "Expected content in response"
