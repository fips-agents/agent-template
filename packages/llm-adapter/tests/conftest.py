"""Shared fixtures for llm-adapter tests."""

from __future__ import annotations

import pytest

from llm_adapter.models import (
    ChatCompletionRequest,
    ChatMessage,
    Tool,
    ToolCall,
    ToolCallFunction,
    ToolFunction,
)


@pytest.fixture
def simple_request():
    return ChatCompletionRequest(
        model="claude-sonnet-4-6",
        messages=[
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Hello"),
        ],
        max_tokens=100,
    )


@pytest.fixture
def tool_request():
    return ChatCompletionRequest(
        model="claude-sonnet-4-6",
        messages=[ChatMessage(role="user", content="Search for weather")],
        max_tokens=100,
        tools=[
            Tool(
                function=ToolFunction(
                    name="search",
                    description="Search the web",
                    parameters={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                )
            )
        ],
    )


@pytest.fixture
def conversation_with_tool_results():
    """Full multi-turn: user -> assistant(tool_calls) -> tool -> tool."""
    return ChatCompletionRequest(
        model="claude-sonnet-4-6",
        messages=[
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="What's the weather?"),
            ChatMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function=ToolCallFunction(
                            name="search",
                            arguments='{"query":"NYC weather"}',
                        ),
                    ),
                    ToolCall(
                        id="call_2",
                        function=ToolCallFunction(
                            name="search",
                            arguments='{"query":"LA weather"}',
                        ),
                    ),
                ],
            ),
            ChatMessage(role="tool", tool_call_id="call_1", content="NYC: 75F sunny"),
            ChatMessage(role="tool", tool_call_id="call_2", content="LA: 82F clear"),
        ],
        max_tokens=200,
    )
