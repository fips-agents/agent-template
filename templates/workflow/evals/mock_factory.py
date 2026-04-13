"""Mock object factories for eval workflow instances and LLM responses."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from fipsagents.baseagent.config import AgentConfig, BackoffConfig, LLMConfig, LoopConfig
from fipsagents.baseagent.llm import ModelResponse

from evals.discovery import (
    _discover_llm_tool_name,
)


def _build_mock_litellm_response(
    content: str | None = None,
    tool_calls: list[Any] | None = None,
) -> Any:
    """Construct a fake litellm response matching ModelResponse expectations."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _make_tool_call_obj(name: str, arguments: dict[str, Any]) -> Any:
    """Build a fake tool-call object in OpenAI format."""
    return SimpleNamespace(
        id=f"call_eval_{name}",
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments),
        ),
    )


def _build_mock_responses(
    query: str,
) -> tuple[list[Any], str]:
    """Produce mock LLM responses for a workflow eval case.

    Returns (call_model_side_effects, summary_text).

    The responses simulate a single tool call round followed by a text
    response (for research nodes) and a plain text response (for
    summarize nodes).
    """
    tool_name = _discover_llm_tool_name()
    side_effects: list[Any] = []

    if tool_name is not None:
        # Simulate a tool call then a text response.
        search_tc = _make_tool_call_obj(tool_name, {"query": query})
        side_effects.append(
            ModelResponse(_build_mock_litellm_response(tool_calls=[search_tc]))
        )

    # Text response after tool calls (or as the only response).
    side_effects.append(
        ModelResponse(
            _build_mock_litellm_response(
                content=f"Based on research about '{query}', here are the findings."
            )
        )
    )

    # Summary response (for the summarize node).
    summary_text = f"Summary of research on '{query}': key findings and conclusions."
    side_effects.append(
        ModelResponse(
            _build_mock_litellm_response(content=summary_text)
        )
    )

    return side_effects, summary_text


def create_mock_config() -> AgentConfig:
    """Create a mock AgentConfig for eval use."""
    return AgentConfig(
        model=LLMConfig(
            endpoint="http://eval-mock:8321/v1",
            name="eval-mock-model",
            temperature=0.0,
            max_tokens=1024,
        ),
        loop=LoopConfig(
            max_iterations=5,
            backoff=BackoffConfig(initial=0.01, max=0.05, multiplier=2.0),
        ),
    )
