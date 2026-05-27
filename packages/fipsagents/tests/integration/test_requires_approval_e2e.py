"""Integration test: requires_approval tool-call -> question -> resume cycle.

Exercises the full ``astep_stream`` loop twice -- first to trigger the
approval pause, then to resume after the user answers "allow".  The LLM is
stubbed (no real model calls) but the agent loop, tool dispatch, sentinel
messages, and pending-state bookkeeping are all real.
"""

import json
from types import SimpleNamespace

import pytest

from fipsagents.baseagent.agent import BaseAgent
from fipsagents.baseagent.events import (
    QuestionAsked,
    StreamComplete,
    ToolResultEvent,
)
from fipsagents.baseagent.tools import ToolRegistry, tool


# -- Tool under test -----------------------------------------------------------


@tool(description="Destructive action", visibility="both", requires_approval=True)
async def destroy(target: str) -> str:
    """Only runs after explicit user approval."""
    return f"destroyed:{target}"


# -- Stub helpers (same pattern as unit tests) ---------------------------------


def _tc_delta(index, *, call_id=None, name=None, arguments=None):
    fn = SimpleNamespace(name=name, arguments=arguments or "")
    return SimpleNamespace(index=index, id=call_id, function=fn)


def _chunk(*, tool_calls=None, content=None, finish_reason=None):
    delta = SimpleNamespace(
        content=content,
        reasoning_content=None,
        tool_calls=tool_calls,
    )
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=None)


class _StubLLM:
    """Yields pre-canned streaming turns in order."""

    def __init__(self, turns):
        self._turns = list(turns)
        self._idx = 0

    async def call_model_stream_raw(self, messages, tools=None, **kw):
        turn = self._turns[self._idx]
        self._idx += 1
        for c in turn:
            yield c


class _StubAgent(BaseAgent):
    def __init__(self, *, llm, extra_tools=None):
        self.llm = llm
        self.config = None
        self.messages = []
        self.tools = ToolRegistry()
        self._question_pending = None
        self._question_events = []
        self._subagent_events = []
        self._subagent_token_usage = []
        self._delegation_depth = 0
        self._inbound_auth_header = None
        self._reasoning_parser = None
        self._permission_source = None
        self._permission_mode = "enforce"
        self._permission_preapproved = set()

        if extra_tools:
            for t in extra_tools:
                self.tools.register(t)

    async def _inject_deferred_memory(self):
        return None


# -- Integration test ----------------------------------------------------------


@pytest.mark.asyncio
async def test_requires_approval_full_cycle():
    """Tool call -> QuestionAsked pause -> resume with approval -> tool executes."""

    # Turn 1: LLM requests the approval-guarded tool.
    # Turn 2 (after resume): LLM produces a text reply, no more tool calls.
    llm = _StubLLM([
        # Turn 1 — triggers approval pause
        [
            _chunk(tool_calls=[_tc_delta(0, call_id="tc_1", name="destroy")]),
            _chunk(tool_calls=[_tc_delta(0, arguments='{"target": "prod-db"}')]),
            _chunk(finish_reason="tool_calls"),
        ],
        # Turn 2 — after resume, LLM calls the same tool again (server
        # re-injects the tool call on resume) -- but this time the call_id
        # is pre-approved so it executes.  Then LLM finishes with text.
        [
            _chunk(tool_calls=[_tc_delta(0, call_id="tc_1", name="destroy")]),
            _chunk(tool_calls=[_tc_delta(0, arguments='{"target": "prod-db"}')]),
            _chunk(finish_reason="tool_calls"),
        ],
        # Turn 3 — final text reply after the tool result
        [
            _chunk(content="Deletion complete."),
            _chunk(finish_reason="stop"),
        ],
    ])
    agent = _StubAgent(llm=llm, extra_tools=[destroy])

    # -- Phase 1: initial call triggers approval pause -------------------------

    phase1_events = []
    async for ev in agent.astep_stream(max_iterations=5):
        phase1_events.append(ev)

    # QuestionAsked should have been emitted
    questions = [e for e in phase1_events if isinstance(e, QuestionAsked)]
    assert len(questions) == 1, f"Expected 1 QuestionAsked, got {len(questions)}"
    q = questions[0]
    assert "destroy" in q.question_text
    assert "prod-db" in q.question_text

    # Stream should have ended with finish_reason="question"
    completes = [e for e in phase1_events if isinstance(e, StreamComplete)]
    assert len(completes) == 1
    assert completes[0].finish_reason == "question"

    # Agent should have pending question state
    assert agent._question_pending is not None
    assert agent._question_pending["permission_ask"] is True
    assert agent._question_pending["tool_name"] == "destroy"
    assert agent._question_pending["tool_call_id"] == "tc_1"

    # A sentinel tool message should be in the conversation
    tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    sentinel = json.loads(tool_msgs[0]["content"])
    assert sentinel["__permission_pending__"] is True
    assert sentinel["tool_name"] == "destroy"

    # -- Phase 2: simulate resume after user approves --------------------------

    # 1. Pre-approve the tool_call_id (server does this on "allow" answer)
    agent._permission_preapproved.add("tc_1")

    # 2. Clear pending question (server does this on answer)
    agent._question_pending = None

    # 3. Replace the sentinel tool result with "Approved" (server does this)
    for msg in agent.messages:
        if msg.get("role") == "tool" and msg.get("tool_call_id") == "tc_1":
            msg["content"] = json.dumps({"approved": True})
            break

    # 4. Run astep_stream again — tool should execute this time
    phase2_events = []
    async for ev in agent.astep_stream(max_iterations=5):
        phase2_events.append(ev)

    # The tool should have actually executed
    tool_results = [e for e in phase2_events if isinstance(e, ToolResultEvent)]
    executed = [r for r in tool_results if r.name == "destroy" and not r.is_error]
    assert len(executed) == 1, (
        f"Expected 1 successful ToolResultEvent for 'destroy', got {len(executed)}; "
        f"all tool results: {[(r.name, r.content, r.is_error) for r in tool_results]}"
    )
    assert "destroyed:prod-db" in executed[0].content

    # No new approval questions should have been raised
    resume_questions = [e for e in phase2_events if isinstance(e, QuestionAsked)]
    assert len(resume_questions) == 0, (
        "Tool was pre-approved but still triggered a QuestionAsked"
    )

    # Stream should end normally
    resume_completes = [e for e in phase2_events if isinstance(e, StreamComplete)]
    assert len(resume_completes) == 1
    assert resume_completes[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_requires_approval_deny_skips_execution():
    """When the user denies approval, the tool should not execute."""

    llm = _StubLLM([
        # Turn 1 — triggers approval pause
        [
            _chunk(tool_calls=[_tc_delta(0, call_id="tc_d", name="destroy")]),
            _chunk(tool_calls=[_tc_delta(0, arguments='{"target": "staging"}')]),
            _chunk(finish_reason="tool_calls"),
        ],
        # Turn 2 — after denial, LLM gets the denial message and responds
        [
            _chunk(content="Understood, I won't delete anything."),
            _chunk(finish_reason="stop"),
        ],
    ])
    agent = _StubAgent(llm=llm, extra_tools=[destroy])

    # Phase 1: trigger approval
    async for _ in agent.astep_stream(max_iterations=5):
        pass

    assert agent._question_pending is not None

    # Phase 2: simulate denial — replace sentinel with denial message,
    # clear pending, do NOT add to _permission_preapproved
    agent._question_pending = None
    for msg in agent.messages:
        if msg.get("role") == "tool" and msg.get("tool_call_id") == "tc_d":
            msg["content"] = "Permission denied by user."
            break

    phase2_events = []
    async for ev in agent.astep_stream(max_iterations=5):
        phase2_events.append(ev)

    # Tool should NOT have executed
    tool_results = [e for e in phase2_events if isinstance(e, ToolResultEvent)]
    destroy_results = [r for r in tool_results if r.name == "destroy"]
    assert len(destroy_results) == 0, (
        "Tool executed despite denial — no ToolResultEvent expected for 'destroy'"
    )

    # LLM should have produced a text response
    completes = [e for e in phase2_events if isinstance(e, StreamComplete)]
    assert len(completes) == 1
    assert completes[0].finish_reason == "stop"
