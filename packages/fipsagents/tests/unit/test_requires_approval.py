"""Tests for @tool(requires_approval=...) in astep_stream."""

from types import SimpleNamespace

import pytest

from fipsagents.baseagent.agent import BaseAgent
from fipsagents.baseagent.events import (
    PermissionDecisionMade,
    QuestionAsked,
    StreamComplete,
    ToolResultEvent,
)
from fipsagents.baseagent.tools import ToolRegistry, tool


# -- Test tools ----------------------------------------------------------------


@tool(description="Safe tool", visibility="both")
async def safe_echo(msg: str) -> str:
    return f"echo:{msg}"


@tool(description="Dangerous delete", visibility="both", requires_approval=True)
async def delete_all(target: str) -> str:
    return f"deleted:{target}"


def _predicate_deny(**kwargs) -> bool:
    """Sync predicate that always requires approval."""
    return True


def _predicate_allow(**kwargs) -> bool:
    """Sync predicate that never requires approval."""
    return False


def _predicate_conditional(target: str = "", **kwargs) -> bool:
    """Require approval only for production targets."""
    return "prod" in target


async def _async_predicate_deny(**kwargs) -> bool:
    return True


async def _async_predicate_conditional(target: str = "", **kwargs) -> bool:
    return "prod" in target


def _make_tool_with_predicate(predicate):
    """Build a @tool-decorated function with a custom requires_approval predicate."""

    @tool(
        description="Guarded tool",
        visibility="both",
        requires_approval=predicate,
    )
    async def guarded(target: str) -> str:
        return f"guarded:{target}"

    return guarded


# -- Stub helpers (same pattern as test_astep_stream_permissions) ---------------


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
    def __init__(self, turns):
        self._turns = list(turns)
        self._idx = 0

    async def call_model_stream_raw(self, messages, tools=None, **kw):
        turn = self._turns[self._idx]
        self._idx += 1
        for c in turn:
            yield c


class _StubAgent(BaseAgent):
    def __init__(self, *, llm, extra_tools=None, permission_source=None):
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
        self._permission_source = permission_source
        self._permission_mode = "enforce"
        self._permission_preapproved = set()

        if extra_tools:
            for t in extra_tools:
                self.tools.register(t)

    async def _inject_deferred_memory(self):
        return None


# -- Tests --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_requires_approval_true_pauses():
    """Tool with requires_approval=True should emit QuestionAsked and pause."""
    llm = _StubLLM([
        [
            _chunk(tool_calls=[_tc_delta(0, call_id="c1", name="delete_all")]),
            _chunk(tool_calls=[_tc_delta(0, arguments='{"target": "db"}')]),
            _chunk(finish_reason="tool_calls"),
        ],
    ])
    agent = _StubAgent(llm=llm, extra_tools=[delete_all])

    events = []
    async for ev in agent.astep_stream(max_iterations=5):
        events.append(ev)

    questions = [e for e in events if isinstance(e, QuestionAsked)]
    assert len(questions) == 1
    assert "delete_all" in questions[0].question_text
    assert "requires approval" in questions[0].question_text

    complete = [e for e in events if isinstance(e, StreamComplete)]
    assert len(complete) == 1
    assert complete[0].finish_reason == "question"

    assert agent._question_pending is not None
    assert agent._question_pending["permission_ask"] is True
    assert agent._question_pending["tool_name"] == "delete_all"
    assert agent._question_pending["tool_args"] == {"target": "db"}
    assert agent._question_pending["tool_call_id"] == "c1"


@pytest.mark.asyncio
async def test_no_requires_approval_executes_normally():
    """Tool without requires_approval should execute without pausing."""
    llm = _StubLLM([
        [
            _chunk(tool_calls=[_tc_delta(0, call_id="c1", name="safe_echo")]),
            _chunk(tool_calls=[_tc_delta(0, arguments='{"msg": "hi"}')]),
            _chunk(finish_reason="tool_calls"),
        ],
        [
            _chunk(content="Done"),
            _chunk(finish_reason="stop"),
        ],
    ])
    agent = _StubAgent(llm=llm, extra_tools=[safe_echo])

    events = []
    async for ev in agent.astep_stream(max_iterations=5):
        events.append(ev)

    questions = [e for e in events if isinstance(e, QuestionAsked)]
    assert len(questions) == 0

    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_results) == 1
    assert "echo:hi" in tool_results[0].content
    assert tool_results[0].is_error is False


@pytest.mark.asyncio
async def test_preapproved_skips_requires_approval():
    """A tool call ID in _permission_preapproved should bypass requires_approval."""
    llm = _StubLLM([
        [
            _chunk(tool_calls=[_tc_delta(0, call_id="c1", name="delete_all")]),
            _chunk(tool_calls=[_tc_delta(0, arguments='{"target": "db"}')]),
            _chunk(finish_reason="tool_calls"),
        ],
        [
            _chunk(content="Done"),
            _chunk(finish_reason="stop"),
        ],
    ])
    agent = _StubAgent(llm=llm, extra_tools=[delete_all])
    agent._permission_preapproved.add("c1")

    events = []
    async for ev in agent.astep_stream(max_iterations=5):
        events.append(ev)

    questions = [e for e in events if isinstance(e, QuestionAsked)]
    assert len(questions) == 0

    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_results) == 1
    assert "deleted:db" in tool_results[0].content


@pytest.mark.asyncio
async def test_callable_predicate_true_pauses():
    """A callable predicate returning True should pause for approval."""
    guarded = _make_tool_with_predicate(_predicate_deny)

    llm = _StubLLM([
        [
            _chunk(tool_calls=[_tc_delta(0, call_id="c1", name="guarded")]),
            _chunk(tool_calls=[_tc_delta(0, arguments='{"target": "staging"}')]),
            _chunk(finish_reason="tool_calls"),
        ],
    ])
    agent = _StubAgent(llm=llm, extra_tools=[guarded])

    events = []
    async for ev in agent.astep_stream(max_iterations=5):
        events.append(ev)

    questions = [e for e in events if isinstance(e, QuestionAsked)]
    assert len(questions) == 1
    assert agent._question_pending is not None
    assert agent._question_pending["tool_name"] == "guarded"


@pytest.mark.asyncio
async def test_callable_predicate_false_executes():
    """A callable predicate returning False should let the tool run."""
    guarded = _make_tool_with_predicate(_predicate_allow)

    llm = _StubLLM([
        [
            _chunk(tool_calls=[_tc_delta(0, call_id="c1", name="guarded")]),
            _chunk(tool_calls=[_tc_delta(0, arguments='{"target": "dev"}')]),
            _chunk(finish_reason="tool_calls"),
        ],
        [
            _chunk(content="Done"),
            _chunk(finish_reason="stop"),
        ],
    ])
    agent = _StubAgent(llm=llm, extra_tools=[guarded])

    events = []
    async for ev in agent.astep_stream(max_iterations=5):
        events.append(ev)

    questions = [e for e in events if isinstance(e, QuestionAsked)]
    assert len(questions) == 0

    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_results) == 1
    assert "guarded:dev" in tool_results[0].content


@pytest.mark.asyncio
async def test_conditional_predicate_matches():
    """A conditional predicate should pause only when condition is met."""
    guarded = _make_tool_with_predicate(_predicate_conditional)

    # "prod-db" triggers the predicate.
    llm = _StubLLM([
        [
            _chunk(tool_calls=[_tc_delta(0, call_id="c1", name="guarded")]),
            _chunk(tool_calls=[_tc_delta(0, arguments='{"target": "prod-db"}')]),
            _chunk(finish_reason="tool_calls"),
        ],
    ])
    agent = _StubAgent(llm=llm, extra_tools=[guarded])

    events = []
    async for ev in agent.astep_stream(max_iterations=5):
        events.append(ev)

    questions = [e for e in events if isinstance(e, QuestionAsked)]
    assert len(questions) == 1


@pytest.mark.asyncio
async def test_conditional_predicate_no_match():
    """A conditional predicate should let the tool run when condition isn't met."""
    guarded = _make_tool_with_predicate(_predicate_conditional)

    # "staging" does NOT trigger the predicate.
    llm = _StubLLM([
        [
            _chunk(tool_calls=[_tc_delta(0, call_id="c1", name="guarded")]),
            _chunk(tool_calls=[_tc_delta(0, arguments='{"target": "staging"}')]),
            _chunk(finish_reason="tool_calls"),
        ],
        [
            _chunk(content="Done"),
            _chunk(finish_reason="stop"),
        ],
    ])
    agent = _StubAgent(llm=llm, extra_tools=[guarded])

    events = []
    async for ev in agent.astep_stream(max_iterations=5):
        events.append(ev)

    questions = [e for e in events if isinstance(e, QuestionAsked)]
    assert len(questions) == 0

    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_results) == 1
    assert "guarded:staging" in tool_results[0].content


@pytest.mark.asyncio
async def test_async_predicate_pauses():
    """An async callable predicate should work correctly."""
    guarded = _make_tool_with_predicate(_async_predicate_deny)

    llm = _StubLLM([
        [
            _chunk(tool_calls=[_tc_delta(0, call_id="c1", name="guarded")]),
            _chunk(tool_calls=[_tc_delta(0, arguments='{"target": "x"}')]),
            _chunk(finish_reason="tool_calls"),
        ],
    ])
    agent = _StubAgent(llm=llm, extra_tools=[guarded])

    events = []
    async for ev in agent.astep_stream(max_iterations=5):
        events.append(ev)

    questions = [e for e in events if isinstance(e, QuestionAsked)]
    assert len(questions) == 1


@pytest.mark.asyncio
async def test_async_conditional_predicate():
    """An async conditional predicate should work for both match and non-match."""
    guarded = _make_tool_with_predicate(_async_predicate_conditional)

    # Match case: "prod-db" triggers.
    llm = _StubLLM([
        [
            _chunk(tool_calls=[_tc_delta(0, call_id="c1", name="guarded")]),
            _chunk(tool_calls=[_tc_delta(0, arguments='{"target": "prod-db"}')]),
            _chunk(finish_reason="tool_calls"),
        ],
    ])
    agent = _StubAgent(llm=llm, extra_tools=[guarded])

    events = []
    async for ev in agent.astep_stream(max_iterations=5):
        events.append(ev)

    questions = [e for e in events if isinstance(e, QuestionAsked)]
    assert len(questions) == 1

    # Non-match case: "dev" should not trigger.
    guarded2 = _make_tool_with_predicate(_async_predicate_conditional)
    # Need a differently named tool since guarded is already registered.
    # Just re-create the agent.
    llm2 = _StubLLM([
        [
            _chunk(tool_calls=[_tc_delta(0, call_id="c2", name="guarded")]),
            _chunk(tool_calls=[_tc_delta(0, arguments='{"target": "dev"}')]),
            _chunk(finish_reason="tool_calls"),
        ],
        [
            _chunk(content="Done"),
            _chunk(finish_reason="stop"),
        ],
    ])
    agent2 = _StubAgent(llm=llm2, extra_tools=[guarded2])

    events2 = []
    async for ev in agent2.astep_stream(max_iterations=5):
        events2.append(ev)

    questions2 = [e for e in events2 if isinstance(e, QuestionAsked)]
    assert len(questions2) == 0

    tool_results2 = [e for e in events2 if isinstance(e, ToolResultEvent)]
    assert len(tool_results2) == 1
    assert "guarded:dev" in tool_results2[0].content


@pytest.mark.asyncio
async def test_requires_approval_takes_precedence_over_permission_source():
    """requires_approval should fire BEFORE the permission source check.

    When requires_approval triggers, no PermissionDecisionMade event should
    be emitted because the permission source check is skipped entirely.
    """
    from fipsagents.server.permissions import PermissionRule, StaticPermissionSource

    llm = _StubLLM([
        [
            _chunk(tool_calls=[_tc_delta(0, call_id="c1", name="delete_all")]),
            _chunk(tool_calls=[_tc_delta(0, arguments='{"target": "db"}')]),
            _chunk(finish_reason="tool_calls"),
        ],
    ])
    src = StaticPermissionSource(
        rules=[PermissionRule(id="r1", tool="delete_all", action="allow")]
    )
    agent = _StubAgent(llm=llm, extra_tools=[delete_all], permission_source=src)

    events = []
    async for ev in agent.astep_stream(max_iterations=5):
        events.append(ev)

    # requires_approval fires first, so we get a QuestionAsked...
    questions = [e for e in events if isinstance(e, QuestionAsked)]
    assert len(questions) == 1

    # ...and no PermissionDecisionMade because we never reached that check.
    perm_events = [e for e in events if isinstance(e, PermissionDecisionMade)]
    assert len(perm_events) == 0


@pytest.mark.asyncio
async def test_pending_state_has_correct_format():
    """The _question_pending dict should match the permission ask format."""
    llm = _StubLLM([
        [
            _chunk(tool_calls=[_tc_delta(0, call_id="c1", name="delete_all")]),
            _chunk(tool_calls=[_tc_delta(0, arguments='{"target": "db"}')]),
            _chunk(finish_reason="tool_calls"),
        ],
    ])
    agent = _StubAgent(llm=llm, extra_tools=[delete_all])

    async for _ in agent.astep_stream(max_iterations=5):
        pass

    pending = agent._question_pending
    assert pending is not None
    # Must have all the fields the server-layer resume handler expects.
    assert "question_id" in pending
    assert pending["question_id"].startswith("perm_")
    assert pending["permission_ask"] is True
    assert pending["tool_name"] == "delete_all"
    assert pending["tool_args"] == {"target": "db"}
    assert pending["tool_call_id"] == "c1"
    assert pending["multiple"] is False
    assert pending["allow_custom"] is False
    assert len(pending["options"]) == 2
    assert pending["options"][0]["value"] == "allow"
    assert pending["options"][1]["value"] == "deny"


@pytest.mark.asyncio
async def test_sentinel_message_appended():
    """A __permission_pending__ sentinel message should be appended."""
    import json

    llm = _StubLLM([
        [
            _chunk(tool_calls=[_tc_delta(0, call_id="c1", name="delete_all")]),
            _chunk(tool_calls=[_tc_delta(0, arguments='{"target": "db"}')]),
            _chunk(finish_reason="tool_calls"),
        ],
    ])
    agent = _StubAgent(llm=llm, extra_tools=[delete_all])

    async for _ in agent.astep_stream(max_iterations=5):
        pass

    # Find the tool-role sentinel message.
    tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    sentinel = json.loads(tool_msgs[0]["content"])
    assert sentinel["__permission_pending__"] is True
    assert sentinel["tool_name"] == "delete_all"
    assert "question_id" in sentinel


@pytest.mark.asyncio
async def test_mixed_tools_only_guarded_pauses():
    """When multiple tools are called, only the one with requires_approval pauses.

    The safe tool (called first) should execute normally. The guarded tool
    (called second) should trigger the approval pause.
    """
    llm = _StubLLM([
        [
            _chunk(tool_calls=[_tc_delta(0, call_id="c_safe", name="safe_echo")]),
            _chunk(tool_calls=[_tc_delta(0, arguments='{"msg": "hi"}')]),
            _chunk(tool_calls=[_tc_delta(1, call_id="c_danger", name="delete_all")]),
            _chunk(tool_calls=[_tc_delta(1, arguments='{"target": "db"}')]),
            _chunk(finish_reason="tool_calls"),
        ],
    ])
    agent = _StubAgent(llm=llm, extra_tools=[safe_echo, delete_all])

    events = []
    async for ev in agent.astep_stream(max_iterations=5):
        events.append(ev)

    # safe_echo should have executed.
    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    safe_results = [e for e in tool_results if e.name == "safe_echo"]
    assert len(safe_results) == 1
    assert "echo:hi" in safe_results[0].content
    assert safe_results[0].is_error is False

    # delete_all should have triggered approval.
    questions = [e for e in events if isinstance(e, QuestionAsked)]
    assert len(questions) == 1
    assert "delete_all" in questions[0].question_text

    complete = [e for e in events if isinstance(e, StreamComplete)]
    assert complete[0].finish_reason == "question"
