"""Tests for the ``delegate_to_agent`` tool factory.

Verifies the full lifecycle of :func:`make_delegate_tool`:

- Happy path (event sequence, JSON output, token recording)
- Unknown agent name
- Depth cap enforcement
- Transport failures (timeout, remote error)
- Identity / header propagation
- Permission scope warning deduplication
- Defensive behavior when Step 6 contract attributes are absent
- Tool description contains registered subagent names + when_to_use hints
"""

from __future__ import annotations

import json
import types

import pytest

from fipsagents.baseagent.config import (
    IdentityServiceAccount,
    InProcessTransportConfig,
    RemoteTransportConfig,
    SubagentConfig,
)
from fipsagents.baseagent.events import SubagentCompleted, SubagentFailed, SubagentInvoked
from fipsagents.baseagent.tools.delegate import (
    _warned_permission_scope,
    make_delegate_tool,
)
from fipsagents.baseagent.tools import _TOOL_MARKER
from fipsagents.subagents.transport import SubagentTransport
from fipsagents.subagents.types import (
    MaxDelegationDepthError,
    SubagentRemoteError,
    SubagentResult,
    SubagentTimeoutError,
)


# ---------------------------------------------------------------------------
# Helpers: config factories
# ---------------------------------------------------------------------------


def _remote_config(
    name: str = "helper",
    when_to_use: str = "Use for research",
    permission_scope: str | None = None,
    identity: str | IdentityServiceAccount = "inherit",
    max_depth: int = 3,
) -> SubagentConfig:
    return SubagentConfig(
        name=name,
        description="A helpful research agent",
        when_to_use=when_to_use,
        transport=RemoteTransportConfig(
            type="remote",
            url="http://helper:8080/v1",
            timeout_seconds=30.0,
        ),
        permission_scope=permission_scope,
        identity=identity,
        max_depth=max_depth,
    )


def _inprocess_config(
    name: str = "local_helper",
    when_to_use: str = "Use for local tasks",
    max_depth: int = 3,
) -> SubagentConfig:
    return SubagentConfig(
        name=name,
        description="A local helper agent",
        when_to_use=when_to_use,
        transport=InProcessTransportConfig(
            type="inprocess",
            class_path="tests.fake.HelperAgent",
        ),
        max_depth=max_depth,
    )


# ---------------------------------------------------------------------------
# Helpers: stub agent + fake transport
# ---------------------------------------------------------------------------


def _make_agent(
    subagent_configs: list[SubagentConfig] | None = None,
    delegation_depth: int = 0,
    inbound_auth: str | None = None,
) -> types.SimpleNamespace:
    """Build a stub agent that satisfies the Step 6 contract."""
    configs = subagent_configs or []
    subagents = {cfg.name: cfg for cfg in configs}
    return types.SimpleNamespace(
        subagents=subagents,
        _subagent_events=[],
        _subagent_token_usage=[],
        _delegation_depth=delegation_depth,
        _inbound_auth_header=inbound_auth,
    )


class FakeTransport(SubagentTransport):
    """Configurable fake transport for unit tests.

    Attributes:
        result: The ``SubagentResult`` to return on ``invoke``.
        raise_exc: If set, raise this exception instead of returning.
        received_headers: The ``headers`` dict passed on the last call.
        call_count: How many times ``invoke`` was called.
    """

    def __init__(
        self,
        result: SubagentResult | None = None,
        raise_exc: Exception | None = None,
    ) -> None:
        self.result = result or SubagentResult(
            agent_name="helper",
            content="Done!",
            tokens_used={"input": 10, "output": 5, "cached": 0},
            tool_calls_made=0,
            cost_usd=0.01,
            span_id=None,
            finish_reason="stop",
        )
        self.raise_exc = raise_exc
        self.received_headers: dict[str, str] | None = None
        self.received_task: str | None = None
        self.received_context: str | None = None
        self.call_count = 0

    async def invoke(
        self,
        *,
        task: str,
        context: str = "",
        headers: dict[str, str] | None = None,
        timeout_seconds: float = 60.0,
    ) -> SubagentResult:
        self.call_count += 1
        self.received_headers = headers or {}
        self.received_task = task
        self.received_context = context
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.result


def _make_transport_factory(transport: FakeTransport):
    """Return a factory that always yields *transport*, ignoring name/config."""
    def factory(name: str, config: SubagentConfig) -> SubagentTransport:
        return transport
    return factory


# ---------------------------------------------------------------------------
# Tests: Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_returns_json_parseable_result(self):
        cfg = _remote_config(name="helper")
        agent = _make_agent([cfg])
        fake_transport = FakeTransport()
        tool_fn = make_delegate_tool(
            agent, transport_factory=_make_transport_factory(fake_transport)
        )
        raw = await tool_fn(agent_name="helper", task="Do something")
        result = json.loads(raw)
        assert result["agent_name"] == "helper"
        assert result["content"] == "Done!"
        assert "tokens_used" in result
        assert "tool_calls_made" in result
        assert "cost_usd" in result
        assert "finish_reason" in result
        assert "span_id" in result

    @pytest.mark.asyncio
    async def test_result_roundtrips_to_subagent_result_shape(self):
        cfg = _remote_config(name="helper")
        agent = _make_agent([cfg])
        fake_transport = FakeTransport()
        tool_fn = make_delegate_tool(
            agent, transport_factory=_make_transport_factory(fake_transport)
        )
        raw = await tool_fn(agent_name="helper", task="Do something")
        result = json.loads(raw)
        # Verify all SubagentResult fields are present in the JSON.
        for key in ("agent_name", "content", "tokens_used", "tool_calls_made",
                    "cost_usd", "span_id", "finish_reason"):
            assert key in result, f"Missing key: {key}"

    @pytest.mark.asyncio
    async def test_event_sequence_is_invoked_then_completed(self):
        cfg = _remote_config(name="helper")
        agent = _make_agent([cfg])
        fake_transport = FakeTransport()
        tool_fn = make_delegate_tool(
            agent, transport_factory=_make_transport_factory(fake_transport)
        )
        await tool_fn(agent_name="helper", task="Do something")

        events = agent._subagent_events
        assert len(events) == 2
        assert isinstance(events[0], SubagentInvoked)
        assert isinstance(events[1], SubagentCompleted)
        assert events[0].agent_name == "helper"
        assert events[1].agent_name == "helper"

    @pytest.mark.asyncio
    async def test_span_id_is_consistent_across_events_and_result(self):
        cfg = _remote_config(name="helper")
        agent = _make_agent([cfg])
        fake_transport = FakeTransport()
        tool_fn = make_delegate_tool(
            agent, transport_factory=_make_transport_factory(fake_transport)
        )
        raw = await tool_fn(agent_name="helper", task="task")
        result = json.loads(raw)

        events = agent._subagent_events
        assert events[0].span_id == events[1].span_id
        assert result["span_id"] == events[0].span_id
        assert result["span_id"].startswith("subagent-")

    @pytest.mark.asyncio
    async def test_token_usage_appended(self):
        cfg = _remote_config(name="helper")
        agent = _make_agent([cfg])
        expected_tokens = {"input": 10, "output": 5, "cached": 0}
        fake_transport = FakeTransport(
            result=SubagentResult(
                agent_name="helper",
                content="ok",
                tokens_used=expected_tokens,
                tool_calls_made=0,
                cost_usd=0.0,
            )
        )
        tool_fn = make_delegate_tool(
            agent, transport_factory=_make_transport_factory(fake_transport)
        )
        await tool_fn(agent_name="helper", task="task")
        assert len(agent._subagent_token_usage) == 1
        assert agent._subagent_token_usage[0] == expected_tokens

    @pytest.mark.asyncio
    async def test_context_forwarded_to_transport(self):
        cfg = _remote_config(name="helper")
        agent = _make_agent([cfg])
        fake_transport = FakeTransport()
        tool_fn = make_delegate_tool(
            agent, transport_factory=_make_transport_factory(fake_transport)
        )
        await tool_fn(agent_name="helper", task="do X", context="background info")
        assert fake_transport.received_task == "do X"
        assert fake_transport.received_context == "background info"


# ---------------------------------------------------------------------------
# Tests: Unknown agent name
# ---------------------------------------------------------------------------


class TestUnknownAgentName:
    @pytest.mark.asyncio
    async def test_raises_value_error(self):
        cfg = _remote_config(name="helper")
        agent = _make_agent([cfg])
        fake_transport = FakeTransport()
        tool_fn = make_delegate_tool(
            agent, transport_factory=_make_transport_factory(fake_transport)
        )
        with pytest.raises(ValueError, match="Unknown subagent 'missing'"):
            await tool_fn(agent_name="missing", task="task")

    @pytest.mark.asyncio
    async def test_no_events_emitted_for_unknown_agent(self):
        cfg = _remote_config(name="helper")
        agent = _make_agent([cfg])
        fake_transport = FakeTransport()
        tool_fn = make_delegate_tool(
            agent, transport_factory=_make_transport_factory(fake_transport)
        )
        with pytest.raises(ValueError):
            await tool_fn(agent_name="missing", task="task")
        assert len(agent._subagent_events) == 0


# ---------------------------------------------------------------------------
# Tests: Depth cap
# ---------------------------------------------------------------------------


class TestDepthCap:
    @pytest.mark.asyncio
    async def test_raises_max_delegation_depth_error(self):
        cfg = _remote_config(name="helper", max_depth=3)
        agent = _make_agent([cfg], delegation_depth=3)  # current_depth+1=4 > max_depth=3
        fake_transport = FakeTransport()
        tool_fn = make_delegate_tool(
            agent, transport_factory=_make_transport_factory(fake_transport)
        )
        with pytest.raises(MaxDelegationDepthError):
            await tool_fn(agent_name="helper", task="task")

    @pytest.mark.asyncio
    async def test_subagent_failed_event_emitted_before_raise(self):
        cfg = _remote_config(name="helper", max_depth=3)
        agent = _make_agent([cfg], delegation_depth=3)
        fake_transport = FakeTransport()
        tool_fn = make_delegate_tool(
            agent, transport_factory=_make_transport_factory(fake_transport)
        )
        with pytest.raises(MaxDelegationDepthError):
            await tool_fn(agent_name="helper", task="task")
        events = agent._subagent_events
        assert len(events) == 1
        assert isinstance(events[0], SubagentFailed)
        assert events[0].error_type == "MaxDelegationDepthError"
        assert events[0].agent_name == "helper"

    @pytest.mark.asyncio
    async def test_depth_at_limit_succeeds(self):
        """depth+1 == max_depth should succeed (not exceed)."""
        cfg = _remote_config(name="helper", max_depth=3)
        agent = _make_agent([cfg], delegation_depth=2)  # current_depth+1=3 == max_depth
        fake_transport = FakeTransport()
        tool_fn = make_delegate_tool(
            agent, transport_factory=_make_transport_factory(fake_transport)
        )
        raw = await tool_fn(agent_name="helper", task="task")
        result = json.loads(raw)
        assert result["agent_name"] == "helper"


# ---------------------------------------------------------------------------
# Tests: Transport failures
# ---------------------------------------------------------------------------


class TestTransportTimeout:
    @pytest.mark.asyncio
    async def test_re_raises_subagent_timeout_error(self):
        cfg = _remote_config(name="helper")
        agent = _make_agent([cfg])
        exc = SubagentTimeoutError("helper", 30.0)
        fake_transport = FakeTransport(raise_exc=exc)
        tool_fn = make_delegate_tool(
            agent, transport_factory=_make_transport_factory(fake_transport)
        )
        with pytest.raises(SubagentTimeoutError):
            await tool_fn(agent_name="helper", task="task")

    @pytest.mark.asyncio
    async def test_timeout_emits_invoked_then_failed(self):
        cfg = _remote_config(name="helper")
        agent = _make_agent([cfg])
        exc = SubagentTimeoutError("helper", 30.0)
        fake_transport = FakeTransport(raise_exc=exc)
        tool_fn = make_delegate_tool(
            agent, transport_factory=_make_transport_factory(fake_transport)
        )
        with pytest.raises(SubagentTimeoutError):
            await tool_fn(agent_name="helper", task="task")
        events = agent._subagent_events
        assert len(events) == 2
        assert isinstance(events[0], SubagentInvoked)
        assert isinstance(events[1], SubagentFailed)
        assert events[1].error_type == "SubagentTimeoutError"


class TestTransportRemoteError:
    @pytest.mark.asyncio
    async def test_re_raises_subagent_remote_error(self):
        cfg = _remote_config(name="helper")
        agent = _make_agent([cfg])
        exc = SubagentRemoteError("helper", status_code=500, detail="internal error")
        fake_transport = FakeTransport(raise_exc=exc)
        tool_fn = make_delegate_tool(
            agent, transport_factory=_make_transport_factory(fake_transport)
        )
        with pytest.raises(SubagentRemoteError):
            await tool_fn(agent_name="helper", task="task")

    @pytest.mark.asyncio
    async def test_remote_error_emits_failed_event(self):
        cfg = _remote_config(name="helper")
        agent = _make_agent([cfg])
        exc = SubagentRemoteError("helper", status_code=503, detail="unavailable")
        fake_transport = FakeTransport(raise_exc=exc)
        tool_fn = make_delegate_tool(
            agent, transport_factory=_make_transport_factory(fake_transport)
        )
        with pytest.raises(SubagentRemoteError):
            await tool_fn(agent_name="helper", task="task")
        events = agent._subagent_events
        assert isinstance(events[0], SubagentInvoked)
        assert isinstance(events[1], SubagentFailed)
        assert events[1].error_type == "SubagentRemoteError"


# ---------------------------------------------------------------------------
# Tests: Identity / header propagation
# ---------------------------------------------------------------------------


class TestIdentityInherit:
    @pytest.mark.asyncio
    async def test_forwards_auth_header_when_present(self):
        cfg = _remote_config(name="helper", identity="inherit")
        agent = _make_agent([cfg], inbound_auth="Bearer abc123")
        fake_transport = FakeTransport()
        tool_fn = make_delegate_tool(
            agent, transport_factory=_make_transport_factory(fake_transport)
        )
        await tool_fn(agent_name="helper", task="task")
        headers = fake_transport.received_headers
        assert headers["authorization"] == "Bearer abc123"
        assert headers["x-subagent-depth"] == "1"

    @pytest.mark.asyncio
    async def test_no_auth_header_when_inbound_is_none(self):
        cfg = _remote_config(name="helper", identity="inherit")
        agent = _make_agent([cfg], inbound_auth=None)
        fake_transport = FakeTransport()
        tool_fn = make_delegate_tool(
            agent, transport_factory=_make_transport_factory(fake_transport)
        )
        await tool_fn(agent_name="helper", task="task")
        headers = fake_transport.received_headers
        assert "authorization" not in headers
        assert headers["x-subagent-depth"] == "1"

    @pytest.mark.asyncio
    async def test_depth_header_incremented_from_delegation_depth(self):
        cfg = _remote_config(name="helper")
        agent = _make_agent([cfg], delegation_depth=2)
        fake_transport = FakeTransport()
        tool_fn = make_delegate_tool(
            agent, transport_factory=_make_transport_factory(fake_transport)
        )
        await tool_fn(agent_name="helper", task="task")
        assert fake_transport.received_headers["x-subagent-depth"] == "3"


class TestIdentityServiceAccount:
    @pytest.mark.asyncio
    async def test_does_not_inject_authorization_header(self, caplog):
        """v1: service-account does not inject auth — logs debug, no raise."""
        # Note: IdentityServiceAccount is incompatible with inprocess transport;
        # use remote transport here.
        cfg = SubagentConfig(
            name="secure_agent",
            description="A secure agent",
            when_to_use="Use for secure tasks",
            transport=RemoteTransportConfig(
                type="remote",
                url="http://secure:8080/v1",
            ),
            identity=IdentityServiceAccount(service_account="acct-reader"),
        )
        agent = _make_agent([cfg], inbound_auth="Bearer parent-token")
        fake_transport = FakeTransport()
        tool_fn = make_delegate_tool(
            agent, transport_factory=_make_transport_factory(fake_transport)
        )
        import logging
        with caplog.at_level(logging.DEBUG, logger="fipsagents.subagent_tool"):
            await tool_fn(agent_name="secure_agent", task="task")

        headers = fake_transport.received_headers
        # service_account identity should NOT forward the parent bearer token
        assert "authorization" not in headers
        # A debug log should have been emitted
        debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("service_account" in m for m in debug_msgs)


# ---------------------------------------------------------------------------
# Tests: Permission scope warning deduplication
# ---------------------------------------------------------------------------


class TestPermissionScopeWarning:
    def setup_method(self):
        # Clear module-level dedupe set before each test for isolation.
        _warned_permission_scope.clear()

    @pytest.mark.asyncio
    async def test_warns_once_per_agent_and_name(self, caplog):
        cfg = _remote_config(name="helper", permission_scope="readonly")
        agent = _make_agent([cfg])
        fake_transport = FakeTransport()
        factory = _make_transport_factory(fake_transport)
        tool_fn = make_delegate_tool(agent, transport_factory=factory)

        import logging
        with caplog.at_level(logging.WARNING, logger="fipsagents.subagent_tool"):
            await tool_fn(agent_name="helper", task="first call")
        first_warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "permission_scope" in r.message
        ]
        assert len(first_warnings) == 1

        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="fipsagents.subagent_tool"):
            await tool_fn(agent_name="helper", task="second call")
        second_warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "permission_scope" in r.message
        ]
        # Should NOT warn again for the same (agent, agent_name) pair.
        assert len(second_warnings) == 0

    @pytest.mark.asyncio
    async def test_different_agents_each_get_own_warning(self, caplog):
        cfg = _remote_config(name="helper", permission_scope="readonly")
        agent1 = _make_agent([cfg])
        agent2 = _make_agent([cfg])
        factory = _make_transport_factory(FakeTransport())

        import logging
        with caplog.at_level(logging.WARNING, logger="fipsagents.subagent_tool"):
            await make_delegate_tool(agent1, transport_factory=factory)(
                agent_name="helper", task="task"
            )
            await make_delegate_tool(agent2, transport_factory=factory)(
                agent_name="helper", task="task"
            )
        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "permission_scope" in r.message
        ]
        assert len(warnings) == 2

    @pytest.mark.asyncio
    async def test_no_warning_when_permission_scope_is_none(self, caplog):
        cfg = _remote_config(name="helper", permission_scope=None)
        agent = _make_agent([cfg])
        factory = _make_transport_factory(FakeTransport())
        tool_fn = make_delegate_tool(agent, transport_factory=factory)

        import logging
        with caplog.at_level(logging.WARNING, logger="fipsagents.subagent_tool"):
            await tool_fn(agent_name="helper", task="task")
        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "permission_scope" in r.message
        ]
        assert len(warnings) == 0


# ---------------------------------------------------------------------------
# Tests: Defensive missing attributes
# ---------------------------------------------------------------------------


class TestDefensiveMissingAttributes:
    @pytest.mark.asyncio
    async def test_no_crash_when_events_missing(self):
        cfg = _remote_config(name="helper")
        # Agent without _subagent_events
        agent = types.SimpleNamespace(
            subagents={"helper": cfg},
            _subagent_token_usage=[],
            _delegation_depth=0,
            _inbound_auth_header=None,
        )
        fake_transport = FakeTransport()
        tool_fn = make_delegate_tool(
            agent, transport_factory=_make_transport_factory(fake_transport)
        )
        raw = await tool_fn(agent_name="helper", task="task")
        result = json.loads(raw)
        assert result["agent_name"] == "helper"

    @pytest.mark.asyncio
    async def test_no_crash_when_token_usage_missing(self):
        cfg = _remote_config(name="helper")
        agent = types.SimpleNamespace(
            subagents={"helper": cfg},
            _subagent_events=[],
            _delegation_depth=0,
            _inbound_auth_header=None,
        )
        fake_transport = FakeTransport()
        tool_fn = make_delegate_tool(
            agent, transport_factory=_make_transport_factory(fake_transport)
        )
        raw = await tool_fn(agent_name="helper", task="task")
        result = json.loads(raw)
        assert result["content"] == "Done!"

    @pytest.mark.asyncio
    async def test_no_crash_when_inbound_auth_missing(self):
        cfg = _remote_config(name="helper", identity="inherit")
        agent = types.SimpleNamespace(
            subagents={"helper": cfg},
            _subagent_events=[],
            _subagent_token_usage=[],
            _delegation_depth=0,
            # _inbound_auth_header is absent
        )
        fake_transport = FakeTransport()
        tool_fn = make_delegate_tool(
            agent, transport_factory=_make_transport_factory(fake_transport)
        )
        raw = await tool_fn(agent_name="helper", task="task")
        assert json.loads(raw)["agent_name"] == "helper"
        # No authorization header should have been added
        assert "authorization" not in (fake_transport.received_headers or {})

    @pytest.mark.asyncio
    async def test_no_crash_when_delegation_depth_missing(self):
        cfg = _remote_config(name="helper", max_depth=3)
        agent = types.SimpleNamespace(
            subagents={"helper": cfg},
            _subagent_events=[],
            _subagent_token_usage=[],
            _inbound_auth_header=None,
            # _delegation_depth absent — should default to 0
        )
        fake_transport = FakeTransport()
        tool_fn = make_delegate_tool(
            agent, transport_factory=_make_transport_factory(fake_transport)
        )
        raw = await tool_fn(agent_name="helper", task="task")
        assert json.loads(raw)["agent_name"] == "helper"
        # Depth 0+1=1, well within max_depth 3
        assert fake_transport.received_headers["x-subagent-depth"] == "1"


# ---------------------------------------------------------------------------
# Tests: Tool metadata / description
# ---------------------------------------------------------------------------


class TestToolMetadata:
    def test_description_contains_subagent_name(self):
        cfg = _remote_config(name="research_helper", when_to_use="Use for research queries")
        agent = _make_agent([cfg])
        tool_fn = make_delegate_tool(agent)
        meta = getattr(tool_fn, _TOOL_MARKER)
        assert "research_helper" in meta.description

    def test_description_contains_when_to_use(self):
        cfg = _remote_config(name="helper", when_to_use="Use when user asks policy questions")
        agent = _make_agent([cfg])
        tool_fn = make_delegate_tool(agent)
        meta = getattr(tool_fn, _TOOL_MARKER)
        assert "Use when user asks policy questions" in meta.description

    def test_description_contains_all_subagents(self):
        cfgs = [
            _remote_config(name="agent_a", when_to_use="Use for A tasks"),
            _remote_config(name="agent_b", when_to_use="Use for B tasks"),
        ]
        agent = _make_agent(cfgs)
        tool_fn = make_delegate_tool(agent)
        meta = getattr(tool_fn, _TOOL_MARKER)
        assert "agent_a" in meta.description
        assert "Use for A tasks" in meta.description
        assert "agent_b" in meta.description
        assert "Use for B tasks" in meta.description

    def test_tool_name_is_delegate_to_agent(self):
        cfg = _remote_config(name="helper")
        agent = _make_agent([cfg])
        tool_fn = make_delegate_tool(agent)
        meta = getattr(tool_fn, _TOOL_MARKER)
        assert meta.name == "delegate_to_agent"

    def test_tool_visibility_is_both(self):
        cfg = _remote_config(name="helper")
        agent = _make_agent([cfg])
        tool_fn = make_delegate_tool(agent)
        meta = getattr(tool_fn, _TOOL_MARKER)
        assert meta.visibility == "both"

    def test_empty_subagents_gives_valid_tool(self):
        agent = _make_agent([])
        tool_fn = make_delegate_tool(agent)
        meta = getattr(tool_fn, _TOOL_MARKER)
        assert meta.name == "delegate_to_agent"
        # Description still valid; Step 6 won't register it in this case.
        assert "delegate" in meta.description.lower()
