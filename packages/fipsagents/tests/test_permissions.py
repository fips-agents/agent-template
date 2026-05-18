"""Tests for PermissionSource ABC, NullPermissionSource, StaticPermissionSource."""
import pytest
from fipsagents.server.permissions import (
    NullPermissionSource,
    PermissionDecision,
    PermissionRule,
    StaticPermissionSource,
    create_permission_source,
)


class TestPermissionDecision:
    def test_fields(self):
        d = PermissionDecision(action="allow", tool="search")
        assert d.action == "allow"
        assert d.tool == "search"
        assert d.rule_id is None
        assert d.scope is None
        assert d.reason is None


class TestNullPermissionSource:
    @pytest.mark.asyncio
    async def test_allows_everything(self):
        src = NullPermissionSource()
        d = await src.resolve("any_tool")
        assert d.action == "allow"
        assert d.tool == "any_tool"

    @pytest.mark.asyncio
    async def test_close_is_noop(self):
        src = NullPermissionSource()
        await src.close()


class TestStaticPermissionSource:
    @pytest.mark.asyncio
    async def test_exact_match_deny(self):
        src = StaticPermissionSource(
            rules=[PermissionRule(id="r1", tool="dangerous_tool", action="deny")]
        )
        d = await src.resolve("dangerous_tool")
        assert d.action == "deny"
        assert d.rule_id == "r1"

    @pytest.mark.asyncio
    async def test_exact_match_ask(self):
        src = StaticPermissionSource(
            rules=[PermissionRule(id="r2", tool="sensitive", action="ask")]
        )
        d = await src.resolve("sensitive")
        assert d.action == "ask"

    @pytest.mark.asyncio
    async def test_wildcard_matches_all(self):
        src = StaticPermissionSource(
            rules=[PermissionRule(id="r3", tool="*", action="deny")]
        )
        d = await src.resolve("anything")
        assert d.action == "deny"

    @pytest.mark.asyncio
    async def test_first_match_wins(self):
        src = StaticPermissionSource(
            rules=[
                PermissionRule(id="r1", tool="tool_a", action="deny"),
                PermissionRule(id="r2", tool="tool_a", action="allow"),
            ]
        )
        d = await src.resolve("tool_a")
        assert d.action == "deny"
        assert d.rule_id == "r1"

    @pytest.mark.asyncio
    async def test_scope_filter(self):
        src = StaticPermissionSource(
            rules=[
                PermissionRule(id="r1", tool="tool_a", action="deny", scope="admin"),
                PermissionRule(id="r2", tool="tool_a", action="allow"),
            ]
        )
        # Without matching scope, r1 is skipped
        d = await src.resolve("tool_a", scope="user")
        assert d.action == "allow"
        assert d.rule_id == "r2"

        # With matching scope, r1 matches
        d2 = await src.resolve("tool_a", scope="admin")
        assert d2.action == "deny"
        assert d2.rule_id == "r1"

    @pytest.mark.asyncio
    async def test_default_when_no_match(self):
        src = StaticPermissionSource(
            rules=[PermissionRule(id="r1", tool="specific", action="deny")],
            default_action="ask",
        )
        d = await src.resolve("other_tool")
        assert d.action == "ask"

    @pytest.mark.asyncio
    async def test_default_action_allow(self):
        src = StaticPermissionSource(rules=[])
        d = await src.resolve("any_tool")
        assert d.action == "allow"

    @pytest.mark.asyncio
    async def test_fnmatch_glob_matching(self):
        src = StaticPermissionSource(
            rules=[PermissionRule(id="r1", tool="kubectl_*", action="deny")]
        )
        d = await src.resolve("kubectl_exec")
        assert d.action == "deny"
        assert d.rule_id == "r1"

    @pytest.mark.asyncio
    async def test_fnmatch_no_false_positive(self):
        src = StaticPermissionSource(
            rules=[PermissionRule(id="r1", tool="kubectl_*", action="deny")]
        )
        d = await src.resolve("docker_exec")
        assert d.action == "allow"

    @pytest.mark.asyncio
    async def test_reason_propagation(self):
        src = StaticPermissionSource(
            rules=[PermissionRule(
                id="r1", tool="dangerous", action="deny",
                reason="Too dangerous",
            )]
        )
        d = await src.resolve("dangerous")
        assert d.reason == "Too dangerous"

    @pytest.mark.asyncio
    async def test_reason_none_when_unset(self):
        src = StaticPermissionSource(
            rules=[PermissionRule(id="r1", tool="tool_a", action="allow")]
        )
        d = await src.resolve("tool_a")
        assert d.reason is None


class TestCreatePermissionSource:
    def test_none_returns_null(self):
        src = create_permission_source(None)
        assert isinstance(src, NullPermissionSource)

    def test_null_string_returns_null(self):
        src = create_permission_source("null")
        assert isinstance(src, NullPermissionSource)

    def test_static_with_rules(self):
        src = create_permission_source(
            "static",
            rules=[{"tool": "search", "action": "allow"}],
        )
        assert isinstance(src, StaticPermissionSource)

    def test_static_with_reason(self):
        src = create_permission_source(
            "static",
            rules=[{"tool": "search", "action": "deny", "reason": "blocked"}],
        )
        assert isinstance(src, StaticPermissionSource)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown permission source"):
            create_permission_source("nonexistent")
