"""Tests for fipsagents.subagents.types."""

from __future__ import annotations

import pytest

from fipsagents.subagents import (
    MaxDelegationDepthError,
    SubagentCrashedError,
    SubagentError,
    SubagentRemoteError,
    SubagentResult,
    SubagentTimeoutError,
)


class TestSubagentResult:
    """Test SubagentResult dataclass."""

    def test_construct_with_all_fields(self) -> None:
        """SubagentResult accepts all fields."""
        result = SubagentResult(
            agent_name="helper",
            content="Analysis complete.",
            tokens_used={"input": 100, "output": 50, "cached": 0},
            tool_calls_made=2,
            cost_usd=0.001234,
            span_id="span_abc123",
            finish_reason="stop",
        )
        assert result.agent_name == "helper"
        assert result.content == "Analysis complete."
        assert result.tokens_used == {"input": 100, "output": 50, "cached": 0}
        assert result.tool_calls_made == 2
        assert result.cost_usd == 0.001234
        assert result.span_id == "span_abc123"
        assert result.finish_reason == "stop"

    def test_construct_with_defaults(self) -> None:
        """SubagentResult uses sensible defaults for optional fields."""
        result = SubagentResult(
            agent_name="helper",
            content="Done.",
            tokens_used={"input": 50, "output": 25, "cached": 0},
            tool_calls_made=0,
            cost_usd=0.0005,
        )
        assert result.span_id is None
        assert result.finish_reason == "stop"

    def test_requires_agent_name(self) -> None:
        """SubagentResult requires agent_name."""
        with pytest.raises(TypeError):
            SubagentResult(
                content="Text",
                tokens_used={"input": 10, "output": 5, "cached": 0},
                tool_calls_made=0,
                cost_usd=0.0,
            )

    def test_requires_content(self) -> None:
        """SubagentResult requires content."""
        with pytest.raises(TypeError):
            SubagentResult(
                agent_name="helper",
                tokens_used={"input": 10, "output": 5, "cached": 0},
                tool_calls_made=0,
                cost_usd=0.0,
            )

    def test_requires_tokens_used(self) -> None:
        """SubagentResult requires tokens_used."""
        with pytest.raises(TypeError):
            SubagentResult(
                agent_name="helper",
                content="Text",
                tool_calls_made=0,
                cost_usd=0.0,
            )

    def test_requires_tool_calls_made(self) -> None:
        """SubagentResult requires tool_calls_made."""
        with pytest.raises(TypeError):
            SubagentResult(
                agent_name="helper",
                content="Text",
                tokens_used={"input": 10, "output": 5, "cached": 0},
                cost_usd=0.0,
            )

    def test_requires_cost_usd(self) -> None:
        """SubagentResult requires cost_usd."""
        with pytest.raises(TypeError):
            SubagentResult(
                agent_name="helper",
                content="Text",
                tokens_used={"input": 10, "output": 5, "cached": 0},
                tool_calls_made=0,
            )


class TestSubagentError:
    """Test SubagentError base class."""

    def test_is_exception(self) -> None:
        """SubagentError is an Exception."""
        err = SubagentError("helper", "something went wrong")
        assert isinstance(err, Exception)

    def test_stores_agent_name(self) -> None:
        """SubagentError stores agent_name."""
        err = SubagentError("helper", "crashed")
        assert err.agent_name == "helper"

    def test_message_format(self) -> None:
        """SubagentError message includes agent name and message."""
        err = SubagentError("research_agent", "timed out")
        assert str(err) == "subagent 'research_agent': timed out"

    def test_inheritable(self) -> None:
        """SubagentError can be caught as a subclass of Exception."""
        err = SubagentError("helper", "failed")
        try:
            raise err
        except SubagentError:
            pass


class TestSubagentTimeoutError:
    """Test SubagentTimeoutError."""

    def test_is_subagent_error(self) -> None:
        """SubagentTimeoutError is a SubagentError."""
        err = SubagentTimeoutError("helper", 30.0)
        assert isinstance(err, SubagentError)

    def test_stores_timeout_seconds(self) -> None:
        """SubagentTimeoutError stores timeout_seconds."""
        err = SubagentTimeoutError("helper", 45.5)
        assert err.timeout_seconds == 45.5

    def test_message_format(self) -> None:
        """SubagentTimeoutError message includes timeout duration."""
        err = SubagentTimeoutError("lookup_service", 60.0)
        assert str(err) == "subagent 'lookup_service': timed out after 60.0s"

    def test_fractional_seconds(self) -> None:
        """SubagentTimeoutError formats fractional seconds."""
        err = SubagentTimeoutError("helper", 15.3456)
        assert "15.3s" in str(err)

    def test_catchable_as_subagent_error(self) -> None:
        """SubagentTimeoutError can be caught as SubagentError."""
        err = SubagentTimeoutError("helper", 30.0)
        try:
            raise err
        except SubagentError:
            pass


class TestSubagentRemoteError:
    """Test SubagentRemoteError."""

    def test_is_subagent_error(self) -> None:
        """SubagentRemoteError is a SubagentError."""
        err = SubagentRemoteError(
            "remote_agent",
            status_code=500,
            detail="Internal server error",
        )
        assert isinstance(err, SubagentError)

    def test_stores_status_code(self) -> None:
        """SubagentRemoteError stores status_code."""
        err = SubagentRemoteError(
            "helper",
            status_code=502,
            detail="Bad gateway",
        )
        assert err.status_code == 502

    def test_stores_detail(self) -> None:
        """SubagentRemoteError stores detail."""
        err = SubagentRemoteError(
            "helper",
            status_code=503,
            detail="Service unavailable",
        )
        assert err.detail == "Service unavailable"

    def test_message_with_status_code(self) -> None:
        """SubagentRemoteError includes status code in message."""
        err = SubagentRemoteError(
            "api_client",
            status_code=500,
            detail="Internal error",
        )
        assert "HTTP 500:" in str(err)
        assert "Internal error" in str(err)

    def test_message_without_status_code(self) -> None:
        """SubagentRemoteError handles None status_code gracefully."""
        err = SubagentRemoteError(
            "connector",
            status_code=None,
            detail="connection reset by peer",
        )
        # Should not have "HTTP None:" prefix
        assert "HTTP None:" not in str(err)
        assert "connection reset by peer" in str(err)

    def test_various_status_codes(self) -> None:
        """SubagentRemoteError works with different status codes."""
        for code in [400, 403, 404, 500, 502, 503]:
            err = SubagentRemoteError("agent", status_code=code, detail="error")
            assert f"HTTP {code}:" in str(err)

    def test_catchable_as_subagent_error(self) -> None:
        """SubagentRemoteError can be caught as SubagentError."""
        err = SubagentRemoteError("agent", status_code=500, detail="error")
        try:
            raise err
        except SubagentError:
            pass


class TestMaxDelegationDepthError:
    """Test MaxDelegationDepthError."""

    def test_is_subagent_error(self) -> None:
        """MaxDelegationDepthError is a SubagentError."""
        err = MaxDelegationDepthError("helper", depth=4, max_depth=3)
        assert isinstance(err, SubagentError)

    def test_stores_depth(self) -> None:
        """MaxDelegationDepthError stores depth."""
        err = MaxDelegationDepthError("agent", depth=5, max_depth=3)
        assert err.depth == 5

    def test_stores_max_depth(self) -> None:
        """MaxDelegationDepthError stores max_depth."""
        err = MaxDelegationDepthError("agent", depth=2, max_depth=3)
        assert err.max_depth == 3

    def test_message_format(self) -> None:
        """MaxDelegationDepthError message includes depth values."""
        err = MaxDelegationDepthError("delegator", depth=4, max_depth=3)
        assert "depth 4" in str(err)
        assert "max_depth 3" in str(err)

    def test_various_depths(self) -> None:
        """MaxDelegationDepthError works with various depth values."""
        for depth, max_d in [(1, 0), (10, 5), (100, 10)]:
            err = MaxDelegationDepthError("agent", depth=depth, max_depth=max_d)
            assert err.depth == depth
            assert err.max_depth == max_d

    def test_catchable_as_subagent_error(self) -> None:
        """MaxDelegationDepthError can be caught as SubagentError."""
        err = MaxDelegationDepthError("agent", depth=5, max_depth=3)
        try:
            raise err
        except SubagentError:
            pass


class TestSubagentCrashedError:
    """Test SubagentCrashedError."""

    def test_is_subagent_error(self) -> None:
        """SubagentCrashedError is a SubagentError."""
        orig = ValueError("bad value")
        err = SubagentCrashedError("helper", original=orig)
        assert isinstance(err, SubagentError)

    def test_stores_original_exception(self) -> None:
        """SubagentCrashedError stores the original exception."""
        orig = RuntimeError("crash!")
        err = SubagentCrashedError("agent", original=orig)
        assert err.original is orig

    def test_message_includes_exception_type(self) -> None:
        """SubagentCrashedError message includes exception type."""
        orig = ValueError("bad input")
        err = SubagentCrashedError("processor", original=orig)
        assert "ValueError" in str(err)

    def test_message_includes_exception_message(self) -> None:
        """SubagentCrashedError message includes exception message."""
        orig = RuntimeError("something failed")
        err = SubagentCrashedError("agent", original=orig)
        assert "something failed" in str(err)

    def test_preserves_exception_info(self) -> None:
        """SubagentCrashedError preserves the original exception."""
        orig = KeyError("missing_key")
        err = SubagentCrashedError("lookup", original=orig)
        assert isinstance(err.original, KeyError)

    def test_with_various_exception_types(self) -> None:
        """SubagentCrashedError works with various exception types."""
        exceptions = [
            ValueError("value error"),
            RuntimeError("runtime error"),
            KeyError("key error"),
            TypeError("type error"),
            AttributeError("attribute error"),
        ]
        for orig in exceptions:
            err = SubagentCrashedError("agent", original=orig)
            assert isinstance(err.original, type(orig))
            assert type(orig).__name__ in str(err)

    def test_catchable_as_subagent_error(self) -> None:
        """SubagentCrashedError can be caught as SubagentError."""
        orig = RuntimeError("failed")
        err = SubagentCrashedError("agent", original=orig)
        try:
            raise err
        except SubagentError:
            pass


class TestSubagentErrorHierarchy:
    """Test the error class hierarchy."""

    def test_all_errors_are_subagent_errors(self) -> None:
        """All error types are subclasses of SubagentError."""
        errors = [
            SubagentTimeoutError("agent", 30.0),
            SubagentRemoteError("agent", status_code=500, detail="error"),
            MaxDelegationDepthError("agent", depth=5, max_depth=3),
            SubagentCrashedError("agent", original=RuntimeError("crash")),
        ]
        for err in errors:
            assert isinstance(err, SubagentError)

    def test_all_errors_are_exceptions(self) -> None:
        """All error types are subclasses of Exception."""
        errors = [
            SubagentTimeoutError("agent", 30.0),
            SubagentRemoteError("agent", status_code=500, detail="error"),
            MaxDelegationDepthError("agent", depth=5, max_depth=3),
            SubagentCrashedError("agent", original=RuntimeError("crash")),
        ]
        for err in errors:
            assert isinstance(err, Exception)

    def test_catch_all_as_subagent_error(self) -> None:
        """All subagent errors can be caught as SubagentError."""
        errors = [
            SubagentTimeoutError("agent1", 30.0),
            SubagentRemoteError("agent2", status_code=500, detail="error"),
            MaxDelegationDepthError("agent3", depth=5, max_depth=3),
            SubagentCrashedError("agent4", original=RuntimeError("crash")),
        ]
        for err in errors:
            try:
                raise err
            except SubagentError as caught:
                assert caught.agent_name is not None


class TestImports:
    """Test that imports work correctly."""

    def test_import_from_subagents(self) -> None:
        """Can import types from fipsagents.subagents."""
        from fipsagents.subagents import (
            MaxDelegationDepthError as MDDError,
            SubagentCrashedError as CEError,
            SubagentError as SError,
            SubagentRemoteError as RError,
            SubagentResult,
            SubagentTimeoutError as TError,
        )
        assert SubagentResult is not None
        assert SError is SubagentError
        assert TError is SubagentTimeoutError
        assert RError is SubagentRemoteError
        assert MDDError is MaxDelegationDepthError
        assert CEError is SubagentCrashedError

    def test_import_from_subagents_types(self) -> None:
        """Can import types from fipsagents.subagents.types."""
        from fipsagents.subagents.types import (
            MaxDelegationDepthError,
            SubagentCrashedError,
            SubagentError,
            SubagentRemoteError,
            SubagentResult,
            SubagentTimeoutError,
        )
        assert SubagentResult is not None
        assert SubagentError is not None
        assert SubagentTimeoutError is not None
        assert SubagentRemoteError is not None
        assert MaxDelegationDepthError is not None
        assert SubagentCrashedError is not None
