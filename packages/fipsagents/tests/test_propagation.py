"""Tests for W3C Trace Context propagation."""

import pytest

from fipsagents.server.propagation import (
    extract_trace_context,
    inject_trace_context,
)


class TestExtractTraceContext:
    def test_extracts_valid_traceparent(self):
        headers = {
            "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        }
        ctx = extract_trace_context(headers)
        assert ctx is not None
        assert ctx.trace_id == "0af7651916cd43dd8448eb211c80319c"
        assert ctx.parent_span_id == "b7ad6b7169203331"
        assert ctx.trace_flags == "01"

    def test_returns_none_for_missing_header(self):
        assert extract_trace_context({}) is None

    def test_returns_none_for_empty_header(self):
        assert extract_trace_context({"traceparent": ""}) is None

    def test_returns_none_for_malformed_header(self):
        assert extract_trace_context({"traceparent": "not-valid"}) is None

    def test_returns_none_for_wrong_length(self):
        # trace_id too short
        assert extract_trace_context({
            "traceparent": "00-0af765-b7ad6b-01"
        }) is None

    def test_strips_whitespace(self):
        headers = {
            "traceparent": "  00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01  "
        }
        ctx = extract_trace_context(headers)
        assert ctx is not None
        assert ctx.trace_id == "0af7651916cd43dd8448eb211c80319c"


class TestInjectTraceContext:
    def test_produces_valid_traceparent(self):
        headers = inject_trace_context("trace_abc123", "span_def456")
        tp = headers["traceparent"]
        # Format: 00-{32hex}-{16hex}-01
        parts = tp.split("-")
        assert len(parts) == 4
        assert parts[0] == "00"
        assert len(parts[1]) == 32
        assert len(parts[2]) == 16
        assert parts[3] == "01"

    def test_deterministic(self):
        h1 = inject_trace_context("trace_a", "span_b")
        h2 = inject_trace_context("trace_a", "span_b")
        assert h1 == h2

    def test_different_inputs_different_outputs(self):
        h1 = inject_trace_context("trace_a", "span_b")
        h2 = inject_trace_context("trace_c", "span_d")
        assert h1 != h2


class TestTraceCollectorWithParentContext:
    @pytest.mark.asyncio
    async def test_parent_trace_id_is_used(self):
        from fipsagents.server.collector import TraceCollector
        from fipsagents.server.tracing import NullTraceStore

        collector = TraceCollector(
            NullTraceStore(),
            parent_trace_id="inherited-trace-id",
            parent_span_id="inherited-span-id",
        )
        assert collector.trace_id == "inherited-trace-id"

    @pytest.mark.asyncio
    async def test_root_span_has_parent(self):
        from fipsagents.server.collector import TraceCollector
        from fipsagents.server.tracing import NullTraceStore

        collector = TraceCollector(
            NullTraceStore(),
            parent_trace_id="upstream-trace",
            parent_span_id="upstream-span",
        )
        collector.begin_request({"model": "test"})
        # The root request span should have the upstream span as parent
        root = collector._request_span
        assert root is not None
        assert root.parent_span_id == "upstream-span"

    @pytest.mark.asyncio
    async def test_no_parent_context(self):
        from fipsagents.server.collector import TraceCollector
        from fipsagents.server.tracing import NullTraceStore

        collector = TraceCollector(NullTraceStore())
        collector.begin_request()
        root = collector._request_span
        assert root is not None
        assert root.parent_span_id is None


class TestRemoteNodeTraceHeaders:
    def test_set_trace_context_produces_headers(self):
        from fipsagents.workflow.remote_node import RemoteNode
        node = RemoteNode("test", endpoint="http://localhost:8080")
        node.set_trace_context("trace_123", "span_456")
        assert "traceparent" in node._trace_headers

    def test_no_trace_context_no_headers(self):
        from fipsagents.workflow.remote_node import RemoteNode
        node = RemoteNode("test", endpoint="http://localhost:8080")
        assert node._trace_headers == {}
