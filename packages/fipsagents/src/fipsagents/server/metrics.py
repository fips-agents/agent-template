"""Prometheus metrics collector for the server layer.

Follows the TraceCollector observer pattern -- wraps async event streams,
yields events unchanged, and records Prometheus metrics as a side effect.
"""

from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator

from fipsagents.baseagent.events import (
    StreamComplete,
    StreamEvent,
    ToolResultEvent,
)

logger = logging.getLogger(__name__)

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Histogram,
        generate_latest,
    )

    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False


class MetricsCollector:
    """Records Prometheus metrics from StreamEvents.

    Requires ``prometheus_client`` (install via ``pip install 'fipsagents[metrics]'``).
    """

    def __init__(self, *, registry: Any = None) -> None:
        if not _HAS_PROMETHEUS:
            raise ImportError(
                "prometheus_client is required for MetricsCollector. "
                "Install with: pip install 'fipsagents[metrics]'"
            )
        self._registry = registry or CollectorRegistry()

        self.requests_total = Counter(
            "agent_requests_total",
            "Total chat completion requests",
            labelnames=["model", "status", "stream"],
            registry=self._registry,
        )
        self.request_duration = Histogram(
            "agent_request_duration_seconds",
            "Chat completion request duration",
            labelnames=["model"],
            registry=self._registry,
        )
        self.model_call_duration = Histogram(
            "agent_model_call_duration_seconds",
            "Individual model call duration",
            labelnames=["model"],
            registry=self._registry,
        )
        self.tool_calls_total = Counter(
            "agent_tool_call_total",
            "Total tool calls",
            labelnames=["tool_name", "status"],
            registry=self._registry,
        )
        self.tokens_total = Counter(
            "agent_tokens_total",
            "Total tokens processed",
            labelnames=["model", "direction"],
            registry=self._registry,
        )

    async def observe(
        self,
        events: AsyncIterator[StreamEvent],
        *,
        model: str,
    ) -> AsyncIterator[StreamEvent]:
        """Wrap an event stream, recording metrics as events flow through."""
        async for event in events:
            if isinstance(event, ToolResultEvent):
                status = "error" if event.is_error else "ok"
                self.tool_calls_total.labels(
                    tool_name=event.name,
                    status=status,
                ).inc()
            elif isinstance(event, StreamComplete):
                m = event.metrics
                if m.prompt_tokens is not None:
                    self.tokens_total.labels(
                        model=model,
                        direction="prompt",
                    ).inc(m.prompt_tokens)
                if m.completion_tokens is not None:
                    self.tokens_total.labels(
                        model=model,
                        direction="completion",
                    ).inc(m.completion_tokens)
                if m.total_time > 0:
                    self.model_call_duration.labels(model=model).observe(
                        m.total_time,
                    )
            yield event

    def record_request_start(self) -> float:
        """Mark the start of a request. Returns monotonic timestamp."""
        return time.monotonic()

    def record_request_end(
        self,
        model: str,
        stream: bool,
        status: str,
        start_time: float,
    ) -> None:
        """Record request completion metrics."""
        duration = time.monotonic() - start_time
        self.requests_total.labels(
            model=model,
            status=status,
            stream=str(stream).lower(),
        ).inc()
        self.request_duration.labels(model=model).observe(duration)

    def generate_metrics(self) -> bytes:
        """Return Prometheus text exposition format."""
        return generate_latest(self._registry)


class NullMetricsCollector:
    """No-op metrics collector when prometheus_client is not installed."""

    async def observe(
        self,
        events: AsyncIterator[StreamEvent],
        *,
        model: str,
    ) -> AsyncIterator[StreamEvent]:
        async for event in events:
            yield event

    def record_request_start(self) -> float:
        return time.monotonic()

    def record_request_end(
        self,
        model: str,
        stream: bool,
        status: str,
        start_time: float,
    ) -> None:
        pass

    def generate_metrics(self) -> bytes:
        return b""


def create_metrics_collector(
    enabled: bool = False,
) -> MetricsCollector | NullMetricsCollector:
    """Create a metrics collector based on config."""
    if not enabled:
        return NullMetricsCollector()
    if not _HAS_PROMETHEUS:
        logger.warning(
            "Metrics enabled but prometheus_client not installed. "
            "Install with: pip install 'fipsagents[metrics]'"
        )
        return NullMetricsCollector()
    return MetricsCollector()
