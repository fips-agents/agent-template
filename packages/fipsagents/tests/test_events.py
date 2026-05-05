"""Tests for fipsagents.baseagent.events — typed StreamEvent variants."""

from __future__ import annotations

from fipsagents.baseagent.events import (
    ContentDelta,
    GuardrailFiredEvent,
    ReasoningDelta,
    StreamComplete,
    StreamEvent,
    StreamMetrics,
    ToolCallDelta,
    ToolResultEvent,
)


class TestGuardrailFiredEvent:
    def test_minimal_construction(self):
        ev = GuardrailFiredEvent(shield_id="content_safety", action="blocked")
        assert ev.shield_id == "content_safety"
        assert ev.action == "blocked"
        assert ev.category is None
        assert ev.message is None

    def test_full_construction(self):
        ev = GuardrailFiredEvent(
            shield_id="prompt_guard",
            action="warned",
            category="jailbreak",
            message="Prompt resembles a known jailbreak pattern.",
        )
        assert ev.category == "jailbreak"
        assert ev.message == "Prompt resembles a known jailbreak pattern."

    def test_is_member_of_stream_event_union(self):
        # Structural check: assigning to the union type must compile and round-trip.
        ev: StreamEvent = GuardrailFiredEvent(
            shield_id="content_safety", action="blocked"
        )
        assert isinstance(ev, GuardrailFiredEvent)


class TestStreamEventUnion:
    def test_all_variants_are_assignable(self):
        # Lightweight smoke check — guards against accidentally dropping a
        # variant from the union when adding new ones.
        events: list[StreamEvent] = [
            ReasoningDelta(content="thinking"),
            ToolCallDelta(index=0, call_id="c", name="t", arguments_delta="{}"),
            ToolResultEvent(call_id="c", name="t", content="ok"),
            ContentDelta(content="hello"),
            GuardrailFiredEvent(shield_id="content_safety", action="blocked"),
            StreamComplete(finish_reason="stop", metrics=StreamMetrics()),
        ]
        assert len(events) == 6
