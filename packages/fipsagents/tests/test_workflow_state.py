"""Tests for fipsagents.workflow.state — WorkflowState base class and END sentinel."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fipsagents.workflow.state import END, WorkflowState


class SampleState(WorkflowState):
    query: str
    result: str = ""


class TestEND:
    def test_end_is_expected_string(self):
        assert END == "__END__"


class TestWorkflowState:
    def test_subclass_with_defaults(self):
        state = SampleState(query="hello")
        assert state.query == "hello"
        assert state.result == ""

    def test_subclass_with_all_fields(self):
        state = SampleState(query="hello", result="world")
        assert state.query == "hello"
        assert state.result == "world"

    def test_model_copy_produces_new_instance(self):
        original = SampleState(query="hello", result="a")
        updated = original.model_copy(update={"result": "b"})

        assert updated.result == "b"
        assert original.result == "a"
        assert original is not updated

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError, match="extra"):
            SampleState(query="hello", bogus="nope")

    def test_roundtrip_json_serialization(self):
        state = SampleState(query="ping", result="pong")
        json_str = state.model_dump_json()
        restored = SampleState.model_validate_json(json_str)

        assert restored.query == state.query
        assert restored.result == state.result
        assert restored == state
