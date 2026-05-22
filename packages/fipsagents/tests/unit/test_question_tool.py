"""Tests for the ``ask_user`` tool factory.

Verifies the full lifecycle of :func:`make_question_tool`:

- QuestionOption model validation and defaults
- QuestionAnswer model
- Question ID generation (format and uniqueness)
- Event emission defensive behavior
- Tool metadata
- Happy path (pending state, event sequence, JSON output)
- Multiple selection and custom text
"""

from __future__ import annotations

import json
import re
import types

import pytest

from fipsagents.baseagent.events import QuestionAsked
from fipsagents.baseagent.tools.question import (
    QuestionAnswer,
    QuestionOption,
    _emit_question,
    _generate_question_id,
    make_question_tool,
)
from fipsagents.baseagent.tools import _TOOL_MARKER


class TestQuestionOption:
    def test_defaults_value_to_label(self):
        opt = QuestionOption(label="Yes")
        assert opt.label == "Yes"
        assert opt.value == "Yes"
        assert opt.description is None

    def test_explicit_value_overrides_default(self):
        opt = QuestionOption(label="Yes", value="y", description="Affirmative")
        assert opt.label == "Yes"
        assert opt.value == "y"
        assert opt.description == "Affirmative"


class TestQuestionAnswer:
    def test_single_selection(self):
        ans = QuestionAnswer(selected=["yes"])
        assert ans.selected == ["yes"]
        assert ans.custom_text is None

    def test_multiple_selection_with_custom(self):
        ans = QuestionAnswer(selected=["a", "b"], custom_text="Also this")
        assert ans.selected == ["a", "b"]
        assert ans.custom_text == "Also this"


class TestGenerateQuestionId:
    def test_id_format(self):
        qid = _generate_question_id()
        assert qid.startswith("q_")
        # Format: q_{12 hex chars}_{12 hex chars}
        pattern = r"^q_[0-9a-f]{12}_[0-9a-f]{12}$"
        assert re.match(pattern, qid), f"ID {qid} does not match expected format"

    def test_uniqueness(self):
        id1 = _generate_question_id()
        id2 = _generate_question_id()
        assert id1 != id2


class TestEmitQuestion:
    def test_appends_to_question_events(self):
        agent = types.SimpleNamespace(_question_events=[])
        evt = QuestionAsked(
            question_id="q_123",
            question_text="Which?",
            options=[{"label": "A", "value": "A", "description": None}],
            multiple=False,
            allow_custom=False,
        )
        _emit_question(agent, evt)
        assert len(agent._question_events) == 1
        assert agent._question_events[0] is evt

    def test_no_crash_when_attr_missing(self):
        agent = types.SimpleNamespace()
        evt = QuestionAsked(
            question_id="q_456",
            question_text="What?",
            options=[],
            multiple=False,
            allow_custom=False,
        )
        _emit_question(agent, evt)
        assert not hasattr(agent, "_question_events")


class TestMakeQuestionTool:
    def test_tool_metadata(self):
        agent = types.SimpleNamespace()
        tool_fn = make_question_tool(agent)
        meta = getattr(tool_fn, _TOOL_MARKER)
        assert meta.name == "ask_user"
        assert meta.visibility == "llm_only"
        assert "operator" in meta.description.lower()

    @pytest.mark.asyncio
    async def test_happy_path(self):
        agent = types.SimpleNamespace(_question_pending=None, _question_events=[])
        tool_fn = make_question_tool(agent)
        result_json = await tool_fn(
            prompt="Which color?",
            options=[{"label": "Red"}, {"label": "Blue", "value": "b"}],
        )
        result = json.loads(result_json)

        assert result["__pending__"] is True
        assert result["prompt"] == "Which color?"
        assert result["multiple"] is False
        assert result["allow_custom"] is False
        assert len(result["options"]) == 2
        assert result["options"][0]["label"] == "Red"
        assert result["options"][0]["value"] == "Red"
        assert result["options"][1]["label"] == "Blue"
        assert result["options"][1]["value"] == "b"
        assert "question_id" in result
        assert result["question_id"].startswith("q_")

        assert agent._question_pending is not None
        assert agent._question_pending["prompt"] == "Which color?"
        assert agent._question_pending["multiple"] is False
        assert agent._question_pending["allow_custom"] is False

        assert len(agent._question_events) == 1
        evt = agent._question_events[0]
        assert isinstance(evt, QuestionAsked)
        assert evt.question_text == "Which color?"
        assert evt.question_id == result["question_id"]
        assert evt.multiple is False
        assert evt.allow_custom is False
        assert len(evt.options) == 2

    @pytest.mark.asyncio
    async def test_multiple_and_custom(self):
        agent = types.SimpleNamespace(_question_pending=None, _question_events=[])
        tool_fn = make_question_tool(agent)
        result_json = await tool_fn(
            prompt="Select all that apply:",
            options=[{"label": "A"}, {"label": "B"}],
            multiple=True,
            allow_custom=True,
        )
        result = json.loads(result_json)

        assert result["multiple"] is True
        assert result["allow_custom"] is True

        assert agent._question_pending["multiple"] is True
        assert agent._question_pending["allow_custom"] is True

        evt = agent._question_events[0]
        assert evt.multiple is True
        assert evt.allow_custom is True
