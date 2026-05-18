"""Tests for Compactor ABC, NullCompactor, and LLMCompactor."""
import pytest
from fipsagents.server.compactor import (
    CompactionResult,
    CompactionState,
    LLMCompactor,
    NullCompactor,
    create_compactor,
)


class TestCompactionDataclasses:
    def test_compaction_state_defaults(self):
        state = CompactionState()
        assert state.last_compacted_at is None
        assert state.last_compacted_message_id is None
        assert state.compaction_count == 0

    def test_compaction_result_defaults(self):
        result = CompactionResult()
        assert result.messages == []
        assert result.original_count == 0
        assert result.compacted_count == 0
        assert result.skipped is False
        assert result.skip_reason is None


class TestNullCompactor:
    @pytest.mark.asyncio
    async def test_should_compact_returns_false(self):
        c = NullCompactor()
        assert await c.should_compact([{"role": "user", "content": "hi"}]) is False

    @pytest.mark.asyncio
    async def test_compact_returns_unchanged(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        c = NullCompactor()
        result = await c.compact(messages)
        assert result.messages is messages
        assert result.original_count == 2
        assert result.compacted_count == 2
        assert result.skipped is True
        assert result.skip_reason == "null_compactor"

    @pytest.mark.asyncio
    async def test_close_is_noop(self):
        c = NullCompactor()
        await c.close()  # should not raise


class TestCreateCompactor:
    def test_none_returns_null(self):
        c = create_compactor(None)
        assert isinstance(c, NullCompactor)

    def test_null_string_returns_null(self):
        c = create_compactor("null")
        assert isinstance(c, NullCompactor)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown compactor"):
            create_compactor("nonexistent")


# -- LLMCompactor tests ---------------------------------------------------

async def _mock_model_fn(messages):
    return "**Goal**: Test goal\n**Progress**: Done testing."


async def _failing_model_fn(messages):
    raise RuntimeError("LLM unavailable")


def _make_conversation(n_user_turns: int) -> list[dict]:
    """Build a conversation with n user/assistant pairs."""
    msgs: list[dict] = [{"role": "system", "content": "You are helpful."}]
    for i in range(n_user_turns):
        msgs.append({"role": "user", "content": f"Question {i}"})
        msgs.append({"role": "assistant", "content": f"Answer {i}"})
    return msgs


class TestLLMCompactorShouldCompact:
    @pytest.mark.asyncio
    async def test_threshold_triggers(self):
        c = LLMCompactor(_mock_model_fn, threshold_messages=5)
        msgs = _make_conversation(3)  # 1 system + 6 non-system = 7 non-system msgs
        assert await c.should_compact(msgs) is True

    @pytest.mark.asyncio
    async def test_under_threshold_no_trigger(self):
        c = LLMCompactor(_mock_model_fn, threshold_messages=50)
        msgs = _make_conversation(3)
        assert await c.should_compact(msgs) is False

    @pytest.mark.asyncio
    async def test_context_limit_triggers(self):
        c = LLMCompactor(
            _mock_model_fn,
            threshold_messages=999,
            context_limit=10,
            reserve_tokens=5,
        )
        msgs = _make_conversation(3)
        assert await c.should_compact(msgs) is True


class TestLLMCompactorCompact:
    @pytest.mark.asyncio
    async def test_preserves_recent_turns(self):
        c = LLMCompactor(_mock_model_fn, threshold_messages=5, keep_recent_turns=2)
        msgs = _make_conversation(5)
        result = await c.compact(msgs)
        assert not result.skipped
        # System msg + summary msg + last 2 user/assistant pairs (4 msgs)
        assert result.compacted_count == 1 + 1 + 4

        # The last two user messages should be in the result.
        user_msgs = [m for m in result.messages if m.get("role") == "user"]
        assert user_msgs[-1]["content"] == "Question 4"
        assert user_msgs[-2]["content"] == "Question 3"

    @pytest.mark.asyncio
    async def test_tool_call_pairing(self):
        """Tool call results should not be split from their tool_calls."""
        c = LLMCompactor(_mock_model_fn, threshold_messages=5, keep_recent_turns=1)
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q0"},
            {"role": "assistant", "content": "a0"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "tc_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "tc_1", "content": "result1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        result = await c.compact(msgs)
        assert not result.skipped
        # The tool_calls assistant and its tool result should both be
        # either compacted or preserved together.
        preserved_ids = {
            m.get("tool_call_id") for m in result.messages if m.get("role") == "tool"
        }
        preserved_tc_ids = set()
        for m in result.messages:
            for tc in m.get("tool_calls", []):
                preserved_tc_ids.add(tc["id"])
        # If tool_calls assistant is preserved, its tool result must also be.
        for tc_id in preserved_tc_ids:
            assert tc_id in preserved_ids

    @pytest.mark.asyncio
    async def test_pending_state_skip(self):
        c = LLMCompactor(_mock_model_fn, threshold_messages=3, keep_recent_turns=1)
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q0"},
            {"role": "assistant", "content": '{"__pending__": true}'},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ]
        result = await c.compact(msgs)
        assert result.skipped is True
        assert result.skip_reason == "pending_state"

    @pytest.mark.asyncio
    async def test_llm_error_fallback(self):
        c = LLMCompactor(_failing_model_fn, threshold_messages=3, keep_recent_turns=1)
        msgs = _make_conversation(3)
        result = await c.compact(msgs)
        assert result.skipped is True
        assert result.skip_reason == "llm_error"
        assert result.messages is msgs

    @pytest.mark.asyncio
    async def test_summary_message_role(self):
        c = LLMCompactor(
            _mock_model_fn,
            threshold_messages=3,
            keep_recent_turns=1,
            summary_role="system",
        )
        msgs = _make_conversation(3)
        result = await c.compact(msgs)
        assert not result.skipped
        summary = [
            m for m in result.messages
            if m.get("content", "").startswith("**Goal**")
        ]
        assert len(summary) == 1
        assert summary[0]["role"] == "system"


class TestCreateCompactorLLM:
    def test_llm_backend_creates_llm_compactor(self):
        c = create_compactor("llm", model_fn=_mock_model_fn, threshold_messages=10)
        assert isinstance(c, LLMCompactor)

    def test_llm_backend_without_model_fn_raises(self):
        with pytest.raises(ValueError, match="model_fn"):
            create_compactor("llm")
