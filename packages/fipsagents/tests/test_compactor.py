"""Tests for Compactor ABC and NullCompactor."""
import pytest
from fipsagents.server.compactor import (
    CompactionResult,
    CompactionState,
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
