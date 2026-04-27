"""Tests for feedback persistence backends."""

import pytest
import pytest_asyncio
from datetime import datetime, timedelta, timezone

from fipsagents.server.feedback import (
    FeedbackRecord,
    FeedbackStats,
    NullFeedbackStore,
    SqliteFeedbackStore,
    create_feedback_store,
    _generate_feedback_id,
    _utc_now_iso,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(**overrides) -> FeedbackRecord:
    """Create a FeedbackRecord with sensible defaults, allowing overrides."""
    defaults = dict(
        feedback_id=_generate_feedback_id(),
        trace_id="trace_abc123",
        session_id="sess_def456",
        rating=1,
        comment=None,
        correction=None,
        model_id=None,
        latency_ms=None,
        turn_index=None,
        agent_type=None,
        created_at=_utc_now_iso(),
    )
    defaults.update(overrides)
    return FeedbackRecord(**defaults)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def sqlite_store(tmp_path):
    store = SqliteFeedbackStore(str(tmp_path / "test.db"))
    yield store
    await store.close()


# ---------------------------------------------------------------------------
# NullFeedbackStore
# ---------------------------------------------------------------------------


class TestNullFeedbackStore:
    @pytest.mark.asyncio
    async def test_add_returns_id(self):
        store = NullFeedbackStore()
        record = _make_record()
        result = await store.add(record)
        assert result == record.feedback_id

    @pytest.mark.asyncio
    async def test_get_returns_none(self):
        store = NullFeedbackStore()
        assert await store.get("anything") is None

    @pytest.mark.asyncio
    async def test_query_returns_empty(self):
        store = NullFeedbackStore()
        assert await store.query() == []

    @pytest.mark.asyncio
    async def test_stats_returns_empty(self):
        store = NullFeedbackStore()
        assert await store.stats() == []

    @pytest.mark.asyncio
    async def test_delete_before_returns_zero(self):
        store = NullFeedbackStore()
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        assert await store.delete_before(cutoff) == 0


# ---------------------------------------------------------------------------
# SqliteFeedbackStore
# ---------------------------------------------------------------------------


class TestSqliteFeedbackStore:
    @pytest.mark.asyncio
    async def test_add_and_get(self, sqlite_store):
        record = _make_record(
            comment="great response",
            correction="minor fix",
            model_id="gpt-oss-20b",
            latency_ms=120.5,
            turn_index=3,
            agent_type="chat",
        )
        returned_id = await sqlite_store.add(record)
        assert returned_id == record.feedback_id

        loaded = await sqlite_store.get(record.feedback_id)
        assert loaded is not None
        assert loaded.feedback_id == record.feedback_id
        assert loaded.trace_id == record.trace_id
        assert loaded.session_id == record.session_id
        assert loaded.rating == record.rating
        assert loaded.comment == record.comment
        assert loaded.correction == record.correction
        assert loaded.model_id == record.model_id
        assert loaded.latency_ms == record.latency_ms
        assert loaded.turn_index == record.turn_index
        assert loaded.agent_type == record.agent_type
        assert loaded.created_at == record.created_at

    @pytest.mark.asyncio
    async def test_add_is_append_only(self, sqlite_store):
        """Adding two records with the same trace_id keeps both."""
        r1 = _make_record(trace_id="shared_trace", rating=1)
        r2 = _make_record(trace_id="shared_trace", rating=-1)
        await sqlite_store.add(r1)
        await sqlite_store.add(r2)

        results = await sqlite_store.query(trace_id="shared_trace")
        assert len(results) == 2
        ids = {r.feedback_id for r in results}
        assert r1.feedback_id in ids
        assert r2.feedback_id in ids

    @pytest.mark.asyncio
    async def test_query_by_trace_id(self, sqlite_store):
        r1 = _make_record(trace_id="trace_A")
        r2 = _make_record(trace_id="trace_B")
        r3 = _make_record(trace_id="trace_A")
        await sqlite_store.add(r1)
        await sqlite_store.add(r2)
        await sqlite_store.add(r3)

        results = await sqlite_store.query(trace_id="trace_A")
        assert len(results) == 2
        assert all(r.trace_id == "trace_A" for r in results)

    @pytest.mark.asyncio
    async def test_query_by_session_id(self, sqlite_store):
        r1 = _make_record(session_id="sess_X")
        r2 = _make_record(session_id="sess_Y")
        r3 = _make_record(session_id="sess_X")
        await sqlite_store.add(r1)
        await sqlite_store.add(r2)
        await sqlite_store.add(r3)

        results = await sqlite_store.query(session_id="sess_X")
        assert len(results) == 2
        assert all(r.session_id == "sess_X" for r in results)

    @pytest.mark.asyncio
    async def test_query_by_time_range(self, sqlite_store):
        old_record = _make_record()
        new_record = _make_record()
        await sqlite_store.add(old_record)
        await sqlite_store.add(new_record)

        # Backdate old_record to 30 days ago
        db = await sqlite_store._get_db()
        old_time = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        await db.execute(
            "UPDATE feedback SET created_at = ? WHERE feedback_id = ?",
            (old_time, old_record.feedback_id),
        )
        await db.commit()

        # Query for records in the last 7 days -- should exclude old_record
        since = datetime.now(timezone.utc) - timedelta(days=7)
        results = await sqlite_store.query(since=since)
        assert len(results) == 1
        assert results[0].feedback_id == new_record.feedback_id

        # Query with until set to 7 days ago -- should include only old_record
        until = datetime.now(timezone.utc) - timedelta(days=7)
        results = await sqlite_store.query(until=until)
        assert len(results) == 1
        assert results[0].feedback_id == old_record.feedback_id

    @pytest.mark.asyncio
    async def test_query_pagination(self, sqlite_store):
        records = [_make_record() for _ in range(5)]
        for r in records:
            await sqlite_store.add(r)

        page1 = await sqlite_store.query(limit=2, offset=0)
        assert len(page1) == 2

        page2 = await sqlite_store.query(limit=2, offset=2)
        assert len(page2) == 2

        page3 = await sqlite_store.query(limit=2, offset=4)
        assert len(page3) == 1

        # No overlap between pages
        all_ids = [r.feedback_id for r in page1 + page2 + page3]
        assert len(set(all_ids)) == 5

    @pytest.mark.asyncio
    async def test_stats_by_day(self, sqlite_store):
        r1 = _make_record(rating=1, agent_type="chat")
        r2 = _make_record(rating=-1, agent_type="chat")
        r3 = _make_record(rating=1, agent_type="chat")
        await sqlite_store.add(r1)
        await sqlite_store.add(r2)
        await sqlite_store.add(r3)

        results = await sqlite_store.stats(window="day")
        assert len(results) >= 1

        # All records are on the same day, find the matching bucket
        today_stats = results[0]
        assert today_stats.thumbs_up == 2
        assert today_stats.thumbs_down == 1
        assert today_stats.total == 3

    @pytest.mark.asyncio
    async def test_stats_filters_by_agent_type(self, sqlite_store):
        r1 = _make_record(rating=1, agent_type="chat")
        r2 = _make_record(rating=-1, agent_type="search")
        r3 = _make_record(rating=1, agent_type="chat")
        await sqlite_store.add(r1)
        await sqlite_store.add(r2)
        await sqlite_store.add(r3)

        results = await sqlite_store.stats(window="day", agent_type="chat")
        assert len(results) == 1
        assert results[0].thumbs_up == 2
        assert results[0].thumbs_down == 0
        assert results[0].total == 2
        assert results[0].agent_type == "chat"

    @pytest.mark.asyncio
    async def test_delete_before(self, sqlite_store):
        """delete_before removes old records but keeps recent ones."""
        old_record = _make_record()
        new_record = _make_record()
        await sqlite_store.add(old_record)
        await sqlite_store.add(new_record)

        # Backdate old_record
        db = await sqlite_store._get_db()
        old_time = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        await db.execute(
            "UPDATE feedback SET created_at = ? WHERE feedback_id = ?",
            (old_time, old_record.feedback_id),
        )
        await db.commit()

        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        deleted = await sqlite_store.delete_before(cutoff)

        assert deleted == 1
        assert await sqlite_store.get(old_record.feedback_id) is None
        assert await sqlite_store.get(new_record.feedback_id) is not None

    @pytest.mark.asyncio
    async def test_close_and_reopen(self, tmp_path):
        """Data persists across close/reopen cycles."""
        db_path = str(tmp_path / "persist.db")

        store = SqliteFeedbackStore(db_path)
        record = _make_record()
        await store.add(record)
        await store.close()

        store2 = SqliteFeedbackStore(db_path)
        loaded = await store2.get(record.feedback_id)
        await store2.close()

        assert loaded is not None
        assert loaded.feedback_id == record.feedback_id
        assert loaded.trace_id == record.trace_id


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreateFeedbackStore:
    def test_null(self):
        store = create_feedback_store(None)
        assert isinstance(store, NullFeedbackStore)

    def test_sqlite(self):
        store = create_feedback_store("sqlite")
        assert isinstance(store, SqliteFeedbackStore)

    def test_postgres_requires_url(self):
        with pytest.raises(ValueError, match="database_url"):
            create_feedback_store("postgres")
