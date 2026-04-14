"""Tests for fipsagents.baseagent.memory_sqlite — SQLiteMemoryClient and factory."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from fipsagents.baseagent.config import MemoryConfig
from fipsagents.baseagent.memory import NullMemoryClient, create_memory_client
from fipsagents.baseagent.memory_sqlite import SQLiteMemoryClient, create_sqlite_client


# ---------------------------------------------------------------------------
# Test helper
# ---------------------------------------------------------------------------


async def _make_client(tmp_path: Path) -> SQLiteMemoryClient:
    """Write a minimal config file and return a connected SQLiteMemoryClient."""
    config_file = tmp_path / ".memory-sqlite.yaml"
    config_file.write_text("db_path: test-memories.db\n")
    return await create_sqlite_client(config_file)


# ---------------------------------------------------------------------------
# TestCreateSqliteClient — factory
# ---------------------------------------------------------------------------


class TestCreateSqliteClient:
    @pytest.mark.asyncio
    async def test_creates_client_from_valid_config(self, tmp_path):
        client = await _make_client(tmp_path)
        assert isinstance(client, SQLiteMemoryClient)

    @pytest.mark.asyncio
    async def test_creates_database_file(self, tmp_path):
        await _make_client(tmp_path)
        assert (tmp_path / "test-memories.db").exists()

    @pytest.mark.asyncio
    async def test_returns_null_when_config_missing(self, tmp_path):
        client = await create_sqlite_client(tmp_path / "no-such-file.yaml")
        assert isinstance(client, NullMemoryClient)

    @pytest.mark.asyncio
    async def test_returns_null_when_config_invalid(self, tmp_path):
        bad_config = tmp_path / ".memory-sqlite.yaml"
        bad_config.write_text("[[[")
        client = await create_sqlite_client(bad_config)
        assert isinstance(client, NullMemoryClient)

    @pytest.mark.asyncio
    async def test_relative_db_path_resolved_from_config_dir(self, tmp_path):
        subdir = tmp_path / "configs"
        subdir.mkdir()
        config_file = subdir / ".memory-sqlite.yaml"
        config_file.write_text("db_path: myagent.db\n")
        client = await create_sqlite_client(config_file)
        assert isinstance(client, SQLiteMemoryClient)
        assert (subdir / "myagent.db").exists()


# ---------------------------------------------------------------------------
# TestSQLiteMemoryWrite
# ---------------------------------------------------------------------------


class TestSQLiteMemoryWrite:
    @pytest.mark.asyncio
    async def test_write_returns_dict_with_id(self, tmp_path):
        client = await _make_client(tmp_path)
        result = await client.write("hello world")
        assert isinstance(result, dict)
        assert "id" in result
        assert "content" in result
        assert "created_at" in result

    @pytest.mark.asyncio
    async def test_write_content_is_searchable(self, tmp_path):
        client = await _make_client(tmp_path)
        await client.write("unique canary content")
        results = await client.search("canary")
        assert len(results) >= 1
        assert any("canary" in r["content"] for r in results)

    @pytest.mark.asyncio
    async def test_write_with_metadata(self, tmp_path):
        client = await _make_client(tmp_path)
        meta = {"project": "test", "scope": "unit"}
        await client.write("memory with metadata", metadata=meta)
        results = await client.search("memory with metadata")
        assert len(results) >= 1
        stored = results[0]
        assert isinstance(stored.get("metadata"), dict)
        assert stored["metadata"].get("project") == "test"

    @pytest.mark.asyncio
    async def test_write_generates_unique_ids(self, tmp_path):
        client = await _make_client(tmp_path)
        r1 = await client.write("first memory")
        r2 = await client.write("second memory")
        assert r1["id"] != r2["id"]


# ---------------------------------------------------------------------------
# TestSQLiteMemorySearch
# ---------------------------------------------------------------------------


class TestSQLiteMemorySearch:
    @pytest.mark.asyncio
    async def test_search_returns_matching_results(self, tmp_path):
        client = await _make_client(tmp_path)
        await client.write("the quick brown fox")
        await client.write("pack my box with five dozen liquor jugs")
        await client.write("sphinx of black quartz judge my vow")
        results = await client.search("fox")
        assert len(results) >= 1
        assert all("fox" in r["content"] for r in results)

    @pytest.mark.asyncio
    async def test_search_empty_db_returns_empty(self, tmp_path):
        client = await _make_client(tmp_path)
        results = await client.search("nothing here")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, tmp_path):
        client = await _make_client(tmp_path)
        for i in range(5):
            await client.write(f"memory item number {i} with searchword")
        results = await client.search("searchword", limit=2)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_bm25_ranking(self, tmp_path):
        client = await _make_client(tmp_path)
        await client.write("Python is great")
        await client.write("Python Python Python is the best Python language")
        results = await client.search("Python")
        assert len(results) == 2
        # The entry with more occurrences should rank first (lower BM25 cost)
        assert "Python Python" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_search_malformed_query_falls_back_to_like(self, tmp_path):
        client = await _make_client(tmp_path)
        await client.write("content containing the word fallback")
        # Unclosed quote is invalid FTS5 syntax; should fall back to LIKE
        results = await client.search('"unclosed fallback')
        assert len(results) >= 1
        assert any("fallback" in r["content"] for r in results)


# ---------------------------------------------------------------------------
# TestSQLiteMemoryUpdate
# ---------------------------------------------------------------------------


class TestSQLiteMemoryUpdate:
    @pytest.mark.asyncio
    async def test_update_changes_content(self, tmp_path):
        client = await _make_client(tmp_path)
        written = await client.write("original content")
        await client.update(written["id"], "revised content")
        results = await client.search("revised")
        assert len(results) >= 1
        assert any("revised" in r["content"] for r in results)

    @pytest.mark.asyncio
    async def test_update_returns_dict(self, tmp_path):
        client = await _make_client(tmp_path)
        written = await client.write("some content")
        result = await client.update(written["id"], "new content")
        assert isinstance(result, dict)
        assert "id" in result
        assert "content" in result
        assert "updated_at" in result

    @pytest.mark.asyncio
    async def test_update_nonexistent_id_returns_none(self, tmp_path):
        client = await _make_client(tmp_path)
        result = await client.update("00000000-0000-0000-0000-000000000000", "content")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_updates_timestamp(self, tmp_path):
        client = await _make_client(tmp_path)
        written = await client.write("timestamped content")
        created_at = written["created_at"]
        # Small sleep to guarantee the timestamp advances
        await asyncio.sleep(0.01)
        updated = await client.update(written["id"], "updated content")
        assert updated["updated_at"] != created_at


# ---------------------------------------------------------------------------
# TestSQLiteMemoryReportContradiction
# ---------------------------------------------------------------------------


class TestSQLiteMemoryReportContradiction:
    @pytest.mark.asyncio
    async def test_report_contradiction_does_not_raise(self, tmp_path):
        client = await _make_client(tmp_path)
        written = await client.write("a fact that will be contradicted")
        # Must complete without raising
        await client.report_contradiction(written["id"], "observed contrary behavior")

    @pytest.mark.asyncio
    async def test_report_contradiction_logs_warning(self, tmp_path, caplog):
        import logging

        client = await _make_client(tmp_path)
        written = await client.write("another fact")
        with caplog.at_level(logging.WARNING):
            await client.report_contradiction(written["id"], "contradicts X")
        assert any("contradict" in record.message.lower() for record in caplog.records)


# ---------------------------------------------------------------------------
# TestSQLiteDispatchIntegration
# ---------------------------------------------------------------------------


class TestSQLiteDispatchIntegration:
    @pytest.mark.asyncio
    async def test_dispatch_sqlite_backend_creates_client(self, tmp_path):
        config_file = tmp_path / ".memory-sqlite.yaml"
        config_file.write_text("db_path: dispatch-test.db\n")
        client = await create_memory_client(
            config=MemoryConfig(backend="sqlite", config_path=str(config_file))
        )
        assert isinstance(client, SQLiteMemoryClient)
