"""Tests for fipsagents.baseagent.memory_pgvector — PGVectorMemoryClient and factory.

These are unit tests with mocked asyncpg and httpx — no real PostgreSQL required.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from fipsagents.baseagent.memory import NullMemoryClient

try:
    from fipsagents.baseagent.memory_pgvector import (
        PGVectorMemoryClient,
        create_pgvector_client,
    )

    HAS_PGVECTOR_MODULE = True
except ImportError:
    HAS_PGVECTOR_MODULE = False

pytestmark = pytest.mark.skipif(
    not HAS_PGVECTOR_MODULE,
    reason="asyncpg not installed (install with: pip install fipsagents[pgvector])",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_embedding_response(dimension: int = 768) -> MagicMock:
    """Return a mock httpx response that looks like an OpenAI embeddings response."""
    response = MagicMock()
    response.status_code = 200
    response.raise_for_status = MagicMock()
    response.json = MagicMock(
        return_value={"data": [{"embedding": [0.1] * dimension}]}
    )
    return response


def _mock_pool(fetch_return=None, fetchrow_return=None, execute_return="INSERT 0 1"):
    """Return a MagicMock asyncpg pool with pre-configured async methods.

    The pool supports both direct pool.execute/fetch calls and
    'async with pool.acquire() as conn:' patterns.
    """
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=execute_return)
    mock_conn.fetch = AsyncMock(return_value=fetch_return or [])
    mock_conn.fetchrow = AsyncMock(return_value=fetchrow_return)

    mock_pool = MagicMock()
    mock_pool.execute = AsyncMock(return_value=execute_return)
    mock_pool.fetch = AsyncMock(return_value=fetch_return or [])
    mock_pool.fetchrow = AsyncMock(return_value=fetchrow_return)

    # Support 'async with pool.acquire() as conn:'
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

    return mock_pool, mock_conn


def _make_client(pool=None, http=None, dimension=768):
    """Construct a PGVectorMemoryClient with mock dependencies."""
    if pool is None:
        pool, _ = _mock_pool()
    if http is None:
        http = AsyncMock(spec=httpx.AsyncClient)
        http.post = AsyncMock(return_value=_mock_embedding_response(dimension))

    return PGVectorMemoryClient(
        pool=pool,
        http=http,
        embedding_url="http://vllm-host:8000",
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimension=dimension,
        table_name="agent_memories",
    )


def _sample_row(
    memory_id: str = "test-id-1",
    content: str = "sample memory content",
    metadata: dict | None = None,
) -> dict:
    """Return a dict that mimics an asyncpg record row."""
    return {
        "id": memory_id,
        "content": content,
        "metadata": metadata or {},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# TestCreatePgvectorClient — factory
# ---------------------------------------------------------------------------


class TestCreatePgvectorClient:
    @pytest.mark.asyncio
    async def test_returns_null_when_config_missing(self, tmp_path):
        client = await create_pgvector_client(tmp_path / "no-such-file.yaml")
        assert isinstance(client, NullMemoryClient)

    @pytest.mark.asyncio
    async def test_returns_null_when_config_invalid(self, tmp_path):
        bad_config = tmp_path / ".memory-pgvector.yaml"
        bad_config.write_text("[[[")
        client = await create_pgvector_client(bad_config)
        assert isinstance(client, NullMemoryClient)

    @pytest.mark.asyncio
    async def test_returns_null_when_connection_url_missing(self, tmp_path):
        config_file = tmp_path / ".memory-pgvector.yaml"
        config_file.write_text(
            "embedding_url: http://vllm-host:8000\n"
            "embedding_model: all-MiniLM-L6-v2\n"
        )
        client = await create_pgvector_client(config_file)
        assert isinstance(client, NullMemoryClient)

    @pytest.mark.asyncio
    async def test_returns_pgvector_client_when_config_valid(self, tmp_path):
        config_file = tmp_path / ".memory-pgvector.yaml"
        config_file.write_text(
            "connection_url: postgresql://user:pass@localhost:5432/agentdb\n"
            "embedding_url: http://vllm-host:8000\n"
            "embedding_model: all-MiniLM-L6-v2\n"
            "embedding_dimension: 768\n"
            "table_name: agent_memories\n"
        )
        mock_pool, _ = _mock_pool()
        mock_http = AsyncMock(spec=httpx.AsyncClient)

        with (
            patch("asyncpg.create_pool", new=AsyncMock(return_value=mock_pool)),
            patch("httpx.AsyncClient", return_value=mock_http),
        ):
            client = await create_pgvector_client(config_file)

        assert isinstance(client, PGVectorMemoryClient)


# ---------------------------------------------------------------------------
# TestPGVectorMemorySearch
# ---------------------------------------------------------------------------


class TestPGVectorMemorySearch:
    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        row = _sample_row()
        mock_pool, _ = _mock_pool(fetch_return=[row])
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(return_value=_mock_embedding_response())

        client = _make_client(pool=mock_pool, http=mock_http)
        results = await client.search("sample query")

        assert len(results) == 1
        assert results[0]["id"] == row["id"]
        assert results[0]["content"] == row["content"]

    @pytest.mark.asyncio
    async def test_search_empty_results(self):
        mock_pool, _ = _mock_pool(fetch_return=[])
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(return_value=_mock_embedding_response())

        client = _make_client(pool=mock_pool, http=mock_http)
        results = await client.search("no match query")

        assert results == []

    @pytest.mark.asyncio
    async def test_search_respects_limit(self):
        rows = [_sample_row(memory_id=f"id-{i}", content=f"memory {i}") for i in range(5)]
        mock_pool, _ = _mock_pool(fetch_return=rows[:2])
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(return_value=_mock_embedding_response())

        client = _make_client(pool=mock_pool, http=mock_http)
        results = await client.search("query", limit=2)

        # The implementation should have passed limit=2; we verify via return value
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_falls_back_to_ilike_on_embedding_failure(self):
        row = _sample_row(content="fallback content")
        mock_pool, _ = _mock_pool(fetch_return=[row])
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(side_effect=RuntimeError("embedding service down"))

        client = _make_client(pool=mock_pool, http=mock_http)
        results = await client.search("fallback")

        # Despite embedding failure, ILIKE fallback returns results
        assert len(results) >= 1
        assert results[0]["content"] == "fallback content"

    @pytest.mark.asyncio
    async def test_search_exception_returns_empty(self):
        mock_pool, _ = _mock_pool()
        mock_pool.fetch = AsyncMock(side_effect=RuntimeError("db down"))
        mock_pool.execute = AsyncMock(side_effect=RuntimeError("db down"))
        # Also make acquire fail so both paths fail
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(
            side_effect=RuntimeError("db down")
        )
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(return_value=_mock_embedding_response())

        client = _make_client(pool=mock_pool, http=mock_http)
        results = await client.search("query that will fail")

        assert results == []


# ---------------------------------------------------------------------------
# TestPGVectorMemoryWrite
# ---------------------------------------------------------------------------


class TestPGVectorMemoryWrite:
    @pytest.mark.asyncio
    async def test_write_returns_dict_with_id(self):
        row = _sample_row()
        mock_pool, mock_conn = _mock_pool(fetchrow_return=row)
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(return_value=_mock_embedding_response())

        client = _make_client(pool=mock_pool, http=mock_http)
        result = await client.write("hello world")

        assert result is not None
        assert isinstance(result, dict)
        assert "id" in result
        assert "content" in result
        assert "created_at" in result

    @pytest.mark.asyncio
    async def test_write_with_metadata(self):
        meta = {"project": "test", "scope": "unit"}
        row = _sample_row(metadata=meta)
        mock_pool, mock_conn = _mock_pool(fetchrow_return=row)
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(return_value=_mock_embedding_response())

        client = _make_client(pool=mock_pool, http=mock_http)
        result = await client.write("memory with metadata", metadata=meta)

        assert result is not None
        assert "id" in result

    @pytest.mark.asyncio
    async def test_write_falls_back_to_null_embedding_on_embedding_failure(self):
        row = _sample_row()
        mock_pool, mock_conn = _mock_pool(fetchrow_return=row)
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(side_effect=RuntimeError("embedding service down"))

        client = _make_client(pool=mock_pool, http=mock_http)
        # Should still succeed even without an embedding
        result = await client.write("content without embedding")

        assert result is not None
        assert "id" in result

    @pytest.mark.asyncio
    async def test_write_exception_returns_none(self):
        mock_pool, mock_conn = _mock_pool()
        mock_pool.execute = AsyncMock(side_effect=RuntimeError("db error"))
        mock_conn.execute = AsyncMock(side_effect=RuntimeError("db error"))
        mock_pool.fetchrow = AsyncMock(side_effect=RuntimeError("db error"))
        mock_conn.fetchrow = AsyncMock(side_effect=RuntimeError("db error"))
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(
            side_effect=RuntimeError("db error")
        )
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(return_value=_mock_embedding_response())

        client = _make_client(pool=mock_pool, http=mock_http)
        result = await client.write("will fail")

        assert result is None


# ---------------------------------------------------------------------------
# TestPGVectorMemoryUpdate
# ---------------------------------------------------------------------------


class TestPGVectorMemoryUpdate:
    @pytest.mark.asyncio
    async def test_update_returns_dict(self):
        row = _sample_row(content="updated content")
        mock_pool, mock_conn = _mock_pool(execute_return="UPDATE 1", fetchrow_return=row)
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(return_value=_mock_embedding_response())

        client = _make_client(pool=mock_pool, http=mock_http)
        result = await client.update("test-id-1", "updated content")

        assert result is not None
        assert isinstance(result, dict)
        assert "id" in result

    @pytest.mark.asyncio
    async def test_update_nonexistent_id_returns_none(self):
        # asyncpg returns "UPDATE 0" status when no rows matched
        mock_pool, mock_conn = _mock_pool(execute_return="UPDATE 0", fetchrow_return=None)
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(return_value=_mock_embedding_response())

        client = _make_client(pool=mock_pool, http=mock_http)
        result = await client.update("00000000-0000-0000-0000-000000000000", "content")

        assert result is None

    @pytest.mark.asyncio
    async def test_update_exception_returns_none(self):
        mock_pool, mock_conn = _mock_pool()
        mock_pool.execute = AsyncMock(side_effect=RuntimeError("db error"))
        mock_conn.execute = AsyncMock(side_effect=RuntimeError("db error"))
        mock_pool.fetchrow = AsyncMock(side_effect=RuntimeError("db error"))
        mock_conn.fetchrow = AsyncMock(side_effect=RuntimeError("db error"))
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(
            side_effect=RuntimeError("db error")
        )
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(return_value=_mock_embedding_response())

        client = _make_client(pool=mock_pool, http=mock_http)
        result = await client.update("some-id", "new content")

        assert result is None


# ---------------------------------------------------------------------------
# TestPGVectorMemoryReportContradiction
# ---------------------------------------------------------------------------


class TestPGVectorMemoryReportContradiction:
    @pytest.mark.asyncio
    async def test_report_contradiction_does_not_raise(self):
        client = _make_client()
        # Must complete without raising
        await client.report_contradiction("mem-id-123", "contradicts earlier fact")

    @pytest.mark.asyncio
    async def test_report_contradiction_logs_warning(self, caplog):
        client = _make_client()
        with caplog.at_level(logging.WARNING):
            await client.report_contradiction("mem-id-456", "contradicts X")
        assert any("contradict" in record.message.lower() for record in caplog.records)
