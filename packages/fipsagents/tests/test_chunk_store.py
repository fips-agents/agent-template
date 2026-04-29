"""Tests for fipsagents.server.chunk_store.

Unit tests use mocked asyncpg + httpx — no real Postgres required.
Live integration tests are guarded by the ``chunking_live`` marker
and skip when ``CHUNKING_DATABASE_URL`` / ``EMBEDDING_URL`` are unset.
"""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fipsagents.server.chunk_store import (
    NullChunkStore,
    _embed_batch,
    _embedding_to_pgvector,
    create_pgvector_chunk_store,
)
from fipsagents.server.chunker import Chunk

try:
    import httpx  # noqa: F401

    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    import asyncpg  # noqa: F401

    from fipsagents.server.chunk_store import (
        PgvectorChunkStore,
        initialise_pgvector_schema,
    )

    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False


pgvector_only = pytest.mark.skipif(
    not (HAS_ASYNCPG and HAS_HTTPX),
    reason="asyncpg/httpx not installed (install with: pip install fipsagents[chunking])",
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _embed_response(embeddings: list[list[float]]):
    """Build a mock httpx response that mimics OpenAI embeddings shape."""
    response = MagicMock()
    response.status_code = 200
    response.raise_for_status = MagicMock()
    response.json = MagicMock(return_value={
        "data": [
            {"index": i, "embedding": emb} for i, emb in enumerate(embeddings)
        ],
    })
    return response


def _mock_pool(
    fetch_return: list | None = None,
    execute_return: str = "DELETE 0",
    executemany_return: str = "INSERT 0 0",
):
    """Build a MagicMock asyncpg pool with async-context acquire support."""
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=execute_return)
    mock_conn.executemany = AsyncMock(return_value=executemany_return)
    mock_conn.fetch = AsyncMock(return_value=fetch_return or [])

    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    pool.close = AsyncMock()
    return pool, mock_conn


def _mock_http(embeddings: list[list[float]] | Exception):
    """Build a mock httpx.AsyncClient that returns *embeddings* on /v1/embeddings."""
    http = MagicMock()
    if isinstance(embeddings, Exception):
        http.post = AsyncMock(side_effect=embeddings)
    else:
        http.post = AsyncMock(return_value=_embed_response(embeddings))
    return http


def _make_store(
    *,
    pool=None,
    http=None,
    dimension: int = 4,
    table: str = "file_chunks",
    batch: int = 32,
):
    if not HAS_ASYNCPG:
        pytest.skip("asyncpg not installed")
    if pool is None:
        pool, _ = _mock_pool()
    if http is None:
        http = _mock_http([[0.1] * dimension])
    return PgvectorChunkStore(
        pool=pool,
        http=http,
        embedding_url="http://embed:8000",
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimension=dimension,
        table_name=table,
        embedding_batch_size=batch,
    )


# ---------------------------------------------------------------------------
# Helpers (always run — no dep on asyncpg)
# ---------------------------------------------------------------------------


class TestEmbeddingFormat:
    def test_simple(self):
        assert _embedding_to_pgvector([0.1, 0.2, 0.3]) == "[0.1,0.2,0.3]"

    def test_empty(self):
        assert _embedding_to_pgvector([]) == "[]"

    def test_negative_values(self):
        assert _embedding_to_pgvector([-0.5, 0.0, 1.0]) == "[-0.5,0.0,1.0]"


class TestEmbedBatch:
    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self):
        # Should not even hit the network when no inputs are given.
        http = MagicMock()
        http.post = AsyncMock()
        out = await _embed_batch(http, "http://embed:8000", "model", [])
        assert out == []
        http.post.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_returns_embeddings_in_input_order(self):
        # Endpoint returns data with shuffled "index" — function should
        # sort by index so chunk-N pairs with embedding-N.
        http = MagicMock()
        http.post = AsyncMock(return_value=MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={
                "data": [
                    {"index": 1, "embedding": [0.2, 0.2]},
                    {"index": 0, "embedding": [0.1, 0.1]},
                    {"index": 2, "embedding": [0.3, 0.3]},
                ],
            }),
        ))
        out = await _embed_batch(http, "http://e:8000", "m", ["a", "b", "c"])
        assert out == [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]

    @pytest.mark.asyncio
    async def test_propagates_http_errors(self):
        http = _mock_http(RuntimeError("upstream 503"))
        with pytest.raises(RuntimeError, match="503"):
            await _embed_batch(http, "http://e:8000", "m", ["x"])


# ---------------------------------------------------------------------------
# NullChunkStore
# ---------------------------------------------------------------------------


class TestNullChunkStore:
    @pytest.mark.asyncio
    async def test_save_returns_zero(self):
        store = NullChunkStore()
        chunks = [Chunk(content="anything")]
        n = await store.save_chunks(
            "file-1", chunks, user_id="u1", session_id="s1",
        )
        assert n == 0

    @pytest.mark.asyncio
    async def test_search_returns_empty(self):
        store = NullChunkStore()
        out = await store.search("file-1", "any query")
        assert out == []

    @pytest.mark.asyncio
    async def test_delete_returns_zero(self):
        store = NullChunkStore()
        assert await store.delete_for_file("file-1") == 0

    @pytest.mark.asyncio
    async def test_close_is_noop(self):
        store = NullChunkStore()
        await store.close()  # should not raise


# ---------------------------------------------------------------------------
# PgvectorChunkStore — construction + validation
# ---------------------------------------------------------------------------


@pgvector_only
class TestConstruction:
    def test_invalid_table_name_rejected(self):
        with pytest.raises(ValueError, match="invalid table_name"):
            _make_store(table="bad table; DROP TABLE users")

    def test_table_name_with_special_chars_rejected(self):
        with pytest.raises(ValueError):
            _make_store(table="file_chunks'")

    def test_valid_table_names_accepted(self):
        for name in ("file_chunks", "_chunks", "agentX", "T_1_2"):
            store = _make_store(table=name)
            assert store._table == name

    def test_batch_size_minimum_one(self):
        store = _make_store(batch=0)
        assert store._batch == 1


# ---------------------------------------------------------------------------
# save_chunks
# ---------------------------------------------------------------------------


@pgvector_only
class TestSaveChunks:
    @pytest.mark.asyncio
    async def test_empty_chunks_skips_db(self):
        pool, conn = _mock_pool()
        http = _mock_http([])
        store = _make_store(pool=pool, http=http)
        n = await store.save_chunks(
            "file-1", [], user_id="u1", session_id="s1",
        )
        assert n == 0
        conn.executemany.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_writes_one_row_per_chunk(self):
        pool, conn = _mock_pool()
        http = _mock_http([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
        store = _make_store(pool=pool, http=http, dimension=4)

        chunks = [
            Chunk(content="alpha", metadata={"page": 1}),
            Chunk(content="beta"),
        ]
        n = await store.save_chunks(
            "file-1", chunks, user_id="u1", session_id="s1",
        )
        assert n == 2
        conn.executemany.assert_awaited_once()
        sql, rows = conn.executemany.await_args.args
        assert "INSERT INTO file_chunks" in sql
        assert len(rows) == 2
        # chunk_index must be 0, 1 (positional column 5 in the row tuple)
        assert rows[0][4] == 0
        assert rows[1][4] == 1
        # content column 6
        assert rows[0][5] == "alpha"
        assert rows[1][5] == "beta"
        # metadata column 7 → JSON string for chunk-0, None for chunk-1
        assert rows[0][6] == json.dumps({"page": 1})
        assert rows[1][6] is None
        # embedding column 8 → pgvector literal
        assert rows[0][7] == "[0.1,0.2,0.3,0.4]"

    @pytest.mark.asyncio
    async def test_chunk_ids_are_unique_per_file(self):
        pool, conn = _mock_pool()
        http = _mock_http([[0.0] * 4 for _ in range(3)])
        store = _make_store(pool=pool, http=http, dimension=4)

        await store.save_chunks(
            "file-1",
            [Chunk(content=f"c{i}") for i in range(3)],
            user_id="u1",
        )
        rows = conn.executemany.await_args.args[1]
        chunk_ids = {row[0] for row in rows}
        assert len(chunk_ids) == 3
        for cid in chunk_ids:
            assert cid.startswith("file-1:")

    @pytest.mark.asyncio
    async def test_embedding_failure_stores_null_embedding(self):
        # Embedding service is down; chunks should still be persisted
        # (without embeddings, so they will not be retrievable via vector
        # search but still cleanable via delete_for_file).
        pool, conn = _mock_pool()
        http = _mock_http(RuntimeError("embed down"))
        store = _make_store(pool=pool, http=http, dimension=4)

        n = await store.save_chunks(
            "file-1",
            [Chunk(content="x"), Chunk(content="y")],
            user_id="u1",
        )
        assert n == 2
        rows = conn.executemany.await_args.args[1]
        # Embedding column is None for every row
        for row in rows:
            assert row[7] is None

    @pytest.mark.asyncio
    async def test_db_failure_returns_zero(self):
        pool, conn = _mock_pool()
        conn.executemany = AsyncMock(side_effect=RuntimeError("connection lost"))
        http = _mock_http([[0.0] * 4])
        store = _make_store(pool=pool, http=http, dimension=4)

        n = await store.save_chunks(
            "file-1", [Chunk(content="x")], user_id="u1",
        )
        assert n == 0

    @pytest.mark.asyncio
    async def test_batches_embedding_calls(self):
        # 70 chunks at batch_size=32 → 3 calls (32 + 32 + 6).
        pool, conn = _mock_pool()
        embeddings_per_call = [
            [[0.0] * 4 for _ in range(32)],
            [[0.0] * 4 for _ in range(32)],
            [[0.0] * 4 for _ in range(6)],
        ]
        http = MagicMock()
        http.post = AsyncMock(side_effect=[
            _embed_response(emb) for emb in embeddings_per_call
        ])
        store = _make_store(pool=pool, http=http, dimension=4, batch=32)

        chunks = [Chunk(content=f"c{i}") for i in range(70)]
        n = await store.save_chunks("file-1", chunks, user_id="u1")
        assert n == 70
        assert http.post.await_count == 3


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


@pgvector_only
class TestSearch:
    @pytest.mark.asyncio
    async def test_returns_chunks_with_scores(self):
        rows = [
            {"content": "first chunk", "metadata": {"page": 1}, "score": 0.92},
            {"content": "second chunk", "metadata": None, "score": 0.81},
        ]
        pool, conn = _mock_pool(fetch_return=rows)
        http = _mock_http([[0.1, 0.2, 0.3, 0.4]])
        store = _make_store(pool=pool, http=http, dimension=4)

        out = await store.search("file-1", "what about page 1?")
        assert len(out) == 2
        assert out[0].content == "first chunk"
        assert out[0].metadata["score"] == pytest.approx(0.92)
        assert out[0].metadata["page"] == 1
        # Chunk with no DB metadata still gets a score key.
        assert out[1].metadata == {"score": pytest.approx(0.81)}

    @pytest.mark.asyncio
    async def test_filters_by_min_score(self):
        rows = [
            {"content": "high",  "metadata": None, "score": 0.95},
            {"content": "med",   "metadata": None, "score": 0.55},
            {"content": "low",   "metadata": None, "score": 0.20},
        ]
        pool, conn = _mock_pool(fetch_return=rows)
        http = _mock_http([[0.0, 0.0, 0.0, 1.0]])
        store = _make_store(pool=pool, http=http, dimension=4)

        out = await store.search("file-1", "q", limit=10, min_score=0.5)
        assert [c.content for c in out] == ["high", "med"]

    @pytest.mark.asyncio
    async def test_limit_passed_to_query(self):
        pool, conn = _mock_pool(fetch_return=[])
        http = _mock_http([[0.0, 0.0, 0.0, 1.0]])
        store = _make_store(pool=pool, http=http, dimension=4)

        await store.search("file-1", "q", limit=7)
        # The third positional arg of fetch is the limit.
        args = conn.fetch.await_args.args
        assert args[-1] == 7
        # The second positional arg is file_id (per-file scoping).
        assert args[-2] == "file-1"

    @pytest.mark.asyncio
    async def test_query_embed_failure_returns_empty(self):
        pool, conn = _mock_pool()
        http = _mock_http(RuntimeError("embed down"))
        store = _make_store(pool=pool, http=http, dimension=4)

        out = await store.search("file-1", "q")
        assert out == []
        # Did not even reach the SQL fetch.
        conn.fetch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_db_failure_returns_empty(self):
        pool, conn = _mock_pool()
        conn.fetch = AsyncMock(side_effect=RuntimeError("connection lost"))
        http = _mock_http([[0.0, 0.0, 0.0, 1.0]])
        store = _make_store(pool=pool, http=http, dimension=4)

        out = await store.search("file-1", "q")
        assert out == []

    @pytest.mark.asyncio
    async def test_metadata_decoded_from_json_string(self):
        # Some asyncpg configurations return JSONB as a str — the store
        # should decode it back to a dict.
        rows = [{
            "content": "c",
            "metadata": json.dumps({"page": 42, "section": "intro"}),
            "score": 0.7,
        }]
        pool, conn = _mock_pool(fetch_return=rows)
        http = _mock_http([[0.0, 0.0, 0.0, 1.0]])
        store = _make_store(pool=pool, http=http, dimension=4)

        out = await store.search("file-1", "q")
        assert out[0].metadata["page"] == 42
        assert out[0].metadata["section"] == "intro"


# ---------------------------------------------------------------------------
# delete_for_file
# ---------------------------------------------------------------------------


@pgvector_only
class TestDeleteForFile:
    @pytest.mark.asyncio
    async def test_returns_count_from_status(self):
        pool, conn = _mock_pool(execute_return="DELETE 5")
        store = _make_store(pool=pool)
        n = await store.delete_for_file("file-1")
        assert n == 5

    @pytest.mark.asyncio
    async def test_zero_when_no_rows_match(self):
        pool, conn = _mock_pool(execute_return="DELETE 0")
        store = _make_store(pool=pool)
        assert await store.delete_for_file("file-not-there") == 0

    @pytest.mark.asyncio
    async def test_returns_zero_on_db_error(self):
        pool, conn = _mock_pool()
        conn.execute = AsyncMock(side_effect=RuntimeError("conn lost"))
        store = _make_store(pool=pool)
        assert await store.delete_for_file("file-1") == 0

    @pytest.mark.asyncio
    async def test_unparseable_status_returns_zero(self):
        pool, conn = _mock_pool(execute_return="OK")
        store = _make_store(pool=pool)
        assert await store.delete_for_file("file-1") == 0


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


@pgvector_only
class TestClose:
    @pytest.mark.asyncio
    async def test_close_calls_pool_close(self):
        pool, _ = _mock_pool()
        store = _make_store(pool=pool)
        await store.close()
        pool.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# initialise_pgvector_schema
# ---------------------------------------------------------------------------


@pgvector_only
class TestInitialiseSchema:
    @pytest.mark.asyncio
    async def test_runs_all_statements(self):
        pool, conn = _mock_pool()
        await initialise_pgvector_schema(pool, "file_chunks", 768)
        # 4 statements: extension, table, file_id index, embedding index.
        assert conn.execute.await_count == 4

    @pytest.mark.asyncio
    async def test_swallows_ivfflat_failure(self):
        pool, conn = _mock_pool()
        # Make only the ivfflat statement fail.
        async def execute_side_effect(stmt, *args, **kwargs):
            if "ivfflat" in stmt:
                raise RuntimeError("table empty")
            return "OK"
        conn.execute = AsyncMock(side_effect=execute_side_effect)
        # Should not raise.
        await initialise_pgvector_schema(pool, "file_chunks", 768)

    @pytest.mark.asyncio
    async def test_propagates_non_ivfflat_failure(self):
        pool, conn = _mock_pool()
        async def execute_side_effect(stmt, *args, **kwargs):
            if "CREATE TABLE" in stmt:
                raise RuntimeError("permission denied")
            return "OK"
        conn.execute = AsyncMock(side_effect=execute_side_effect)
        with pytest.raises(RuntimeError, match="permission denied"):
            await initialise_pgvector_schema(pool, "file_chunks", 768)

    @pytest.mark.asyncio
    async def test_invalid_table_name_rejected(self):
        pool, _ = _mock_pool()
        with pytest.raises(ValueError, match="invalid table_name"):
            await initialise_pgvector_schema(pool, "bad name", 768)


# ---------------------------------------------------------------------------
# create_pgvector_chunk_store factory
# ---------------------------------------------------------------------------


@pgvector_only
class TestCreatePgvectorChunkStore:
    @pytest.mark.asyncio
    async def test_falls_back_when_database_url_empty(self):
        store = await create_pgvector_chunk_store(
            database_url="",
            embedding_url="http://e:8000",
            embedding_model="m",
        )
        assert isinstance(store, NullChunkStore)

    @pytest.mark.asyncio
    async def test_falls_back_when_embedding_url_empty(self):
        store = await create_pgvector_chunk_store(
            database_url="postgresql://localhost/x",
            embedding_url="",
            embedding_model="m",
        )
        assert isinstance(store, NullChunkStore)

    @pytest.mark.asyncio
    async def test_falls_back_on_invalid_table_name(self):
        store = await create_pgvector_chunk_store(
            database_url="postgresql://localhost/x",
            embedding_url="http://e:8000",
            embedding_model="m",
            table_name="bad name",
        )
        assert isinstance(store, NullChunkStore)

    @pytest.mark.asyncio
    async def test_falls_back_when_pool_fails(self):
        with patch(
            "asyncpg.create_pool",
            new=AsyncMock(side_effect=RuntimeError("connection refused")),
        ):
            store = await create_pgvector_chunk_store(
                database_url="postgresql://localhost/x",
                embedding_url="http://e:8000",
                embedding_model="m",
            )
        assert isinstance(store, NullChunkStore)

    @pytest.mark.asyncio
    async def test_returns_pgvector_store_on_success(self):
        pool, conn = _mock_pool()
        with patch(
            "asyncpg.create_pool",
            new=AsyncMock(return_value=pool),
        ):
            store = await create_pgvector_chunk_store(
                database_url="postgresql://localhost/x",
                embedding_url="http://e:8000",
                embedding_model="m",
                table_name="file_chunks",
                embedding_dimension=4,
            )
        assert isinstance(store, PgvectorChunkStore)
        # Schema bootstrap ran (4 DDL statements).
        assert conn.execute.await_count == 4


# ---------------------------------------------------------------------------
# Live integration tests (marked, env-gated)
# ---------------------------------------------------------------------------


@pytest.mark.chunking_live
@pgvector_only
class TestLivePgvector:
    """Live tests against a real Postgres+pgvector and embedding endpoint.

    Skipped unless ``CHUNKING_DATABASE_URL`` and ``EMBEDDING_URL`` are
    set. Run manually with::

        CHUNKING_DATABASE_URL=postgresql://... \\
        EMBEDDING_URL=http://... \\
        EMBEDDING_MODEL=all-MiniLM-L6-v2 \\
        EMBEDDING_DIMENSION=384 \\
        pytest -m chunking_live
    """

    @pytest.fixture
    def live_config(self):
        db = os.environ.get("CHUNKING_DATABASE_URL")
        embed = os.environ.get("EMBEDDING_URL")
        if not (db and embed):
            pytest.skip(
                "live test requires CHUNKING_DATABASE_URL and EMBEDDING_URL",
            )
        return {
            "database_url": db,
            "embedding_url": embed,
            "embedding_model": os.environ.get(
                "EMBEDDING_MODEL", "all-MiniLM-L6-v2",
            ),
            "embedding_dimension": int(
                os.environ.get("EMBEDDING_DIMENSION", "384"),
            ),
            "table_name": os.environ.get(
                "CHUNKING_TABLE", "file_chunks_test",
            ),
        }

    @pytest.mark.asyncio
    async def test_round_trip(self, live_config):
        store = await create_pgvector_chunk_store(**live_config)
        assert isinstance(store, PgvectorChunkStore)
        try:
            file_id = f"test-{os.getpid()}"
            await store.delete_for_file(file_id)  # clean slate

            chunks = [
                Chunk(content="The quick brown fox jumps over the lazy dog."),
                Chunk(content="Pack my box with five dozen liquor jugs."),
                Chunk(content="How vexingly quick daft zebras jump."),
            ]
            n = await store.save_chunks(
                file_id, chunks, user_id="test-user",
            )
            assert n == 3

            results = await store.search(file_id, "fox jumps", limit=2)
            assert len(results) >= 1
            # The fox sentence should rank first or near first.
            assert any("fox" in r.content for r in results)

            removed = await store.delete_for_file(file_id)
            assert removed == 3
        finally:
            await store.close()
