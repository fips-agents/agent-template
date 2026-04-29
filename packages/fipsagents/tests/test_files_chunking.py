"""Tests for the chunking integration in OpenAIChatServer (PR-C of #137).

Covers:

- ``ChunkingConfig`` budget presets and override behavior.
- ``FileRecord.chunk_status`` / ``chunk_count`` round-trip through
  :class:`SqliteFileStore`.
- The upload path schedules an async chunking task and writes status
  transitions back to the metadata store.
- The chat-completion path takes the chunked branch when chunks are
  available and falls back to full-text otherwise.
- ``DELETE /v1/files/{id}`` cascades to the chunk store.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from fipsagents.baseagent.config import ChunkingConfig, FilesConfig  # noqa: E402
from fipsagents.server import OpenAIChatServer  # noqa: E402
from fipsagents.server.chunk_store import ChunkStore  # noqa: E402
from fipsagents.server.chunker import Chunk  # noqa: E402
from fipsagents.server.files import FileRecord, SqliteFileStore  # noqa: E402

from tests.test_server_openai import _make_agent_class  # noqa: E402


# ---------------------------------------------------------------------------
# ChunkingConfig
# ---------------------------------------------------------------------------


class TestChunkingConfigPresets:
    def test_default_disabled(self):
        cfg = ChunkingConfig()
        assert cfg.enabled is False
        assert cfg.backend == "null"

    def test_small_preset_applies_to_unset_fields(self):
        cfg = ChunkingConfig(budget="small")
        assert cfg.chunk_size_tokens == 400
        assert cfg.retrieval_top_k == 3
        assert cfg.small_file_threshold_tokens == 2000

    def test_medium_preset_matches_defaults(self):
        cfg = ChunkingConfig(budget="medium")
        assert cfg.chunk_size_tokens == 600
        assert cfg.retrieval_top_k == 5
        assert cfg.small_file_threshold_tokens == 4000

    def test_large_preset_applies(self):
        cfg = ChunkingConfig(budget="large")
        assert cfg.chunk_size_tokens == 800
        assert cfg.retrieval_top_k == 8
        assert cfg.small_file_threshold_tokens == 8000

    def test_explicit_value_overrides_preset(self):
        # Preset = small (400), but the user explicitly asked for 1000.
        cfg = ChunkingConfig(budget="small", chunk_size_tokens=1000)
        assert cfg.chunk_size_tokens == 1000
        # Other preset fields still applied.
        assert cfg.retrieval_top_k == 3

    def test_custom_budget_does_not_apply_a_preset(self):
        cfg = ChunkingConfig(budget="custom")
        # "custom" is allowed in the schema but is not a key in the
        # preset table, so the regular field defaults apply.
        assert cfg.chunk_size_tokens == 600
        assert cfg.retrieval_top_k == 5

    def test_validation_rejects_invalid_dimension(self):
        with pytest.raises(ValueError):
            ChunkingConfig(embedding_dimension=0)

    def test_validation_rejects_invalid_top_k(self):
        with pytest.raises(ValueError):
            ChunkingConfig(retrieval_top_k=0)

    def test_min_score_must_be_unit_interval(self):
        with pytest.raises(ValueError):
            ChunkingConfig(retrieval_min_score=1.5)
        with pytest.raises(ValueError):
            ChunkingConfig(retrieval_min_score=-0.1)


class TestFilesConfigComposition:
    def test_files_config_has_chunking_block(self):
        files = FilesConfig()
        assert isinstance(files.chunking, ChunkingConfig)
        assert files.chunking.enabled is False


# ---------------------------------------------------------------------------
# FileRecord round-trip through SqliteFileStore
# ---------------------------------------------------------------------------


class TestFileRecordChunkFields:
    @pytest.mark.asyncio
    async def test_record_defaults(self):
        record = FileRecord(
            file_id="f1", filename="x.txt", mime_type="text/plain",
            size_bytes=4, sha256="x" * 64,
        )
        assert record.chunk_status == "pending"
        assert record.chunk_count == 0

    @pytest.mark.asyncio
    async def test_sqlite_round_trip(self, tmp_path):
        store = SqliteFileStore(
            str(tmp_path / "agent.db"),
            bytes_dir=str(tmp_path / "files"),
        )
        try:
            record = FileRecord(
                file_id="f1", filename="x.txt", mime_type="text/plain",
                size_bytes=5, sha256="",
                chunk_status="processing", chunk_count=0,
            )
            await store.save(record, b"hello")

            got = await store.get_metadata("f1")
            assert got is not None
            assert got.chunk_status == "processing"
            assert got.chunk_count == 0

            updated = await store.update_chunk_status(
                "f1", chunk_status="completed", chunk_count=7,
            )
            assert updated is True

            got = await store.get_metadata("f1")
            assert got.chunk_status == "completed"
            assert got.chunk_count == 7
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_update_chunk_status_unknown_file(self, tmp_path):
        store = SqliteFileStore(
            str(tmp_path / "agent.db"),
            bytes_dir=str(tmp_path / "files"),
        )
        try:
            updated = await store.update_chunk_status(
                "no-such-file", chunk_status="completed",
            )
            assert updated is False
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_legacy_table_migration(self, tmp_path):
        """A 0.16.0/0.17.0 SQLite file (no chunk columns) gets migrated."""
        import aiosqlite

        db_path = str(tmp_path / "legacy.db")
        legacy_ddl = """
        CREATE TABLE files (
            file_id TEXT PRIMARY KEY,
            session_id TEXT,
            user_id TEXT NOT NULL DEFAULT 'anonymous',
            filename TEXT NOT NULL,
            mime_type TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            extracted_text TEXT,
            parse_status TEXT NOT NULL DEFAULT 'pending',
            parse_error TEXT,
            bytes_path TEXT NOT NULL,
            created_at TEXT NOT NULL,
            deleted_at TEXT
        )
        """
        async with aiosqlite.connect(db_path) as db:
            await db.execute(legacy_ddl)
            await db.commit()

        # Re-open via SqliteFileStore — should ALTER ADD the new columns.
        store = SqliteFileStore(db_path, bytes_dir=str(tmp_path / "files"))
        try:
            record = FileRecord(
                file_id="f1", filename="x.txt", mime_type="text/plain",
                size_bytes=5, sha256="",
            )
            await store.save(record, b"hello")
            got = await store.get_metadata("f1")
            assert got is not None
            assert got.chunk_status == "pending"
            assert got.chunk_count == 0
        finally:
            await store.close()


# ---------------------------------------------------------------------------
# Server wiring helpers
# ---------------------------------------------------------------------------


class _StubChunkStore(ChunkStore):
    """In-memory chunk store for end-to-end wiring tests.

    Inherits from :class:`ChunkStore` (not :class:`NullChunkStore`) so
    the server's ``isinstance(..., NullChunkStore)`` short-circuits do
    not skip the chunked path.
    """

    def __init__(self) -> None:
        self.saved: dict[str, list[Chunk]] = {}
        self.search_calls: list[tuple[str, str]] = []
        self.deleted: list[str] = []

    async def save_chunks(
        self,
        file_id: str,
        chunks: list[Chunk],
        *,
        user_id: str,
        session_id: str | None = None,
    ) -> int:
        self.saved[file_id] = list(chunks)
        return len(chunks)

    async def search(
        self,
        file_id: str,
        query: str,
        *,
        limit: int = 5,
        min_score: float = 0.0,
    ) -> list[Chunk]:
        self.search_calls.append((file_id, query))
        chunks = self.saved.get(file_id, [])
        return chunks[:limit]

    async def delete_for_file(self, file_id: str) -> int:
        self.deleted.append(file_id)
        return len(self.saved.pop(file_id, []))


def _build_server_with_chunking(
    tmp_path,
    *,
    enabled: bool = True,
    threshold: int = 50,
    chunk_size: int = 30,
    events: list[Any] | None = None,
    stub_chunk_store: _StubChunkStore | None = None,
):
    """Wire a server with chunking configured against an in-memory stub."""
    AgentClass = _make_agent_class(events or [], model_name="m1")
    bytes_dir = str(tmp_path / "files")
    sqlite_path = str(tmp_path / "agent.db")
    stub = stub_chunk_store or _StubChunkStore()

    class _A(AgentClass):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.config.server.storage.backend = "sqlite"
            self.config.server.storage.sqlite_path = sqlite_path
            self.config.server.files.enabled = True
            self.config.server.files.bytes_dir = bytes_dir
            self.config.server.files.backend = "sqlite"
            self.config.server.files.sqlite_path = ""
            # Replace the SimpleNamespace chunking stub with realistic
            # values; the harness already set enabled=False.
            self.config.server.files.chunking.enabled = enabled
            self.config.server.files.chunking.backend = "null"  # we inject the stub
            self.config.server.files.chunking.chunk_size_tokens = chunk_size
            self.config.server.files.chunking.chunk_overlap_tokens = 0
            self.config.server.files.chunking.small_file_threshold_tokens = threshold
            self.config.server.files.chunking.retrieval_top_k = 5
            self.config.server.files.chunking.retrieval_min_score = 0.0

    server = OpenAIChatServer(_A)
    return server, stub


# ---------------------------------------------------------------------------
# Upload path: async chunking task
# ---------------------------------------------------------------------------


class TestUploadKicksOffChunking:
    def _patch_chunk_store(self, server, stub):
        """After lifespan setup, replace the NullChunkStore with our stub."""
        # We hook into _chunk_uploaded_file via the live attribute swap:
        # FastAPI's lifespan has already run by the time TestClient enters
        # the context, so we replace the field with our stub for the
        # duration of the request.
        server._chunk_store = stub

    def test_small_file_is_skipped(self, tmp_path):
        server, stub = _build_server_with_chunking(
            tmp_path, enabled=True, threshold=50,
        )
        with TestClient(server.app) as client:
            self._patch_chunk_store(server, stub)
            resp = client.post(
                "/v1/files",
                files={"file": ("hi.txt", b"short", "text/plain")},
            )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        # ~1 token; well below threshold 50.
        assert body["chunk_status"] == "skipped"
        assert body["chunk_count"] == 0
        assert stub.saved == {}

    def test_large_file_triggers_async_chunking(self, tmp_path):
        server, stub = _build_server_with_chunking(
            tmp_path, enabled=True, threshold=50, chunk_size=20,
        )
        # Build content that is well over the threshold.
        para = "The quick brown fox jumps over the lazy dog.\n\n"
        big = (para * 30).encode("utf-8")  # ~250 tokens
        with TestClient(server.app) as client:
            self._patch_chunk_store(server, stub)
            resp = client.post(
                "/v1/files",
                files={"file": ("doc.txt", big, "text/plain")},
            )
            assert resp.status_code == 201, resp.text
            file_id = resp.json()["file_id"]
            # Initial response shows processing.
            assert resp.json()["chunk_status"] == "processing"

            # Poll the GET endpoint until the background task completes.
            # GET runs on the same event loop as the upload task, so
            # the chunk store sees a consistent view.
            import time
            deadline = time.time() + 2.0
            got: dict = {}
            while time.time() < deadline:
                got = client.get(f"/v1/files/{file_id}").json()
                if got["chunk_status"] == "completed":
                    break
                time.sleep(0.05)
            assert got.get("chunk_status") == "completed", got

            # Verify chunks landed in the stub.
            assert file_id in stub.saved
            assert len(stub.saved[file_id]) > 0
            assert got["chunk_count"] == len(stub.saved[file_id])

    def test_chunking_disabled_keeps_status_skipped(self, tmp_path):
        server, stub = _build_server_with_chunking(
            tmp_path, enabled=False, threshold=10,
        )
        with TestClient(server.app) as client:
            resp = client.post(
                "/v1/files",
                files={"file": ("doc.txt", b"x" * 1000, "text/plain")},
            )
        body = resp.json()
        # Disabled → never touches the chunked path.
        assert body["chunk_status"] == "skipped"
        assert body["chunk_count"] == 0
        assert stub.saved == {}

# ---------------------------------------------------------------------------
# Delete cascade
# ---------------------------------------------------------------------------


class TestDeleteCascadesToChunks:
    def test_delete_calls_chunk_store(self, tmp_path):
        server, stub = _build_server_with_chunking(
            tmp_path, enabled=True, threshold=10_000,  # nothing chunks
        )
        with TestClient(server.app) as client:
            server._chunk_store = stub
            up = client.post(
                "/v1/files",
                files={"file": ("x.txt", b"hi", "text/plain")},
            )
            file_id = up.json()["file_id"]

            resp = client.delete(f"/v1/files/{file_id}")
            assert resp.status_code == 200, resp.text
            assert stub.deleted == [file_id]


# ---------------------------------------------------------------------------
# Retrieval branch in chat completions
# ---------------------------------------------------------------------------


class TestChatRetrievalBranch:
    @pytest.mark.asyncio
    async def test_resolve_falls_back_to_full_text_when_no_chunks(self, tmp_path):
        """When chunk_count == 0 the resolver injects the full extracted_text."""
        # Pure unit test: build a server, manually populate file metadata
        # with chunk_status=skipped, call _resolve_file_attachments, and
        # assert it took the full-text branch.
        server, stub = _build_server_with_chunking(
            tmp_path, enabled=True, threshold=10_000,
        )
        with TestClient(server.app) as _:  # trigger lifespan
            server._chunk_store = stub
            # Insert a file that has extracted_text but no chunks.
            await server._file_store.save(  # type: ignore[union-attr]
                FileRecord(
                    file_id="f1", filename="doc.txt", mime_type="text/plain",
                    size_bytes=5, sha256="",
                    extracted_text="hello full text body",
                    parse_status="completed",
                    chunk_status="skipped",
                    chunk_count=0,
                ),
                b"hello",
            )
            msgs = await server._resolve_file_attachments(
                ["f1"], last_user_message="any query",
            )
        assert len(msgs) == 1
        assert "hello full text body" in msgs[0]["content"]
        # Stub never queried because no chunks exist.
        assert stub.search_calls == []

    @pytest.mark.asyncio
    async def test_resolve_chunked_path_when_chunks_exist(self, tmp_path):
        server, stub = _build_server_with_chunking(
            tmp_path, enabled=True, threshold=10,
        )
        with TestClient(server.app) as _:
            server._chunk_store = stub
            stub.saved["f1"] = [
                Chunk(content="alpha bravo"),
                Chunk(content="charlie delta"),
            ]
            await server._file_store.save(  # type: ignore[union-attr]
                FileRecord(
                    file_id="f1", filename="doc.txt", mime_type="text/plain",
                    size_bytes=5, sha256="",
                    extracted_text="this is the full text we should NOT see",
                    parse_status="completed",
                    chunk_status="completed",
                    chunk_count=2,
                ),
                b"hello",
            )
            msgs = await server._resolve_file_attachments(
                ["f1"], last_user_message="bravo?",
            )
        assert len(msgs) == 1
        body = msgs[0]["content"]
        assert "alpha bravo" in body
        assert "charlie delta" in body
        # Sentinel that the full text did NOT leak into the prompt.
        assert "we should NOT see" not in body
        # Stub got called with the user query.
        assert stub.search_calls == [("f1", "bravo?")]

    @pytest.mark.asyncio
    async def test_resolve_falls_back_when_chunks_disabled(self, tmp_path):
        server, stub = _build_server_with_chunking(
            tmp_path, enabled=False, threshold=10,
        )
        with TestClient(server.app) as _:
            await server._file_store.save(  # type: ignore[union-attr]
                FileRecord(
                    file_id="f1", filename="doc.txt", mime_type="text/plain",
                    size_bytes=5, sha256="",
                    extracted_text="full text wins when chunking off",
                    parse_status="completed",
                    chunk_status="skipped",
                    chunk_count=0,
                ),
                b"hello",
            )
            msgs = await server._resolve_file_attachments(
                ["f1"], last_user_message="query",
            )
        assert "full text wins" in msgs[0]["content"]


# ---------------------------------------------------------------------------
# Background task draining at shutdown
# ---------------------------------------------------------------------------


class TestShutdownDrainsTasks:
    def test_pending_tasks_drained_in_lifespan(self, tmp_path):
        server, stub = _build_server_with_chunking(
            tmp_path, enabled=True, threshold=10, chunk_size=20,
        )
        # Make the stub's save coroutine slow so a task is still in
        # flight at shutdown.
        async def slow_save(file_id, chunks, *, user_id, session_id=None):
            await asyncio.sleep(0.1)
            stub.saved[file_id] = list(chunks)
            return len(chunks)
        stub.save_chunks = AsyncMock(side_effect=slow_save)

        with TestClient(server.app) as client:
            server._chunk_store = stub
            big = b"word " * 200
            client.post(
                "/v1/files",
                files={"file": ("doc.txt", big, "text/plain")},
            )
        # After context exit, lifespan shutdown ran. The slow save
        # should have completed before the chunk store was closed.
        assert stub.save_chunks.await_count >= 1
