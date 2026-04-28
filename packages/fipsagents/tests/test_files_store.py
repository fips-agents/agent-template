"""Tests for file persistence backends."""


import hashlib
import os
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

from fipsagents.server.files import (
    FileRecord,
    NullFileStore,
    SqliteFileStore,
    _bytes_path,
    _generate_file_id,
    _sha256,
    create_file_store,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    *,
    file_id: str | None = None,
    data: bytes = b"hello world",
    filename: str = "hello.txt",
    mime_type: str = "text/plain",
    session_id: str | None = None,
    user_id: str = "anonymous",
) -> FileRecord:
    fid = file_id or _generate_file_id()
    return FileRecord(
        file_id=fid,
        filename=filename,
        mime_type=mime_type,
        size_bytes=len(data),
        sha256=_sha256(data),
        user_id=user_id,
        session_id=session_id,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def sqlite_store(tmp_path):
    store = SqliteFileStore(
        str(tmp_path / "test.db"),
        bytes_dir=str(tmp_path / "files"),
    )
    yield store
    await store.close()


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


class TestModuleHelpers:
    def test_generate_file_id_format(self):
        fid = _generate_file_id()
        assert fid.startswith("file_")
        assert len(fid) == len("file_") + 24

    def test_generate_file_id_unique(self):
        ids = {_generate_file_id() for _ in range(100)}
        assert len(ids) == 100

    def test_sha256_deterministic(self):
        assert _sha256(b"hello") == hashlib.sha256(b"hello").hexdigest()

    def test_bytes_path_sharded(self):
        path = _bytes_path("/tmp/files", "file_abcdef1234567890")
        assert path == os.path.join("/tmp/files", "fi", "file_abcdef1234567890")

    def test_bytes_path_short_id_safe(self):
        # Pathological case — file_id shorter than 2 chars.
        path = _bytes_path("/tmp/files", "x")
        assert path.startswith("/tmp/files/00")


# ---------------------------------------------------------------------------
# NullFileStore
# ---------------------------------------------------------------------------


class TestNullFileStore:
    @pytest.mark.asyncio
    async def test_save_returns_id(self):
        store = NullFileStore()
        rec = _make_record()
        result = await store.save(rec, b"hello world")
        assert result == rec.file_id

    @pytest.mark.asyncio
    async def test_get_metadata_returns_none(self):
        store = NullFileStore()
        assert await store.get_metadata("anything") is None

    @pytest.mark.asyncio
    async def test_get_bytes_returns_none(self):
        store = NullFileStore()
        assert await store.get_bytes("anything") is None

    @pytest.mark.asyncio
    async def test_get_extracted_text_returns_none(self):
        store = NullFileStore()
        assert await store.get_extracted_text("anything") is None

    @pytest.mark.asyncio
    async def test_update_extracted_text_returns_false(self):
        store = NullFileStore()
        assert (
            await store.update_extracted_text("x", extracted_text="t") is False
        )

    @pytest.mark.asyncio
    async def test_list_for_session_empty(self):
        store = NullFileStore()
        assert await store.list_for_session("any") == []

    @pytest.mark.asyncio
    async def test_delete_returns_false(self):
        store = NullFileStore()
        assert await store.delete("anything") is False

    @pytest.mark.asyncio
    async def test_delete_before_returns_zero(self):
        store = NullFileStore()
        assert await store.delete_before(datetime.now(timezone.utc)) == 0


# ---------------------------------------------------------------------------
# SqliteFileStore — round-trip
# ---------------------------------------------------------------------------


class TestSqliteFileStoreRoundTrip:
    @pytest.mark.asyncio
    async def test_save_and_get_metadata(self, sqlite_store):
        rec = _make_record(filename="doc.pdf", mime_type="application/pdf")
        await sqlite_store.save(rec, b"hello world")

        loaded = await sqlite_store.get_metadata(rec.file_id)
        assert loaded is not None
        assert loaded.file_id == rec.file_id
        assert loaded.filename == "doc.pdf"
        assert loaded.mime_type == "application/pdf"
        assert loaded.size_bytes == 11
        assert loaded.sha256 == _sha256(b"hello world")
        assert loaded.parse_status == "pending"
        assert loaded.extracted_text is None

    @pytest.mark.asyncio
    async def test_save_and_get_bytes(self, sqlite_store):
        data = b"\x00\x01\x02 binary content \xff\xfe"
        rec = _make_record(data=data)
        await sqlite_store.save(rec, data)

        loaded = await sqlite_store.get_bytes(rec.file_id)
        assert loaded == data

    @pytest.mark.asyncio
    async def test_get_metadata_unknown_returns_none(self, sqlite_store):
        assert await sqlite_store.get_metadata("file_does_not_exist") is None

    @pytest.mark.asyncio
    async def test_get_bytes_unknown_returns_none(self, sqlite_store):
        assert await sqlite_store.get_bytes("file_does_not_exist") is None

    @pytest.mark.asyncio
    async def test_save_size_mismatch_raises(self, sqlite_store):
        rec = _make_record(data=b"hello")
        # Lie about the size.
        rec.size_bytes = 999
        with pytest.raises(ValueError, match="size_bytes mismatch"):
            await sqlite_store.save(rec, b"hello")

    @pytest.mark.asyncio
    async def test_save_sha_mismatch_raises(self, sqlite_store):
        rec = _make_record(data=b"hello")
        rec.sha256 = "0" * 64  # wrong hash
        with pytest.raises(ValueError, match="sha256 mismatch"):
            await sqlite_store.save(rec, b"hello")

    @pytest.mark.asyncio
    async def test_save_fills_in_empty_sha(self, sqlite_store):
        rec = _make_record(data=b"hello")
        rec.sha256 = ""  # caller didn't pre-compute
        await sqlite_store.save(rec, b"hello")
        loaded = await sqlite_store.get_metadata(rec.file_id)
        assert loaded.sha256 == _sha256(b"hello")

    @pytest.mark.asyncio
    async def test_bytes_stored_under_sharded_path(self, sqlite_store, tmp_path):
        rec = _make_record()
        await sqlite_store.save(rec, b"hello world")
        # Bytes should live under <bytes_dir>/<2-char-shard>/<file_id>.
        expected = _bytes_path(str(tmp_path / "files"), rec.file_id)
        assert os.path.exists(expected)


# ---------------------------------------------------------------------------
# SqliteFileStore — extracted text lifecycle
# ---------------------------------------------------------------------------


class TestSqliteFileStoreExtractedText:
    @pytest.mark.asyncio
    async def test_get_extracted_text_initially_none(self, sqlite_store):
        rec = _make_record()
        await sqlite_store.save(rec, b"hello world")
        assert await sqlite_store.get_extracted_text(rec.file_id) is None

    @pytest.mark.asyncio
    async def test_update_extracted_text_completed(self, sqlite_store):
        rec = _make_record()
        await sqlite_store.save(rec, b"hello world")

        result = await sqlite_store.update_extracted_text(
            rec.file_id,
            extracted_text="Parsed body text.",
            parse_status="completed",
        )
        assert result is True

        text = await sqlite_store.get_extracted_text(rec.file_id)
        assert text == "Parsed body text."

        meta = await sqlite_store.get_metadata(rec.file_id)
        assert meta.parse_status == "completed"
        assert meta.parse_error is None

    @pytest.mark.asyncio
    async def test_update_extracted_text_failed(self, sqlite_store):
        rec = _make_record()
        await sqlite_store.save(rec, b"hello world")

        result = await sqlite_store.update_extracted_text(
            rec.file_id,
            parse_status="failed",
            parse_error="parser exploded",
        )
        assert result is True

        meta = await sqlite_store.get_metadata(rec.file_id)
        assert meta.parse_status == "failed"
        assert meta.parse_error == "parser exploded"
        assert meta.extracted_text is None

    @pytest.mark.asyncio
    async def test_update_unknown_file_returns_false(self, sqlite_store):
        result = await sqlite_store.update_extracted_text(
            "file_missing", extracted_text="x"
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_update_no_fields_returns_existence(self, sqlite_store):
        rec = _make_record()
        await sqlite_store.save(rec, b"hello world")
        # No fields to set, but the file exists.
        assert await sqlite_store.update_extracted_text(rec.file_id) is True
        assert await sqlite_store.update_extracted_text("file_nope") is False


# ---------------------------------------------------------------------------
# SqliteFileStore — session listing
# ---------------------------------------------------------------------------


class TestSqliteFileStoreListForSession:
    @pytest.mark.asyncio
    async def test_list_returns_only_session_files(self, sqlite_store):
        s1_files = []
        for i in range(3):
            rec = _make_record(
                filename=f"a{i}.txt",
                session_id="sess_a",
                data=f"body {i}".encode(),
            )
            await sqlite_store.save(rec, f"body {i}".encode())
            s1_files.append(rec.file_id)

        rec_other = _make_record(
            filename="b.txt", session_id="sess_b", data=b"other",
        )
        await sqlite_store.save(rec_other, b"other")

        listed = await sqlite_store.list_for_session("sess_a")
        assert len(listed) == 3
        assert all(r.session_id == "sess_a" for r in listed)
        assert {r.file_id for r in listed} == set(s1_files)

    @pytest.mark.asyncio
    async def test_list_orders_newest_first(self, sqlite_store):
        # created_at is set at construction; force distinct timestamps.
        rec1 = _make_record(filename="old.txt", session_id="sess_x")
        rec1.created_at = "2024-01-01T00:00:00+00:00"
        await sqlite_store.save(rec1, b"hello world")

        rec2 = _make_record(filename="new.txt", session_id="sess_x")
        rec2.created_at = "2024-12-31T23:59:59+00:00"
        await sqlite_store.save(rec2, b"hello world")

        listed = await sqlite_store.list_for_session("sess_x")
        assert [r.filename for r in listed] == ["new.txt", "old.txt"]

    @pytest.mark.asyncio
    async def test_list_respects_limit_and_offset(self, sqlite_store):
        for i in range(5):
            rec = _make_record(filename=f"f{i}.txt", session_id="sess_p")
            rec.created_at = f"2024-01-0{i+1}T00:00:00+00:00"
            await sqlite_store.save(rec, b"hello world")

        page1 = await sqlite_store.list_for_session("sess_p", limit=2)
        page2 = await sqlite_store.list_for_session(
            "sess_p", limit=2, offset=2,
        )
        assert len(page1) == 2
        assert len(page2) == 2
        assert {r.file_id for r in page1}.isdisjoint({r.file_id for r in page2})

    @pytest.mark.asyncio
    async def test_list_empty_for_unknown_session(self, sqlite_store):
        assert await sqlite_store.list_for_session("sess_nope") == []


# ---------------------------------------------------------------------------
# SqliteFileStore — delete + housekeeping
# ---------------------------------------------------------------------------


class TestSqliteFileStoreDelete:
    @pytest.mark.asyncio
    async def test_delete_removes_metadata_and_bytes(
        self, sqlite_store, tmp_path,
    ):
        rec = _make_record()
        await sqlite_store.save(rec, b"hello world")
        path = _bytes_path(str(tmp_path / "files"), rec.file_id)
        assert os.path.exists(path)

        assert await sqlite_store.delete(rec.file_id) is True
        assert await sqlite_store.get_metadata(rec.file_id) is None
        assert await sqlite_store.get_bytes(rec.file_id) is None
        assert not os.path.exists(path)

    @pytest.mark.asyncio
    async def test_delete_unknown_returns_false(self, sqlite_store):
        assert await sqlite_store.delete("file_nope") is False

    @pytest.mark.asyncio
    async def test_delete_bytes_missing_still_succeeds(
        self, sqlite_store, tmp_path,
    ):
        # Race / external cleanup: bytes vanished before delete().
        rec = _make_record()
        await sqlite_store.save(rec, b"hello world")
        path = _bytes_path(str(tmp_path / "files"), rec.file_id)
        os.remove(path)

        assert await sqlite_store.delete(rec.file_id) is True
        assert await sqlite_store.get_metadata(rec.file_id) is None

    @pytest.mark.asyncio
    async def test_delete_before_removes_old_files(self, sqlite_store):
        old = _make_record(filename="old.txt")
        old.created_at = "2024-01-01T00:00:00+00:00"
        await sqlite_store.save(old, b"hello world")

        new = _make_record(filename="new.txt")
        # Default created_at is "now" — naturally newer than the cutoff.
        await sqlite_store.save(new, b"hello world")

        cutoff = datetime(2024, 6, 1, tzinfo=timezone.utc)
        deleted = await sqlite_store.delete_before(cutoff)
        assert deleted == 1
        assert await sqlite_store.get_metadata(old.file_id) is None
        assert await sqlite_store.get_metadata(new.file_id) is not None

    @pytest.mark.asyncio
    async def test_delete_before_no_matches(self, sqlite_store):
        rec = _make_record()
        await sqlite_store.save(rec, b"hello world")

        long_ago = datetime.now(timezone.utc) - timedelta(days=3650)
        deleted = await sqlite_store.delete_before(long_ago)
        assert deleted == 0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreateFileStore:
    def test_null_when_backend_is_none(self):
        store = create_file_store(None)
        assert isinstance(store, NullFileStore)

    def test_sqlite_backend(self, tmp_path):
        store = create_file_store(
            "sqlite",
            sqlite_path=str(tmp_path / "x.db"),
            bytes_dir=str(tmp_path / "files"),
        )
        assert isinstance(store, SqliteFileStore)

    def test_unimplemented_backends_raise(self):
        with pytest.raises(NotImplementedError, match="postgres"):
            create_file_store("postgres")
        with pytest.raises(NotImplementedError, match="http"):
            create_file_store("http")
