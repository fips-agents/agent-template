"""Tests for the BytesStore ABC and its implementations.

Per ADR-0001, file metadata and file bytes are split. Most coverage of
the integration paths (FileStore + BytesStore composition) lives in
``test_files_store.py`` and ``test_files_endpoint.py``; this file
exercises the bytes-only contract directly so each implementation has
its own unit-level proof.

The S3 path uses a mocked aioboto3 client; the live MinIO integration
test lives in ``tests/integration/test_s3_minio.py`` (mark-driven, opt-in).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from fipsagents.server.bytes_store import (
    BytesStore,
    LocalFsBytesStore,
    NullBytesStore,
    S3BytesStore,
    _shard_key,
    create_bytes_store,
)


# ---------------------------------------------------------------------------
# _shard_key — keyspace layout is the cross-backend contract
# ---------------------------------------------------------------------------


class TestShardKey:
    def test_two_char_prefix(self):
        assert _shard_key("file_abcdef1234") == "fi/file_abcdef1234"

    def test_short_id_pads(self):
        assert _shard_key("a") == "00/a"

    def test_id_without_prefix(self):
        assert _shard_key("xyz") == "xy/xyz"


# ---------------------------------------------------------------------------
# NullBytesStore
# ---------------------------------------------------------------------------


class TestNullBytesStore:
    async def test_put_then_get_returns_none(self):
        store = NullBytesStore()
        await store.put("file_abc", b"payload")
        assert await store.get("file_abc") is None

    async def test_delete_returns_false(self):
        store = NullBytesStore()
        assert await store.delete("file_abc") is False


# ---------------------------------------------------------------------------
# LocalFsBytesStore
# ---------------------------------------------------------------------------


class TestLocalFsBytesStore:
    async def test_put_creates_sharded_path(self, tmp_path: Path):
        store = LocalFsBytesStore(str(tmp_path))
        await store.put("file_abcdef1234", b"hello")
        # Layout: <root>/fi/file_abcdef1234
        assert (tmp_path / "fi" / "file_abcdef1234").read_bytes() == b"hello"

    async def test_get_round_trip(self, tmp_path: Path):
        store = LocalFsBytesStore(str(tmp_path))
        await store.put("file_x9", b"data")
        assert await store.get("file_x9") == b"data"

    async def test_get_returns_none_for_missing(self, tmp_path: Path):
        store = LocalFsBytesStore(str(tmp_path))
        assert await store.get("file_missing") is None

    async def test_delete_returns_true_for_present(self, tmp_path: Path):
        store = LocalFsBytesStore(str(tmp_path))
        await store.put("file_aa", b"x")
        assert await store.delete("file_aa") is True
        assert await store.get("file_aa") is None

    async def test_delete_returns_false_for_missing(self, tmp_path: Path):
        store = LocalFsBytesStore(str(tmp_path))
        assert await store.delete("file_zz") is False

    async def test_delete_cleans_empty_shard_dir(self, tmp_path: Path):
        store = LocalFsBytesStore(str(tmp_path))
        await store.put("file_aa", b"x")
        shard = tmp_path / "fi"
        assert shard.exists()
        await store.delete("file_aa")
        # Shard dir is empty now → cleaned up best-effort.
        assert not shard.exists()

    async def test_put_atomic_replace(self, tmp_path: Path):
        """Last-write-wins; first put then second put → second wins."""
        store = LocalFsBytesStore(str(tmp_path))
        await store.put("file_a", b"first")
        await store.put("file_a", b"second")
        assert await store.get("file_a") == b"second"


# ---------------------------------------------------------------------------
# S3BytesStore — mocked aioboto3 client (live MinIO test is integration)
# ---------------------------------------------------------------------------


class _FakeS3Body:
    """Mimics aioboto3's StreamingBody.read() coroutine."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    async def read(self) -> bytes:
        return self._data

    async def close(self) -> None:
        return None


class _FakeS3Error(Exception):
    """Mimics botocore.exceptions.ClientError shape."""

    def __init__(self, code: str) -> None:
        self.response = {"Error": {"Code": code}}
        super().__init__(code)


class _FakeS3Client:
    """Minimal in-memory stand-in for aioboto3's async S3 client.

    Captures the calls we care about (put_object, get_object,
    head_object, delete_object) so the test can assert on Bucket / Key
    shape and NoSuchKey handling without spinning up MinIO.
    """

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.put_calls: list[dict[str, Any]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def put_object(self, **kwargs):
        self.put_calls.append(kwargs)
        self.objects[(kwargs["Bucket"], kwargs["Key"])] = kwargs["Body"]

    async def get_object(self, *, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise _FakeS3Error("NoSuchKey")
        return {"Body": _FakeS3Body(self.objects[(Bucket, Key)])}

    async def head_object(self, *, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise _FakeS3Error("404")
        return {"ContentLength": len(self.objects[(Bucket, Key)])}

    async def delete_object(self, *, Bucket, Key):
        self.objects.pop((Bucket, Key), None)


class TestS3BytesStore:
    def _patch_client(self, store: S3BytesStore, fake: _FakeS3Client) -> None:
        # Skip __aenter__ by injecting the fake directly; a real client
        # is gated behind the [s3] extra and would import aioboto3.
        store._client = fake  # type: ignore[attr-defined]
        store._client_cm = fake  # type: ignore[attr-defined]

    def test_constructor_requires_bucket(self):
        with pytest.raises(ValueError, match="non-empty bucket"):
            S3BytesStore(bucket="")

    def test_key_layout_includes_shard(self):
        store = S3BytesStore(bucket="b", prefix="my-agent")
        assert store._key("file_abcdef1234") == "my-agent/fi/file_abcdef1234"

    def test_key_layout_no_prefix(self):
        store = S3BytesStore(bucket="b")
        assert store._key("file_abcdef1234") == "fi/file_abcdef1234"

    async def test_put_and_get_round_trip(self):
        store = S3BytesStore(bucket="bx")
        fake = _FakeS3Client()
        self._patch_client(store, fake)
        await store.put("file_aa", b"hello", content_type="text/plain")
        assert fake.put_calls[0]["Bucket"] == "bx"
        assert fake.put_calls[0]["Key"] == "fi/file_aa"
        assert fake.put_calls[0]["ContentType"] == "text/plain"
        assert await store.get("file_aa") == b"hello"

    async def test_get_returns_none_on_nosuchkey(self):
        store = S3BytesStore(bucket="bx")
        self._patch_client(store, _FakeS3Client())
        assert await store.get("file_missing") is None

    async def test_delete_returns_true_when_existed(self):
        store = S3BytesStore(bucket="bx")
        fake = _FakeS3Client()
        self._patch_client(store, fake)
        await store.put("file_aa", b"x")
        assert await store.delete("file_aa") is True
        assert ("bx", "fi/file_aa") not in fake.objects

    async def test_delete_returns_false_when_missing(self):
        store = S3BytesStore(bucket="bx")
        self._patch_client(store, _FakeS3Client())
        assert await store.delete("file_zz") is False


# ---------------------------------------------------------------------------
# create_bytes_store factory
# ---------------------------------------------------------------------------


class TestCreateBytesStore:
    def test_local_fs_default(self, tmp_path: Path):
        store = create_bytes_store(None, bytes_dir=str(tmp_path))
        assert isinstance(store, LocalFsBytesStore)

    def test_local_fs_explicit(self, tmp_path: Path):
        store = create_bytes_store("local_fs", bytes_dir=str(tmp_path))
        assert isinstance(store, LocalFsBytesStore)

    def test_null(self):
        assert isinstance(create_bytes_store("null"), NullBytesStore)

    def test_s3_requires_bucket(self):
        with pytest.raises(ValueError, match="bucket"):
            create_bytes_store("s3")

    def test_s3_with_bucket(self):
        store = create_bytes_store("s3", s3_bucket="my-bucket")
        assert isinstance(store, S3BytesStore)

    def test_unknown_backend_rejects(self):
        with pytest.raises(ValueError, match="unknown bytes_backend"):
            create_bytes_store("nfs")


# ---------------------------------------------------------------------------
# Cross-backend BytesStore contract sanity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [
        lambda tmp_path: LocalFsBytesStore(str(tmp_path)),
        lambda tmp_path: NullBytesStore(),
    ],
    ids=["local_fs", "null"],
)
class TestBytesStoreContract:
    """Properties every BytesStore should satisfy. Skips Null where the
    contract differs intentionally (accept-and-discard)."""

    async def test_close_does_not_raise(self, factory, tmp_path: Path):
        store: BytesStore = factory(tmp_path)
        await store.close()
        # Idempotency — second close should also be safe.
        await store.close()
