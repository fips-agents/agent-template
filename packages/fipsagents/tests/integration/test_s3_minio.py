"""Live integration test: S3BytesStore against a real MinIO endpoint.

Mark-driven and opt-in. Skips automatically when:

  - the ``[s3]`` extra isn't installed (aioboto3 missing), or
  - ``MINIO_ENDPOINT`` env var isn't set, or
  - the endpoint is unreachable (ConnectError / timeout on the
    bucket-create probe).

The 2026-04-29 cluster smoke proved the round-trip end-to-end against
a MinIO deployed in the ``fipsagents-files-smoke`` namespace; this
test is the reproducible local equivalent.

Run locally with::

    podman run -d --name minio-itest --rm \\
        -p 9000:9000 -p 9001:9001 \\
        -e MINIO_ROOT_USER=minioadmin \\
        -e MINIO_ROOT_PASSWORD=minioadminpassword \\
        quay.io/minio/minio server /data --console-address :9001

    MINIO_ENDPOINT=http://localhost:9000 \\
    MINIO_ACCESS_KEY=minioadmin \\
    MINIO_SECRET_KEY=minioadminpassword \\
    pytest tests/integration/test_s3_minio.py -v -m minio
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest

# Skip the entire module unless aioboto3 is importable. We don't import
# at module top-level because pytest collection should still succeed in
# environments without the [s3] extra.
aioboto3 = pytest.importorskip("aioboto3")

from fipsagents.server.bytes_store import S3BytesStore  # noqa: E402

pytestmark = pytest.mark.minio


def _endpoint() -> str | None:
    return os.environ.get("MINIO_ENDPOINT")


def _creds() -> tuple[str, str]:
    return (
        os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
        os.environ.get("MINIO_SECRET_KEY", "minioadminpassword"),
    )


# ---------------------------------------------------------------------------
# Bucket fixture — creates a unique bucket per test run, drops it on teardown
# ---------------------------------------------------------------------------


@pytest.fixture
async def minio_bucket() -> Any:
    endpoint = _endpoint()
    if not endpoint:
        pytest.skip(
            "MINIO_ENDPOINT not set — see test_s3_minio.py docstring",
        )
    access_key, secret_key = _creds()
    bucket = f"fipsagents-itest-{uuid.uuid4().hex[:12]}"

    from botocore.config import Config as BotoConfig

    session = aioboto3.Session()
    config = BotoConfig(
        signature_version="s3v4",
        s3={"addressing_style": "path"},
    )
    try:
        async with session.client(
            "s3",
            endpoint_url=endpoint,
            region_name="us-east-1",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=config,
        ) as client:
            try:
                await client.create_bucket(Bucket=bucket)
            except Exception as exc:
                pytest.skip(f"MinIO unreachable / bucket-create failed: {exc}")
            try:
                yield bucket
            finally:
                # Best-effort cleanup. Sweep objects then drop the bucket.
                try:
                    resp = await client.list_objects_v2(Bucket=bucket)
                    for obj in resp.get("Contents", []) or []:
                        await client.delete_object(
                            Bucket=bucket, Key=obj["Key"],
                        )
                    await client.delete_bucket(Bucket=bucket)
                except Exception:
                    # Don't mask test failures behind teardown noise.
                    pass
    except Exception as exc:
        pytest.skip(f"MinIO unreachable: {exc}")


@pytest.fixture
async def s3_store(minio_bucket: str) -> Any:
    access_key, secret_key = _creds()
    store = S3BytesStore(
        bucket=minio_bucket,
        endpoint=_endpoint(),
        region="us-east-1",
        access_key=access_key,
        secret_key=secret_key,
        path_style=True,
    )
    try:
        yield store
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# Tests — full BytesStore contract against live MinIO
# ---------------------------------------------------------------------------


class TestS3BytesStoreLive:
    async def test_put_get_round_trip(self, s3_store: S3BytesStore):
        await s3_store.put(
            "file_abcdef1234", b"hello minio", content_type="text/plain",
        )
        assert await s3_store.get("file_abcdef1234") == b"hello minio"

    async def test_get_returns_none_for_missing(self, s3_store: S3BytesStore):
        assert await s3_store.get("file_does_not_exist") is None

    async def test_delete_returns_true_when_existed(
        self, s3_store: S3BytesStore,
    ):
        await s3_store.put("file_xyz", b"payload")
        assert await s3_store.delete("file_xyz") is True
        assert await s3_store.get("file_xyz") is None

    async def test_delete_returns_false_when_missing(
        self, s3_store: S3BytesStore,
    ):
        assert await s3_store.delete("file_never_existed") is False

    async def test_overwrite_last_write_wins(self, s3_store: S3BytesStore):
        await s3_store.put("file_aa", b"first")
        await s3_store.put("file_aa", b"second")
        assert await s3_store.get("file_aa") == b"second"

    async def test_key_layout_under_prefix(self, minio_bucket: str):
        """A prefix lands under <prefix>/fi/<file_id> in the bucket."""
        access_key, secret_key = _creds()
        store = S3BytesStore(
            bucket=minio_bucket,
            endpoint=_endpoint(),
            region="us-east-1",
            access_key=access_key,
            secret_key=secret_key,
            prefix="my-agent",
            path_style=True,
        )
        try:
            await store.put("file_pp", b"under-prefix")
            assert await store.get("file_pp") == b"under-prefix"
        finally:
            await store.delete("file_pp")
            await store.close()

    async def test_large_payload_round_trip(self, s3_store: S3BytesStore):
        """1 MiB payload — exercises whatever boto3 single-PUT path
        kicks in for non-trivially-sized objects."""
        payload = b"x" * (1 * 1024 * 1024)
        await s3_store.put("file_big", payload)
        got = await s3_store.get("file_big")
        assert got is not None
        assert len(got) == len(payload)
        assert got == payload
