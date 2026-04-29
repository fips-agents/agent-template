# ADR-0001: S3-Compatible Bytes Backend for FileStore

- **Status**: Accepted
- **Date**: 2026-04-28
- **Deciders**: rdwj
- **Related issues**: [agent-template#100](https://github.com/fips-agents/agent-template/issues/100)
- **Related code**: `packages/fipsagents/src/fipsagents/server/files.py`, `FilesConfig` in `packages/fipsagents/src/fipsagents/baseagent/config.py`

## Context

The File Upload track shipped in `fipsagents 0.16.0` (`SqliteFileStore`, `PostgresFileStore`, `NullFileStore`) treats *file metadata* and *file bytes* as a single concern owned by each `FileStore` implementation. Both `SqliteFileStore` and `PostgresFileStore` keep metadata in their respective databases and write bytes to a sharded local-FS directory configured by `FilesConfig.bytes_dir`. This is fine for development and for a single replica behind a ReadWriteOnce PVC, but it does not work for multi-replica enterprise deployments and it does not satisfy the MinIO-as-object-store promise documented in Module 9 of the examples site.

Two production scenarios force a redesign:

1. **Multi-replica deployments**. Multiple agent pods need to read the same uploaded bytes. RWO PVCs cannot be mounted across pods; RWX PVCs are uneven across cloud providers and storage classes; bytes-on-NFS is operationally fragile. Object storage (MinIO on-cluster, S3 / Azure Blob / GCS off-cluster) is the standard answer.
2. **Bytes/metadata lifecycle divergence**. Metadata is small, transactional, and audit-relevant; bytes are large, content-addressable, and benefit from object-store features (lifecycle rules, versioning, server-side encryption, replication, signed URLs). Shoehorning both into the same store class duplicates orthogonal concerns.

We need to commit to a shape *before* writing the S3 backend so the API surface, configuration shape, and migration path are stable.

## Decision

**Option (a): split bytes from metadata via a `BytesStore` ABC. `SqliteFileStore` and `PostgresFileStore` compose with a `BytesStore` instead of owning the local-FS path themselves.**

```
FileStore (metadata)         BytesStore (bytes)
├── NullFileStore            ├── NullBytesStore
├── SqliteFileStore   ──┐    ├── LocalFsBytesStore
└── PostgresFileStore  ─┴──> └── S3BytesStore
                                 (and MinIO via S3 protocol)
```

`FilesConfig` gains a `bytes_backend` discriminated union (`local_fs` | `s3` | `null`). `create_file_store()` builds a `BytesStore` from `bytes_backend` config and threads it into the chosen `FileStore` constructor. Existing `SqliteFileStore(db_path=..., bytes_dir=...)` and `PostgresFileStore(database_url=..., bytes_dir=...)` keep their current signatures via a thin shim that constructs a `LocalFsBytesStore(bytes_dir)` internally — backward-compatible by construction.

## Alternatives Considered

### Option (b): variant classes — `S3FileStore`, `MinIOFileStore`

Each new combination of metadata + bytes location ships as a new `FileStore` implementation. To get "Postgres metadata + S3 bytes" the user picks `PostgresS3FileStore`; to get "Postgres metadata + MinIO" the user picks `PostgresMinIOFileStore`; etc.

- **Pro**: Each class is self-contained — no composition, no factory wiring.
- **Pro**: Slightly easier to reason about at a single call site (one class, one schema, one save path).
- **Con**: Combinatorial explosion. Each new bytes target multiplies the class count. Postgres × {local, S3, MinIO, GCS, Azure, custom} = 6 classes for one metadata backend.
- **Con**: Duplicates the metadata path. Postgres SQL is the same regardless of where bytes live; copy-pasting the schema, pool management, and shutdown logic across `PostgresFileStore` / `PostgresS3FileStore` / `PostgresMinIOFileStore` invites drift.
- **Con**: Doesn't model what's actually orthogonal. Metadata storage and bytes storage are independent operational decisions — pricing, replication, access patterns, lifecycle rules don't overlap. Forcing them into a single class hides that.

### Option (c): keep bytes inside the metadata DB (no separate bytes store)

Store bytes as a `BYTEA` column in Postgres or a `BLOB` in SQLite. Metadata and bytes share lifecycle, single transaction guarantee.

- **Pro**: Atomic save/delete by construction.
- **Pro**: No extra moving part.
- **Con**: Postgres `BYTEA` is bandwidth-poor and fights TOAST for large rows. SQLite `BLOB` is fine for dev but doesn't help production.
- **Con**: Misses the whole point — object storage is what enterprise deployments want for files. Punts on the original problem.
- **Con**: Database backup/restore now scales with file content, not just metadata.

Rejected. Mentioned for completeness; not a serious contender.

## Consequences

### Positive

- **One metadata path, many bytes targets.** Postgres metadata code is written once. Adding GCS or Azure Blob later means one new `BytesStore` implementation, no new `FileStore`.
- **Clear configuration shape.** `metadata.backend: postgres` + `bytes_backend: { type: s3, endpoint: ..., bucket: ... }` reads naturally and matches how operators actually think about deployment topology.
- **MinIO promise concrete.** Module 9 of the examples site documents MinIO as the future target. Option (a) gives a one-line answer: "set `bytes_backend.type: s3` with `endpoint: http://minio.<ns>.svc:9000`."
- **Backward-compatible.** `FilesConfig.bytes_dir` continues to work when `bytes_backend` is unset — `create_file_store` constructs a `LocalFsBytesStore(bytes_dir)` and threads it in. Existing 0.16.0 deployments need no config changes.
- **Testability.** `NullBytesStore` and `LocalFsBytesStore` exist for unit tests; `S3BytesStore` is exercised against MinIO in integration tests. The `FileStore` test suite stays storage-agnostic.

### Negative

- **One more abstraction.** `FileStore` no longer owns the bytes path directly — implementations compose with `BytesStore`. New contributors have to read both ABCs to understand a save.
- **Atomicity is now a two-step.** `FileStore.save(record, data)` must coordinate `BytesStore.put` and `<metadata-store>.insert`. Failure modes:
  - Bytes written, metadata insert fails → orphan bytes. Mitigated by housekeeping (delete bytes whose metadata doesn't exist after a grace window).
  - Metadata inserted, bytes write fails → orphan metadata pointing at nothing. Mitigated by writing bytes first, then metadata (the existing pattern in `SqliteFileStore`/`PostgresFileStore` already does this).
- **Configuration surface grows.** Operators now configure both metadata backend and bytes backend. Defaults must be sensible: when `files.enabled: true` and no `bytes_backend` is set, fall back to `local_fs` with the existing `bytes_dir` default.
- **S3 retries / partial uploads.** Object stores need explicit retry, multipart upload thresholds, and content-MD5 verification. None of this exists in the local-FS path. `S3BytesStore` will be ~150-300 lines of new code with its own test suite.

## Implementation Sketch

`packages/fipsagents/src/fipsagents/server/bytes_store.py` (new):

```python
class BytesStore(ABC):
    @abstractmethod
    async def put(self, file_id: str, data: bytes, *, content_type: str | None = None) -> None: ...

    @abstractmethod
    async def get(self, file_id: str) -> bytes | None: ...

    @abstractmethod
    async def delete(self, file_id: str) -> bool: ...

    async def close(self) -> None: ...


class NullBytesStore(BytesStore): ...

class LocalFsBytesStore(BytesStore):
    """Sharded local-filesystem bytes (extracted from current SqliteFileStore/PostgresFileStore)."""
    def __init__(self, bytes_dir: str) -> None: ...

class S3BytesStore(BytesStore):
    """S3-compatible bytes (AWS S3, MinIO, GCS S3-mode, Backblaze B2, Cloudflare R2)."""
    def __init__(
        self,
        *,
        endpoint: str | None,           # None for AWS; set for MinIO/etc.
        region: str,
        bucket: str,
        access_key: str | None = None,  # None → IAM role / env / metadata service
        secret_key: str | None = None,
        prefix: str = "",               # optional key prefix
        path_style: bool = False,       # MinIO needs True
    ) -> None: ...
```

Likely dependency: `aioboto3` (async wrapper over boto3). Optional extra: `pip install fipsagents[s3]`.

`FilesConfig` (extended):

```yaml
files:
  enabled: true
  backend: postgres                     # metadata: null | sqlite | postgres
  max_file_size_bytes: 52428800
  bytes_dir: ./files                    # legacy — used when bytes_backend.type == "local_fs"
  bytes_backend:
    type: s3                            # local_fs | s3 | null
    endpoint: http://minio.minio.svc:9000
    region: us-east-1
    bucket: fipsagents-files
    path_style: true
    access_key: ${S3_ACCESS_KEY}
    secret_key: ${S3_SECRET_KEY}
    prefix: ""
```

When `bytes_backend` is unset, the factory constructs `LocalFsBytesStore(bytes_dir)` for dev parity with 0.16.0.

`SqliteFileStore` / `PostgresFileStore` (refactored):

- Constructor signature `__init__(self, db_path/database_url, *, bytes_store: BytesStore)`.
- `bytes_dir=...` keyword preserved as a deprecated shim that wraps `LocalFsBytesStore` for one minor version, then is removed.
- `save()` calls `self._bytes_store.put(...)` then writes metadata. `delete()` calls metadata-delete then `self._bytes_store.delete(...)`. `get_bytes()` delegates to `self._bytes_store.get()`.
- `_bytes_path()` and the sharded-FS helpers move into `LocalFsBytesStore`.

Approximate diff size: −200 LOC (extract local-FS code) / +400 LOC (`BytesStore` ABC + `LocalFsBytesStore` + `S3BytesStore` + tests). Net +200.

## Migration

For deployments running 0.16.0 with `bytes_dir` on a PVC:

1. Stand up MinIO (or point at an existing S3 bucket).
2. Bulk-copy `<bytes_dir>/**` to the bucket preserving the sharded layout (`fi/file_<32 hex>` → `s3://<bucket>/<prefix>/fi/file_<32 hex>`). One-shot script; `aws s3 sync` works.
3. Update `agent.yaml`: add `bytes_backend: { type: s3, ... }`, optionally remove `bytes_dir`.
4. Restart the pod. `BytesStore.get()` will now read from S3.
5. Verify with a known `file_id` (`GET /v1/files/{id}`).
6. Decommission the PVC.

Metadata (`SqliteFileStore` rows / `PostgresFileStore` rows) is untouched — `bytes_path` column becomes vestigial and is ignored when `BytesStore` is in play. (Optional follow-up: drop the column in a later migration; not blocking.)

For green-field deployments: set `bytes_backend.type: s3` from day one; never provision a PVC.

## Out of Scope

- **Signed URL generation.** A future endpoint (`GET /v1/files/{id}/signed-url`) returning a presigned S3 URL would let clients download large files without proxying through the agent. Not part of this ADR — does not affect the storage contract.
- **Server-side encryption configuration.** Bucket-level SSE is the operator's concern; the agent does not stamp `x-amz-server-side-encryption` headers per object. If we ever need per-object SSE-C, it goes on `BytesStore.put()` as an optional kwarg.
- **Lifecycle rules.** Bucket lifecycle (auto-expire after N days) belongs in the bucket's S3 config, not in agent code. `FileStore.delete_before(cutoff)` continues to handle the metadata side.
- **Multi-region replication.** Same — operator concern, configured at the bucket level.
- **Chunking + pgvector retrieval for large files.** This is the next bullet on agent-template#100 and gets its own ADR. The bytes backend chosen here is orthogonal: chunks live in pgvector regardless of where the original bytes live.

## Follow-ups

- File a tracking issue: "feat(server): S3-compatible BytesStore (closes one bullet of #100)".
- Once shipped, update Module 9 of the examples site: replace "MinIO as future intent" with the actual MinIO config snippet.
- Add an integration test that runs the full upload → save → get round-trip against a containerized MinIO (`minio/minio` is fine; ~50 MB image).
- Decide whether `[s3]` extra is `aioboto3` or `aiobotocore` directly. `aioboto3` is the friendlier API; `aiobotocore` is lower-level but already a transitive dep of several Red Hat AI quickstart components.
