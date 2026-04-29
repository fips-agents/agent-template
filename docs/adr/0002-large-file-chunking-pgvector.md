# ADR-0002: Large-File Chunking and pgvector Retrieval for `/v1/files`

- **Status**: Accepted
- **Date**: 2026-04-29
- **Deciders**: rdwj
- **Related issues**: [agent-template#100](https://github.com/fips-agents/agent-template/issues/100)
- **Related ADRs**: [ADR-0001](0001-s3-bytes-backend.md) (orthogonal — chunking lives at the metadata layer regardless of where bytes live)
- **Related code**: `packages/fipsagents/src/fipsagents/server/files.py`, `packages/fipsagents/src/fipsagents/server/parser.py`, `packages/fipsagents/src/fipsagents/server/app.py::_resolve_file_attachments`, `packages/fipsagents/src/fipsagents/baseagent/memory_pgvector.py`

## Context

The File Upload track shipped in `fipsagents 0.16.0` injects the *entire* `extracted_text` of every referenced file into the chat-completion request as a `system` message (see `_resolve_file_attachments` in `app.py`). This works for short documents — meeting notes, code snippets, READMEs — but fails for production-shape inputs:

- A 200-page contract parsed by Docling produces ~80–120 K tokens of extracted text. Granite 8B's 32 K window cannot hold it; gpt-oss-20b's 128 K window holds it but at a cost of ~$0.02/turn just for the file context, repeated for every follow-up message in the session.
- A multi-megabyte CSV's plaintext extraction overflows even 128 K windows.
- Even when the file fits, dumping the entire document for *every* turn is wasteful: the user's question is usually about a specific section. RAG-style retrieval reads the right ~2 K tokens instead of the whole document.

We have two production-shape needs:

1. **Token-bound deployments** (Granite 8B / 32 K). Files larger than ~10 K tokens silently get truncated by the model or fail the request. Need a way to retrieve only the chunks relevant to the user's current message.
2. **Cost-bound deployments** (gpt-oss-20b / 128 K and larger). The full document fits, but injecting it on every turn is 10–50× the necessary context. Retrieval beats dumping.

Existing infrastructure makes this cheap to add:

- `PGVectorMemoryClient` already wires asyncpg + an OpenAI-compatible embeddings endpoint (`memory_pgvector.py`), already speaks pgvector's `<=>` cosine operator, and is already configured for the agent's deployed embedding model. The whole "embed + similarity search" path is solved.
- `FilesConfig` already has `backend: postgres` for metadata. When the chunked path is enabled, the same Postgres database can host the chunk table — no new infrastructure to provision.
- The cluster already has an embedding model deployed and the URL is in `cluster_endpoints`.

What's missing: the chunker (split `extracted_text` into retrievable units), the chunk table (parallel to `agent_memories` but scoped per `file_id`), and the retrieval branch in `_resolve_file_attachments`.

We need to commit to a shape *before* writing code so the chunker choice, schema, retrieval API, and config surface are stable.

## Decision

**Add a `ChunkStore` ABC composed into the metadata `FileStore`. On upload, when `extracted_text` exceeds a configurable threshold, chunk it and write embeddings to pgvector. At chat-completion time, when a referenced file has chunks, similarity-search against the user's last message and inject only the top-K chunks instead of the full text.**

```
FileStore (metadata)              ChunkStore (chunks + embeddings)
├── NullFileStore                 ├── NullChunkStore
├── SqliteFileStore  ──┐          ├── PgvectorChunkStore
└── PostgresFileStore ─┴────────> └── (future: opensearch, weaviate, qdrant)
```

`FilesConfig` gains a `chunking` block. When `chunking.enabled: false` (default), the existing 0.17.0 full-text path runs unchanged. When enabled, the server runs an async post-parse step that calls the chunker and writes to the configured `ChunkStore`.

### Chunking strategy

**Two-tier dispatch by parser output**, mirroring the existing two-tier parser pattern:

1. **Plaintext / structured-text outputs** (whatever `PlaintextParser` produced, plus Docling outputs that lack hierarchy): recursive token-bounded splitter. Split on paragraph boundaries first, then sentences, then hard-cut at the token cap. ~600 tokens per chunk, ~100 token overlap. Token counts via `tiktoken` (best-effort; falls back to char/4 heuristic when tiktoken is unavailable).

2. **Docling outputs with hierarchy** (PDF, DOCX, PPTX): use Docling's native `HybridChunker`. Preserves heading paths, page numbers, table cell groupings. Same ~600-token target, but boundaries snap to structural units. Falls back to the recursive splitter if Docling's chunker is unavailable.

The chunker is a `Chunker` ABC with two implementations: `RecursiveTokenChunker` (default, no extra deps) and `DoclingChunker` (auto-selected when the parser was Docling and the `[files]` extra is installed).

### Storage

`PgvectorChunkStore` writes to a `file_chunks` table, deliberately separate from `agent_memories`:

```sql
CREATE TABLE file_chunks (
    chunk_id        TEXT PRIMARY KEY,
    file_id         TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    session_id      TEXT,
    chunk_index     INT NOT NULL,
    content         TEXT NOT NULL,
    metadata        JSONB,
    embedding       vector({dimension}),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (file_id, chunk_index)
);
CREATE INDEX file_chunks_file_id_idx ON file_chunks (file_id);
CREATE INDEX file_chunks_embedding_idx
    ON file_chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
```

Lifecycle is tied to the `FileRecord`: `FileStore.delete()` cascades to `ChunkStore.delete_for_file(file_id)`. Foreign keys are *not* declared at the DB level — the metadata table can live in SQLite while chunks live in Postgres, so app-level cascade is the universal contract.

### Retrieval

`_resolve_file_attachments` extends with a third branch:

```
for each referenced file_id:
    record = file_store.get_metadata(file_id)
    if record.extracted_text is None:
        inject "content not available" header (existing behavior)
    elif record.chunk_count == 0 OR token_count(extracted_text) <= small_file_threshold:
        inject full extracted_text (existing behavior)
    else:
        # chunked path
        query = last user-role message content
        chunks = chunk_store.search(file_id, query, limit=K)
        inject "[Attached file: ...]\n<top-K chunks joined with '\n---\n'>"
```

The retrieval is **per-file**, not global — search is scoped to `file_id IN (referenced_files)` so a user asking about file A doesn't get chunks from file B that they didn't reference. This is also a hard requirement for multi-tenant deployments where the user's `file_ids` are the authorization boundary.

### Configuration shape

```yaml
files:
  enabled: true
  backend: postgres
  bytes_backend:
    type: s3
    # ... per ADR-0001
  chunking:
    enabled: true
    backend: pgvector                  # null | pgvector
    # When enabled=true and backend=pgvector, these fields are required:
    database_url: ${PGVECTOR_URL}      # may share files.database_url
    embedding_url: ${EMBEDDING_URL}    # OpenAI-compatible endpoint
    embedding_model: ${EMBEDDING_MODEL:-all-MiniLM-L6-v2}
    embedding_dimension: 768
    table_name: file_chunks
    # Behavior knobs (sensible defaults):
    chunk_size_tokens: 600
    chunk_overlap_tokens: 100
    small_file_threshold_tokens: 4000  # below this → inject full text, skip chunking
    retrieval_top_k: 5                 # chunks per file per turn
    retrieval_min_score: 0.0           # cosine similarity floor (0 = no filter)
```

Defaults (`chunk_size_tokens: 600`, `top_k: 5`) target ~3 K tokens of injected file context per turn — comfortable for 32 K models when one file is referenced, comfortable for 128 K models when several are.

When `chunking.backend: pgvector` and the agent already has `memory.backend: pgvector` pointing at the same database, the chunker reuses the connection pool. When they're different databases (or memory is disabled), a separate pool is acquired. The `embedding_url` / `embedding_model` may be reused across both; this is recommended but not enforced.

### Lifecycle

- **On upload** (`POST /v1/files`): after parsing completes successfully, the server compares `token_count(extracted_text)` against `small_file_threshold_tokens`. If above, it kicks off chunking + embedding via `asyncio.create_task` (does not block the upload response). The endpoint response includes `chunk_status: pending` initially. A new field on `FileRecord` (`chunk_count: int`, default 0) tracks completion.
- **On chat-completion** (`POST /v1/chat/completions` with `file_ids`): the retrieval branch fires only when `record.chunk_count > 0`. If chunking is still in flight (status `pending`), the server falls back to full-text injection — preserves correctness during the warm-up window at the cost of one slow turn.
- **On delete** (`DELETE /v1/files/{file_id}`): `FileStore.delete()` calls `ChunkStore.delete_for_file()` before deleting the metadata row.

### Backward compatibility

- `chunking.enabled: false` is the default. Existing 0.17.0 deployments need no config changes. The chunker code path doesn't load.
- `FileRecord.chunk_count` is a new field with a default of 0. Old SQLite/Postgres rows read as `chunk_count=0` and naturally take the full-text branch.
- The `[chunking]` extra is opt-in: `pip install fipsagents[chunking]` pulls in `tiktoken` and (transitively) the pgvector deps already covered by `[pgvector]`. Production deployments that already have `[pgvector]` installed only need to add `tiktoken`.

## Alternatives Considered

### Alternative 1: Reuse `agent_memories` table

Write chunks into the existing `agent_memories` table with `metadata.kind = "file_chunk"`, `metadata.file_id = ...`. Search the same table at retrieval time, filtered by `metadata`.

- **Pro**: Zero new tables. Reuses `PGVectorMemoryClient` end-to-end.
- **Con**: Conflates two distinct lifecycle domains. Memory is long-lived, user/project-scoped, weighted, weight-decayed, and surfaced into prompts via `build_memory_prefix`. Chunks are tied to a `file_id`'s lifetime, never decay, and inject only on explicit `file_ids` reference. Mixing them means the memory prefix accidentally surfaces file chunks; the chunk retrieval accidentally surfaces user preferences.
- **Con**: `agent_memories` lacks `file_id` and `chunk_index` first-class columns, so per-file scoping becomes a `metadata->>'file_id' = $1` JSONB query — slower than an indexed BIGINT-style column and ugly to read.
- **Con**: `delete()` on a file would have to scan `agent_memories` by JSONB key. Cascade-delete becomes O(table) per file deletion.

Rejected. The "one table for everything" appeal is real, but the lifecycle mismatch makes it the wrong call.

### Alternative 2: Server-side context-window-aware truncation only (no chunking)

When `extracted_text` exceeds the model's context window, truncate to the head N tokens with a "[truncated]" marker. No retrieval, no embeddings.

- **Pro**: Trivial to implement (~20 LOC).
- **Pro**: No new infrastructure.
- **Con**: Misses the actual problem. A user asking about page 150 of a 200-page contract gets the first 30 pages and a sad note. Useless.
- **Con**: Doesn't address the cost-on-every-turn issue at all — even when the document fits, dumping it on every follow-up is wasteful.

Rejected as a complete solution. Will be implemented anyway as a *fallback* for files where chunking is disabled or failed (better than a 422 from the model).

### Alternative 3: External vector DB (Weaviate / Qdrant / OpenSearch)

Don't reuse pgvector — stand up a dedicated vector database for chunk retrieval. Likely Weaviate or Qdrant on-cluster, or an enterprise OpenSearch deployment.

- **Pro**: Better recall on huge corpora (millions of chunks). Hybrid (BM25 + vector) search built in.
- **Pro**: Decouples file chunking from the agent's metadata DB.
- **Con**: New cluster service to deploy, monitor, back up, and secure. The agent-template stack has been deliberately small; adding a vector DB doubles operational surface.
- **Con**: Network hop on every chat completion.
- **Con**: Splits the "where does my file's content live" answer across two systems for operators to reason about.
- **Con**: pgvector handles tens of millions of chunks fine — recall is not the constraint at the scale of "files referenced by chat completions in one agent's lifetime".

Pluggable design (the `ChunkStore` ABC) keeps this as a future option without paying for it now. Document `pgvector` as the only supported backend; leave `OpenSearchChunkStore` etc. as future work behind a marker like `not yet implemented` raise.

### Alternative 4: Agent-side tool (LLM decides when to retrieve)

Instead of server-side retrieval, expose a `read_file_chunks(file_id, query)` tool to the LLM and let the model call it when it needs file content. No automatic injection.

- **Pro**: Natural fit with the existing tool-calling pattern. The LLM can decide when it has enough context.
- **Pro**: Multi-step retrieval is easy — the LLM can ask multiple queries.
- **Con**: Pushes complexity onto the model. Granite 3.3 8B notoriously can't follow tool-calling protocol (per CLAUDE.md). Many of our deployment targets won't be able to use this path reliably.
- **Con**: Multi-turn + tool-call latency adds 2–5 seconds per file reference. The server-side retrieval is one extra DB query — sub-50ms.
- **Con**: Doesn't match the OpenAI / Anthropic file-API ergonomics that the upload endpoint deliberately mimics. Their semantics are "you reference a file, the system provides the content"; tool-call retrieval breaks that contract.

Rejected as the primary path. May be added later as a complement (the `ChunkStore` is already the natural backing for such a tool).

### Alternative 5: Inline chunking in `Parser` outputs

Have `DoclingParser.parse()` return chunks directly, skipping the `extracted_text` step entirely.

- **Pro**: One less transformation.
- **Con**: Loses the full-text path. Many use cases (search, audit, reranking, debugging) want the original extracted text. Storing only chunks means re-assembly to recover it, and the assembly is lossy after the chunker added overlap and trimmed structural separators.
- **Con**: Couples the parser interface to a chunking strategy. Today's `PlaintextParser` doesn't know anything about token boundaries; making it produce chunks forces it to.

Rejected. Keep extraction and chunking as separate, composable steps.

## Consequences

### Positive

- **Token-bound deployments unblock.** Granite 8B users can upload arbitrarily large files; the model only sees the relevant ~3 K tokens per turn.
- **Cost on subsequent turns drops 10–50×** for cost-bound deployments, since the file context shrinks from full-text to top-K chunks per turn.
- **Reuses the deployed embedding model.** The cluster already runs an embedding endpoint for `PGVectorMemoryClient`. No new model to deploy.
- **Reuses the pgvector deployment.** When the agent already has `memory.backend: pgvector`, chunking is "another table in the same database" — operationally one-line additive.
- **Per-file scoping for free.** The `file_id` column makes per-tenant authorization trivial. The user's referenced `file_ids` ARE the authorization boundary.
- **Backward compatible.** `chunking.enabled: false` is the default; 0.17.0 deployments upgrade with no config changes and no behavior change.
- **Pluggable.** `ChunkStore` ABC mirrors the `BytesStore` and `FileStore` patterns. Adding OpenSearch/Qdrant/Weaviate later is a new implementation, not a redesign.

### Negative

- **One more service dependency** (the embedding endpoint) on the chat-completion hot path. Already exists when memory is configured for pgvector, but for users with markdown/sqlite memory backends, this is genuinely new. Mitigated: when chunking is `enabled: false` or chunks haven't been computed yet, the path doesn't fire.
- **Async chunking introduces a "warm-up" window.** Between upload and chunk completion, chat completions referencing the file fall back to full-text injection. For very large files (5+ minutes of embedding work), this can mean one or two slow turns before the chunked path engages. Document loudly; consider a polling endpoint (`GET /v1/files/{id}/chunk_status`) if it becomes a friction point.
- **Embedding latency on upload.** A 200-page document at 600 tokens/chunk = ~150 chunks × 30ms/embedding = ~5 seconds of embedding work. Runs in a background task, so doesn't block the upload response, but adds load to the embedding model. Mitigated by batching: pgvector + most embedding endpoints accept `input: [...]` arrays; one HTTP round-trip per ~32 chunks.
- **Storage cost for chunks.** A 200-page document → ~150 rows in `file_chunks`, ~600 chars + 768-dim float vector each → ~5 KB/row → ~750 KB per document. At 10 K documents = ~7.5 GB. Within pgvector's comfort zone, but operators should plan PVC sizing accordingly.
- **Two query latency added per chat completion** (one embedding lookup + one pgvector search per referenced file). ~50–150ms total. Significant for streaming TTFT but invisible in non-streaming. Could be parallelized across `file_ids` with `asyncio.gather`; the implementation should do this.
- **Embedding model version coupling.** If the operator changes the embedding model, existing chunks become unsearchable (different vector space). The migration path is "delete and re-upload affected files" — the same as `PGVectorMemoryClient`'s posture. Document this as an operator concern; do not attempt online re-embedding.

## Implementation Sketch

`packages/fipsagents/src/fipsagents/server/chunker.py` (new):

```python
class Chunker(ABC):
    @abstractmethod
    async def chunk(
        self,
        text: str,
        *,
        chunk_size_tokens: int = 600,
        chunk_overlap_tokens: int = 100,
    ) -> list[Chunk]: ...


class Chunk:
    content: str
    metadata: dict[str, Any]  # parser-supplied (page_number, section_path, etc.)


class RecursiveTokenChunker(Chunker): ...
class DoclingChunker(Chunker): ...
```

`packages/fipsagents/src/fipsagents/server/chunk_store.py` (new):

```python
class ChunkStore(ABC):
    @abstractmethod
    async def save_chunks(
        self,
        file_id: str,
        chunks: list[Chunk],
        *,
        user_id: str,
        session_id: str | None,
    ) -> int: ...

    @abstractmethod
    async def search(
        self,
        file_id: str,
        query: str,
        *,
        limit: int = 5,
        min_score: float = 0.0,
    ) -> list[Chunk]: ...

    @abstractmethod
    async def delete_for_file(self, file_id: str) -> int: ...

    async def close(self) -> None: ...


class NullChunkStore(ChunkStore): ...

class PgvectorChunkStore(ChunkStore):
    def __init__(
        self,
        pool: asyncpg.Pool,
        http: httpx.AsyncClient,
        embedding_url: str,
        embedding_model: str,
        embedding_dimension: int,
        table_name: str = "file_chunks",
    ) -> None: ...
```

`FilesConfig` (extended):

```python
class ChunkingConfig(BaseModel):
    enabled: bool = False
    backend: Literal["null", "pgvector"] = "null"
    database_url: str = ""
    embedding_url: str = ""
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 768
    table_name: str = "file_chunks"
    chunk_size_tokens: int = 600
    chunk_overlap_tokens: int = 100
    small_file_threshold_tokens: int = 4000
    retrieval_top_k: int = 5
    retrieval_min_score: float = 0.0
```

`FileRecord` (extended):

```python
@dataclass
class FileRecord:
    # ... existing fields ...
    chunk_status: Literal["pending", "processing", "completed", "skipped", "failed"] = "pending"
    chunk_count: int = 0
```

`OpenAIChatServer` (extended):

```python
async def _resolve_file_attachments(
    self,
    file_ids: list[str],
    last_user_message: str,
) -> list[dict[str, Any]]:
    messages = []
    for fid in file_ids:
        record = await self._file_store.get_metadata(fid)
        # ... existing not-found branch ...

        if (
            self._chunk_store is not None
            and record.chunk_count > 0
            and self._token_count(record.extracted_text) > self._chunking.small_file_threshold_tokens
        ):
            chunks = await self._chunk_store.search(
                fid,
                last_user_message,
                limit=self._chunking.retrieval_top_k,
                min_score=self._chunking.retrieval_min_score,
            )
            content = "\n---\n".join(c.content for c in chunks)
            messages.append({"role": "system", "content": f"{header}\n{content}"})
        elif record.extracted_text:
            # existing full-text branch
            ...
        else:
            # existing not-available branch
            ...
    return messages
```

Approximate diff size: +800 LOC (new `chunker.py`, new `chunk_store.py`, server wiring, config, schema migration). +200 LOC of tests.

## Migration

For deployments running 0.17.0:

1. Provision pgvector (or reuse the one already used by memory).
2. Add `files.chunking` block to `agent.yaml`. Set `enabled: true`, `backend: pgvector`, point at the embedding endpoint and database.
3. Restart. Existing files keep working under the full-text path (`chunk_count = 0`). New uploads run through chunking.
4. (Optional) Backfill: a new admin endpoint `POST /v1/files/{id}/rechunk` re-runs chunking for an existing file. One-off script can iterate `GET /v1/files` and call this endpoint.

For green-field deployments: enable `chunking` from day one; no PVC needed for chunks (pgvector handles it).

## Out of Scope

- **Reranker on retrieval results.** The cluster has a deployed reranker (`cluster_endpoints`); plumbing it into chunk retrieval is a follow-up. The `min_score` knob is a poor-man's version for now.
- **Hybrid (BM25 + vector) search.** pgvector + the `pg_trgm` extension can do this, but it's a follow-up. The Postgres `to_tsvector` text-search fallback already exists in `PGVectorMemoryClient` and could be ported here if needed.
- **Query rewriting / HyDE.** The user message goes to the embedding model verbatim. LLM-driven query rewriting (turn "what does it say about X?" into a richer query) is a future quality knob.
- **Cross-file retrieval.** Search is scoped per-file by design (auth boundary). Searching across all files a user has uploaded is a different feature, not this ADR.
- **Re-embedding on model change.** Operator concern. Document as "delete and re-upload" for now; an online migration tool can be a follow-up.
- **Chunk-level signed URLs / page references.** The `metadata` JSONB column carries `page_number` / `section_path` from Docling, but exposing them in the API response (e.g. citations like `[doc.pdf, p. 42]`) is a follow-up.

## Decisions

Three open questions were resolved before merge:

1. **`tiktoken` is a soft dep with a char/4 fallback.** `RecursiveTokenChunker` will use `tiktoken` when importable and fall back to a `len(text) // 4` heuristic otherwise. The `[chunking]` extra does not pin `tiktoken` as a hard dep; deployments that want tighter token accounting opt into it via `pip install fipsagents[chunking,tiktoken]` (or just install `tiktoken` directly). Rationale: `tiktoken` pulls Rust extensions that complicate FIPS builds and adds ~5 MB; the heuristic is within ±20% for English-like text and is good enough for chunk-boundary decisions.

2. **`MemoryConfig.budget` drives chunking defaults too.** When `chunking` is enabled and `MemoryConfig.budget` is set, the chunking knobs inherit a preset:
   - `small` → `chunk_size_tokens: 400`, `retrieval_top_k: 3`, `small_file_threshold_tokens: 2000`
   - `medium` (default) → `chunk_size_tokens: 600`, `retrieval_top_k: 5`, `small_file_threshold_tokens: 4000`
   - `large` → `chunk_size_tokens: 800`, `retrieval_top_k: 8`, `small_file_threshold_tokens: 8000`

   Explicit `chunking.*` values always override the preset. Implementation mirrors `_apply_budget_presets` on `MemoryConfig` — a `model_validator(mode="before")` using `data.setdefault()`.

3. **App-level cascade is the contract; no DB-level FK.** `FileStore.delete()` calls `ChunkStore.delete_for_file()` before deleting the metadata row. No foreign key is declared at the database level because metadata may live in SQLite while chunks live in Postgres — a cross-database FK is impossible. The `FileStore.delete()` docstring will document the cascade contract explicitly. Operators running both metadata and chunks in the same Postgres instance may add a manual FK if they want, but the framework does not depend on it.

## Follow-ups

- File a tracking issue: "feat(server): chunking + pgvector retrieval for /v1/files (closes one bullet of #100)". *(Filed as #137.)*
- Integration test: end-to-end against a containerized Postgres-with-pgvector and the cluster's embedding endpoint. Marker `chunking_live`. Skips when `EMBEDDING_URL` is unset.
- Update Module 9 of the examples site once shipped: replace "files inject as full text" with "files >4 K tokens are chunked and retrieved per turn".
