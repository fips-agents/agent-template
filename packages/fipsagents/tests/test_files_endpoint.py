"""Tests for the /v1/files endpoint and chat-completion file_ids injection."""


from __future__ import annotations

from typing import Any

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from fipsagents.baseagent.events import (  # noqa: E402
    ContentDelta,
    StreamComplete,
    StreamMetrics,
)
from fipsagents.server import OpenAIChatServer  # noqa: E402

from tests.test_server_openai import _make_agent_class  # type: ignore  # noqa: E402


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------


def _build_server_with_files(
    tmp_path,
    *,
    enabled: bool = True,
    max_size: int = 1024 * 1024,
    allowed: list[str] | None = None,
    events: list[Any] | None = None,
):
    """Wire up a server with the SQLite-backed FileStore enabled."""
    AgentClass = _make_agent_class(events or [], model_name="m1")

    bytes_dir = str(tmp_path / "files")
    sqlite_path = str(tmp_path / "agent.db")

    class _A(AgentClass):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.config.server.storage.backend = "sqlite"
            self.config.server.storage.sqlite_path = sqlite_path
            self.config.server.files.enabled = enabled
            self.config.server.files.bytes_dir = bytes_dir
            self.config.server.files.max_file_size_bytes = max_size
            self.config.server.files.allowed_mime_types = (
                list(allowed) if allowed is not None else []
            )
            self.config.server.files.backend = "sqlite"
            self.config.server.files.sqlite_path = ""

    return OpenAIChatServer(_A)


# ---------------------------------------------------------------------------
# POST /v1/files
# ---------------------------------------------------------------------------


class TestUploadEndpoint:
    def test_upload_returns_metadata(self, tmp_path):
        server = _build_server_with_files(tmp_path)
        with TestClient(server.app) as client:
            resp = client.post(
                "/v1/files",
                files={"file": ("hello.txt", b"hello world", "text/plain")},
            )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["filename"] == "hello.txt"
        assert body["mime_type"] == "text/plain"
        assert body["size_bytes"] == 11
        assert body["sha256"] == (
            # SHA-256 of b"hello world"
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        )
        # Plaintext is parsed inline at upload time.
        assert body["parse_status"] == "completed"
        assert body["parse_error"] is None
        assert body["file_id"].startswith("file_")
        assert body["session_id"] is None

    def test_upload_with_session_id_form_field(self, tmp_path):
        server = _build_server_with_files(tmp_path)
        with TestClient(server.app) as client:
            resp = client.post(
                "/v1/files",
                files={"file": ("a.txt", b"hello world", "text/plain")},
                data={"session_id": "sess-abc"},
            )
        assert resp.status_code == 201
        assert resp.json()["session_id"] == "sess-abc"

    def test_upload_invalid_session_id_rejected(self, tmp_path):
        server = _build_server_with_files(tmp_path)
        with TestClient(server.app) as client:
            resp = client.post(
                "/v1/files",
                files={"file": ("a.txt", b"hello world", "text/plain")},
                data={"session_id": "bad/session id!"},
            )
        assert resp.status_code == 400
        assert "session_id" in resp.json()["detail"]

    def test_upload_oversize_returns_413(self, tmp_path):
        # Limit set to 16 bytes; payload is 32.
        server = _build_server_with_files(tmp_path, max_size=16)
        with TestClient(server.app) as client:
            resp = client.post(
                "/v1/files",
                files={"file": ("big.txt", b"x" * 32, "text/plain")},
            )
        assert resp.status_code == 413
        assert "max_file_size_bytes" in resp.json()["detail"]

    def test_upload_unallowed_mime_returns_415(self, tmp_path):
        server = _build_server_with_files(
            tmp_path, allowed=["application/pdf"],
        )
        with TestClient(server.app) as client:
            resp = client.post(
                "/v1/files",
                files={"file": ("a.txt", b"hello world", "text/plain")},
            )
        assert resp.status_code == 415
        assert "text/plain" in resp.json()["detail"]

    def test_upload_allowlist_with_match_succeeds(self, tmp_path):
        server = _build_server_with_files(
            tmp_path, allowed=["text/plain", "application/pdf"],
        )
        with TestClient(server.app) as client:
            resp = client.post(
                "/v1/files",
                files={"file": ("a.txt", b"hello world", "text/plain")},
            )
        assert resp.status_code == 201

    def test_upload_disabled_returns_404(self, tmp_path):
        server = _build_server_with_files(tmp_path, enabled=False)
        with TestClient(server.app) as client:
            resp = client.post(
                "/v1/files",
                files={"file": ("a.txt", b"hello", "text/plain")},
            )
        assert resp.status_code == 404
        assert "not enabled" in resp.json()["detail"]

    def test_upload_records_user_id_from_header(self, tmp_path):
        server = _build_server_with_files(tmp_path)
        with TestClient(server.app) as client:
            resp = client.post(
                "/v1/files",
                files={"file": ("a.txt", b"hello world", "text/plain")},
                headers={"X-Auth-Subject": "alice@example.com"},
            )
            assert resp.status_code == 201
            file_id = resp.json()["file_id"]

            meta_resp = client.get(f"/v1/files/{file_id}")
        assert meta_resp.status_code == 200
        assert meta_resp.json()["user_id"] == "alice@example.com"


# ---------------------------------------------------------------------------
# GET /v1/files/{file_id}
# ---------------------------------------------------------------------------


class TestGetFile:
    def test_returns_metadata(self, tmp_path):
        server = _build_server_with_files(tmp_path)
        with TestClient(server.app) as client:
            up = client.post(
                "/v1/files",
                files={"file": ("doc.txt", b"hello world", "text/plain")},
            )
            assert up.status_code == 201
            file_id = up.json()["file_id"]

            resp = client.get(f"/v1/files/{file_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["file_id"] == file_id
        assert body["filename"] == "doc.txt"
        # Plaintext is parsed inline; extracted_text is populated.
        assert body["parse_status"] == "completed"
        assert body["extracted_text"] == "hello world"

    def test_unknown_file_returns_404(self, tmp_path):
        server = _build_server_with_files(tmp_path)
        with TestClient(server.app) as client:
            resp = client.get("/v1/files/file_does_not_exist")
        assert resp.status_code == 404

    def test_disabled_returns_404(self, tmp_path):
        server = _build_server_with_files(tmp_path, enabled=False)
        with TestClient(server.app) as client:
            resp = client.get("/v1/files/anything")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# files.sqlite_path overrides storage.sqlite_path for the file store
# (issue #131 — SqliteFileStore metadata DB co-located with bytes_dir on PVC)
# ---------------------------------------------------------------------------


class TestFilesSqlitePathOverride:
    def test_files_sqlite_path_overrides_storage_path(self, tmp_path):
        """When set, FilesConfig.sqlite_path wins over StorageConfig.sqlite_path."""
        AgentClass = _make_agent_class([], model_name="m1")
        bytes_dir = str(tmp_path / "files")
        storage_db = str(tmp_path / "storage.db")
        files_db = str(tmp_path / "metadata" / "files.db")

        class _A(AgentClass):  # type: ignore[misc, valid-type]
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.config.server.storage.backend = "sqlite"
                self.config.server.storage.sqlite_path = storage_db
                self.config.server.files.enabled = True
                self.config.server.files.backend = "sqlite"
                self.config.server.files.bytes_dir = bytes_dir
                self.config.server.files.sqlite_path = files_db

        server = OpenAIChatServer(_A)
        with TestClient(server.app) as client:
            up = client.post(
                "/v1/files",
                files={"file": ("a.txt", b"persist me", "text/plain")},
            )
            assert up.status_code == 201, up.text
            file_id = up.json()["file_id"]
            got = client.get(f"/v1/files/{file_id}")

        from pathlib import Path
        # Metadata DB lands at files.sqlite_path (parent dir auto-created
        # by SqliteConnectionManager).
        assert Path(files_db).exists(), (
            f"FilesConfig.sqlite_path was ignored; expected DB at {files_db}"
        )
        # End-to-end: the metadata round-trip uses the override path —
        # if the file store had been wired to storage.sqlite_path, the
        # GET would 404 because the upload landed in files_db.
        assert got.status_code == 200
        assert got.json()["file_id"] == file_id

    def test_empty_files_sqlite_path_falls_back_to_storage(self, tmp_path):
        """Empty FilesConfig.sqlite_path defers to StorageConfig.sqlite_path."""
        # Default behavior — covered by the existing _build_server_with_files
        # fixture which sets storage.sqlite_path and leaves files.sqlite_path
        # empty. Just confirm uploads land at the storage path.
        from pathlib import Path
        server = _build_server_with_files(tmp_path)
        storage_db = tmp_path / "agent.db"
        with TestClient(server.app) as client:
            resp = client.post(
                "/v1/files",
                files={"file": ("a.txt", b"x", "text/plain")},
            )
        assert resp.status_code == 201
        assert storage_db.exists()


# ---------------------------------------------------------------------------
# file_ids injection into /v1/chat/completions
# ---------------------------------------------------------------------------


class _RecordingMixin:
    """Capture the messages list the agent sees on each request."""

    captured_messages: list[list[dict]] = []  # class-level: shared across instances

    async def astep_stream(self, *, max_iterations: int = 10):
        # The OpenAIChatServer.collect path overwrites self.messages with
        # the inbound conversation just before invoking the stream — so
        # snapshot here.
        type(self).captured_messages.append(list(self.messages))
        for ev in self._events:
            yield ev


def _stream_events_with_text(text: str) -> list[Any]:
    return [
        ContentDelta(content=text),
        StreamComplete(
            finish_reason="stop",
            metrics=StreamMetrics(prompt_tokens=1, completion_tokens=1),
        ),
    ]


def _build_recording_server(tmp_path):
    AgentClass = _make_agent_class(
        _stream_events_with_text("ok"), model_name="m1",
    )
    bytes_dir = str(tmp_path / "files")
    sqlite_path = str(tmp_path / "agent.db")

    class _A(_RecordingMixin, AgentClass):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.config.server.storage.backend = "sqlite"
            self.config.server.storage.sqlite_path = sqlite_path
            self.config.server.files.enabled = True
            self.config.server.files.bytes_dir = bytes_dir
            self.config.server.files.backend = "sqlite"
            self.config.server.files.sqlite_path = ""

    _RecordingMixin.captured_messages = []
    return OpenAIChatServer(_A)


class TestFileIdsInjection:
    def test_inject_unparsed_file_emits_stub(self, tmp_path):
        # The injection path emits a stub for every non-completed parse_status.
        # The exact terminal status depends on the local environment:
        #   * Docling missing → PlaintextParser fall-through can't decode the
        #     PDF magic bytes, parse_status="skipped".
        #   * Docling present (eg dev machines with the [files] extra
        #     installed) → DoclingParser tries the truncated payload and
        #     errors out, parse_status="failed".
        # Either way the injection contract is the same: a stub that cites
        # filename + MIME and surfaces parse_status so the LLM knows the
        # bytes weren't parsed.
        server = _build_recording_server(tmp_path)
        with TestClient(server.app) as client:
            up = client.post(
                "/v1/files",
                files={"file": ("notes.pdf", b"%PDF-1.4 ...", "application/pdf")},
            )
            file_id = up.json()["file_id"]

            resp = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Summarise the file."}],
                    "file_ids": [file_id],
                    "stream": False,
                },
            )
        assert resp.status_code == 200, resp.text

        captured = _RecordingMixin.captured_messages[-1]
        # Expected order: file-context system msg, then user.
        assert len(captured) == 2
        assert captured[0]["role"] == "system"
        assert "notes.pdf" in captured[0]["content"]
        assert "application/pdf" in captured[0]["content"]
        assert (
            "parse_status: skipped" in captured[0]["content"]
            or "parse_status: failed" in captured[0]["content"]
        ), captured[0]["content"]
        assert captured[1]["role"] == "user"
        assert captured[1]["content"] == "Summarise the file."

    def test_inject_plaintext_uses_inline_parsed_text(self, tmp_path):
        """End-to-end: upload txt → injection sees decoded UTF-8 text."""
        server = _build_recording_server(tmp_path)
        with TestClient(server.app) as client:
            up = client.post(
                "/v1/files",
                files={"file": ("memo.txt", b"call mom\nbuy milk", "text/plain")},
            )
            assert up.status_code == 201
            assert up.json()["parse_status"] == "completed"
            file_id = up.json()["file_id"]

            resp = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "what's on the list?"}],
                    "file_ids": [file_id],
                    "stream": False,
                },
            )
        assert resp.status_code == 200
        captured = _RecordingMixin.captured_messages[-1]
        assert "call mom" in captured[0]["content"]
        assert "buy milk" in captured[0]["content"]

    def test_inject_parsed_file_emits_full_text(self, tmp_path):
        server = _build_recording_server(tmp_path)
        with TestClient(server.app) as client:
            up = client.post(
                "/v1/files",
                files={"file": ("notes.txt", b"hello world", "text/plain")},
            )
            file_id = up.json()["file_id"]

            # Manually populate extracted_text — PR 3 will do this via Docling.
            store = server._file_store
            import asyncio
            asyncio.get_event_loop().run_until_complete(
                store.update_extracted_text(
                    file_id,
                    extracted_text="Parsed body text.",
                    parse_status="completed",
                )
            )

            resp = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Read it."}],
                    "file_ids": [file_id],
                    "stream": False,
                },
            )
        assert resp.status_code == 200
        captured = _RecordingMixin.captured_messages[-1]
        assert len(captured) == 2
        assert "Parsed body text." in captured[0]["content"]
        assert "parse_status" not in captured[0]["content"]

    def test_inject_unknown_file_returns_400(self, tmp_path):
        server = _build_recording_server(tmp_path)
        with TestClient(server.app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "x"}],
                    "file_ids": ["file_does_not_exist"],
                    "stream": False,
                },
            )
        assert resp.status_code == 400
        assert "file_does_not_exist" in resp.json()["detail"]

    def test_inject_when_disabled_returns_400(self, tmp_path):
        # File store enabled at upload time; then we flip it off and
        # try to inject. Easier path: build a server with files disabled
        # and try to use file_ids — should 400.
        AgentClass = _make_agent_class(
            _stream_events_with_text("ok"), model_name="m1",
        )

        class _A(AgentClass):  # type: ignore[misc, valid-type]
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.config.server.files.enabled = False

        server = OpenAIChatServer(_A)
        with TestClient(server.app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "x"}],
                    "file_ids": ["file_anything"],
                    "stream": False,
                },
            )
        assert resp.status_code == 400
        assert "not enabled" in resp.json()["detail"]

    def test_no_file_ids_skips_injection(self, tmp_path):
        server = _build_recording_server(tmp_path)
        with TestClient(server.app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                },
            )
        assert resp.status_code == 200
        captured = _RecordingMixin.captured_messages[-1]
        # Only the user message — no system context injected.
        assert len(captured) == 1
        assert captured[0]["role"] == "user"


# ---------------------------------------------------------------------------
# DELETE /v1/files/{file_id}
# ---------------------------------------------------------------------------


class TestDeleteFile:
    def test_delete_existing_file(self, tmp_path):
        server = _build_server_with_files(tmp_path)
        with TestClient(server.app) as client:
            up = client.post(
                "/v1/files",
                files={"file": ("a.txt", b"hello world", "text/plain")},
            )
            file_id = up.json()["file_id"]

            resp = client.delete(f"/v1/files/{file_id}")
            assert resp.status_code == 200
            body = resp.json()
            assert body["deleted"] is True
            assert body["file_id"] == file_id

            # Subsequent GET should 404.
            assert client.get(f"/v1/files/{file_id}").status_code == 404

    def test_delete_unknown_returns_404(self, tmp_path):
        server = _build_server_with_files(tmp_path)
        with TestClient(server.app) as client:
            resp = client.delete("/v1/files/file_does_not_exist")
        assert resp.status_code == 404

    def test_delete_disabled_returns_404(self, tmp_path):
        server = _build_server_with_files(tmp_path, enabled=False)
        with TestClient(server.app) as client:
            resp = client.delete("/v1/files/anything")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /v1/files (list for session)
# ---------------------------------------------------------------------------


class TestListFiles:
    def test_list_filters_by_session(self, tmp_path):
        server = _build_server_with_files(tmp_path)
        with TestClient(server.app) as client:
            for i in range(3):
                client.post(
                    "/v1/files",
                    files={"file": (f"a{i}.txt", f"body {i}".encode(), "text/plain")},
                    data={"session_id": "sess-a"},
                )
            client.post(
                "/v1/files",
                files={"file": ("b.txt", b"other", "text/plain")},
                data={"session_id": "sess-b"},
            )

            resp = client.get("/v1/files", params={"session_id": "sess-a"})
        assert resp.status_code == 200
        records = resp.json()
        assert len(records) == 3
        assert all(r["session_id"] == "sess-a" for r in records)

    def test_list_unknown_session_empty(self, tmp_path):
        server = _build_server_with_files(tmp_path)
        with TestClient(server.app) as client:
            resp = client.get("/v1/files", params={"session_id": "nope"})
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_respects_limit_and_offset(self, tmp_path):
        server = _build_server_with_files(tmp_path)
        with TestClient(server.app) as client:
            for i in range(5):
                client.post(
                    "/v1/files",
                    files={"file": (f"f{i}.txt", b"hello world", "text/plain")},
                    data={"session_id": "sess-p"},
                )

            page1 = client.get(
                "/v1/files",
                params={"session_id": "sess-p", "limit": 2, "offset": 0},
            ).json()
            page2 = client.get(
                "/v1/files",
                params={"session_id": "sess-p", "limit": 2, "offset": 2},
            ).json()
        assert len(page1) == 2
        assert len(page2) == 2
        assert {r["file_id"] for r in page1}.isdisjoint(
            {r["file_id"] for r in page2},
        )

    def test_list_requires_session_id(self, tmp_path):
        server = _build_server_with_files(tmp_path)
        with TestClient(server.app) as client:
            resp = client.get("/v1/files")
        # FastAPI raises 422 for missing required query params.
        assert resp.status_code == 422

    def test_list_invalid_session_id_returns_400(self, tmp_path):
        server = _build_server_with_files(tmp_path)
        with TestClient(server.app) as client:
            resp = client.get(
                "/v1/files", params={"session_id": "bad/session id!"},
            )
        assert resp.status_code == 400
        assert "session_id" in resp.json()["detail"]

    def test_list_disabled_returns_404(self, tmp_path):
        server = _build_server_with_files(tmp_path, enabled=False)
        with TestClient(server.app) as client:
            resp = client.get("/v1/files", params={"session_id": "anything"})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# MIME sniffing via python-magic
# ---------------------------------------------------------------------------


# Real-ish payloads with the right magic bytes. python-magic returns
# different verdicts for tiny stubs vs full-structure files; these
# stubs are large enough for libmagic to identify them.
_PDF_BYTES = (
    b"%PDF-1.4\n%\xc7\xec\x8f\xa2\n"
    b"1 0 obj\n<< /Type /Catalog >>\nendobj\n"
    b"trailer\n<< /Root 1 0 R >>\n"
    b"%%EOF\n"
)


class TestMimeSniffing:
    def test_sniffer_overrides_misleading_content_type(self, tmp_path):
        """A client claims text/plain but the bytes are a PDF — record
        and allowlist must use the sniffed MIME."""
        server = _build_server_with_files(tmp_path)
        with TestClient(server.app) as client:
            resp = client.post(
                "/v1/files",
                # Lying about content-type: claims text/plain, sends PDF.
                files={"file": ("a.pdf", _PDF_BYTES, "text/plain")},
            )
        assert resp.status_code == 201
        # libmagic identified it correctly despite the client's claim.
        assert resp.json()["mime_type"] == "application/pdf"

    def test_allowlist_applies_to_sniffed_value(self, tmp_path):
        """Allowlist rejects bytes whose sniffed type isn't allowed,
        even when the client claims an allowed type."""
        server = _build_server_with_files(
            tmp_path, allowed=["text/plain"],
        )
        with TestClient(server.app) as client:
            resp = client.post(
                "/v1/files",
                # Claims text/plain (allowed) but the bytes are a PDF.
                files={"file": ("a.pdf", _PDF_BYTES, "text/plain")},
            )
        assert resp.status_code == 415
        assert "application/pdf" in resp.json()["detail"]

    def test_allowlist_passes_when_sniffed_matches(self, tmp_path):
        """Allowlist accepts when the sniffed MIME is on the list,
        regardless of what the client claimed."""
        server = _build_server_with_files(
            tmp_path, allowed=["application/pdf"],
        )
        with TestClient(server.app) as client:
            resp = client.post(
                "/v1/files",
                # Client claims a junk content-type; bytes are real PDF.
                files={"file": ("a.pdf", _PDF_BYTES, "application/garbage")},
            )
        assert resp.status_code == 201
        assert resp.json()["mime_type"] == "application/pdf"

    def test_falls_back_to_claim_when_libmagic_unavailable(
        self, tmp_path, monkeypatch,
    ):
        """When libmagic is missing, sniffer returns None and the
        endpoint falls back to the client-supplied Content-Type."""
        from fipsagents.server import files as files_mod

        def _no_magic():
            raise ImportError("libmagic missing in this test")

        monkeypatch.setattr(files_mod, "_get_magic_module", _no_magic)
        # Reset the once-per-process warning gate so the fallback fires.
        monkeypatch.setattr(files_mod, "_magic_unavailable_logged", False)

        server = _build_server_with_files(tmp_path)
        with TestClient(server.app) as client:
            resp = client.post(
                "/v1/files",
                files={"file": ("a.txt", b"hello world", "text/plain")},
            )
        assert resp.status_code == 201
        # Falls back to the client-supplied type.
        assert resp.json()["mime_type"] == "text/plain"

    def test_plaintext_sniffed_correctly(self, tmp_path):
        """Sanity check — plaintext bytes detect as text/plain even
        when the client labels them application/octet-stream."""
        server = _build_server_with_files(tmp_path)
        with TestClient(server.app) as client:
            resp = client.post(
                "/v1/files",
                files={"file": (
                    "notes.txt", b"hello world", "application/octet-stream",
                )},
            )
        assert resp.status_code == 201
        assert resp.json()["mime_type"] == "text/plain"


# ---------------------------------------------------------------------------
# Virus scanning integration with /v1/files
# ---------------------------------------------------------------------------


class _FakeScanner:
    """In-process stand-in for VirusScanner used in endpoint tests."""

    def __init__(self, *, infected=False, error=None, viruses=None):
        from fipsagents.server.scanner import ScanResult
        if error is not None:
            self._result = ScanResult.failed(error)
        elif infected:
            self._result = ScanResult.found(viruses or ["EICAR-Test-File"])
        else:
            self._result = ScanResult.clean()
        self.calls: list[tuple[bytes, str]] = []

    async def scan(self, data, *, filename):
        self.calls.append((data, filename))
        return self._result

    async def close(self):
        pass


def _build_server_with_scanner(tmp_path, scanner, *, fail_mode="open"):
    AgentClass = _make_agent_class([], model_name="m1")
    bytes_dir = str(tmp_path / "files")
    sqlite_path = str(tmp_path / "agent.db")

    class _A(AgentClass):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.config.server.storage.backend = "sqlite"
            self.config.server.storage.sqlite_path = sqlite_path
            self.config.server.files.enabled = True
            self.config.server.files.bytes_dir = bytes_dir
            self.config.server.files.backend = "sqlite"
            self.config.server.files.sqlite_path = ""
            self.config.server.files.scanner.fail_mode = fail_mode

    server = OpenAIChatServer(_A)
    # Replace the scanner that lifespan would create with our fake.
    # We have to do this _inside_ the TestClient context — see the
    # individual tests below.
    server._injected_scanner = scanner  # type: ignore[attr-defined]
    return server


class TestVirusScanningIntegration:
    def test_clean_upload_succeeds(self, tmp_path):
        scanner = _FakeScanner(infected=False)
        server = _build_server_with_scanner(tmp_path, scanner)
        with TestClient(server.app) as client:
            server._virus_scanner = scanner  # override after lifespan
            resp = client.post(
                "/v1/files",
                files={"file": ("a.txt", b"hello world", "text/plain")},
            )
        assert resp.status_code == 201
        # Scanner saw the bytes.
        assert len(scanner.calls) == 1
        assert scanner.calls[0][0] == b"hello world"

    def test_infected_upload_returns_422(self, tmp_path):
        scanner = _FakeScanner(infected=True, viruses=["EICAR-Test-File"])
        server = _build_server_with_scanner(tmp_path, scanner)
        with TestClient(server.app) as client:
            server._virus_scanner = scanner
            resp = client.post(
                "/v1/files",
                files={"file": ("evil.bin", b"x5O...", "text/plain")},
            )
        assert resp.status_code == 422
        body = resp.json()
        assert body["detail"]["error"] == "infected"
        assert body["detail"]["viruses"] == ["EICAR-Test-File"]

    def test_scanner_failure_open_mode_accepts(self, tmp_path):
        scanner = _FakeScanner(error="connection refused")
        server = _build_server_with_scanner(tmp_path, scanner, fail_mode="open")
        with TestClient(server.app) as client:
            server._virus_scanner = scanner
            resp = client.post(
                "/v1/files",
                files={"file": ("a.txt", b"hello world", "text/plain")},
            )
        # Open mode: upload accepted despite scanner failure.
        assert resp.status_code == 201

    def test_scanner_failure_closed_mode_rejects(self, tmp_path):
        scanner = _FakeScanner(error="connection refused")
        server = _build_server_with_scanner(
            tmp_path, scanner, fail_mode="closed",
        )
        with TestClient(server.app) as client:
            server._virus_scanner = scanner
            resp = client.post(
                "/v1/files",
                files={"file": ("a.txt", b"hello world", "text/plain")},
            )
        assert resp.status_code == 503
        body = resp.json()
        assert body["detail"]["error"] == "scanner_unavailable"

    def test_default_no_scanner_url_passes(self, tmp_path):
        """Without a scanner URL configured, the default NullScanner
        accepts every upload — no special test setup needed."""
        server = _build_server_with_files(tmp_path)
        with TestClient(server.app) as client:
            resp = client.post(
                "/v1/files",
                files={"file": ("a.txt", b"hello world", "text/plain")},
            )
        assert resp.status_code == 201
