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
        assert body["parse_status"] == "pending"
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
        assert body["parse_status"] == "pending"
        assert body["extracted_text"] is None

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

    _RecordingMixin.captured_messages = []
    return OpenAIChatServer(_A)


class TestFileIdsInjection:
    def test_inject_unparsed_file_emits_stub(self, tmp_path):
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
        assert "parse_status: pending" in captured[0]["content"]
        assert captured[1]["role"] == "user"
        assert captured[1]["content"] == "Summarise the file."

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
