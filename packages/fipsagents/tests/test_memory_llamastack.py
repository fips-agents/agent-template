"""Tests for fipsagents.baseagent.memory_llamastack — LlamaStackMemoryClient and factory.

All HTTP calls are mocked — no real LlamaStack server required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from fipsagents.baseagent.memory import NullMemoryClient
from fipsagents.baseagent.memory_llamastack import (
    LlamaStackMemoryClient,
    create_llamastack_client,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(status_code: int = 200, json_data: dict | None = None) -> MagicMock:
    """Return a mock httpx response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.raise_for_status = MagicMock()
    return resp


def _mock_httpx_client() -> AsyncMock:
    """Create a mock httpx.AsyncClient with async method support."""
    client = AsyncMock(spec=httpx.AsyncClient)
    return client


def _make_client(
    http: AsyncMock | None = None,
    vector_store_id: str = "vs_test",
) -> LlamaStackMemoryClient:
    """Construct a LlamaStackMemoryClient with a mock httpx.AsyncClient."""
    if http is None:
        http = _mock_httpx_client()
    return LlamaStackMemoryClient(client=http, vector_store_id=vector_store_id)


# ---------------------------------------------------------------------------
# TestCreateLlamaStackClient — factory
# ---------------------------------------------------------------------------


def _mock_factory_client() -> AsyncMock:
    """Return an unspec'd AsyncMock suitable for use inside @patch tests.

    When httpx.AsyncClient is already patched, using spec=httpx.AsyncClient
    would try to spec against the mock itself and raise InvalidSpecError.
    """
    return AsyncMock()


class TestCreateLlamaStackClient:
    @pytest.mark.asyncio
    @patch("fipsagents.baseagent.memory_llamastack.httpx.AsyncClient")
    async def test_creates_client_from_valid_config(self, mock_client_cls, tmp_path):
        mock_client = _mock_factory_client()
        mock_client_cls.return_value = mock_client

        list_resp = _mock_response(
            json_data={"data": [{"id": "vs_123", "name": "agent-memory"}]}
        )
        mock_client.get = AsyncMock(return_value=list_resp)

        config = tmp_path / ".memory-llamastack.yaml"
        config.write_text("endpoint: http://localhost:8321\nvector_store: agent-memory\n")

        client = await create_llamastack_client(config)
        assert isinstance(client, LlamaStackMemoryClient)

    @pytest.mark.asyncio
    @patch("fipsagents.baseagent.memory_llamastack.httpx.AsyncClient")
    async def test_creates_vector_store_when_not_found(self, mock_client_cls, tmp_path):
        mock_client = _mock_factory_client()
        mock_client_cls.return_value = mock_client

        list_resp = _mock_response(json_data={"data": []})
        create_resp = _mock_response(json_data={"id": "vs_new", "name": "agent-memory"})
        mock_client.get = AsyncMock(return_value=list_resp)
        mock_client.post = AsyncMock(return_value=create_resp)

        config = tmp_path / ".memory-llamastack.yaml"
        config.write_text("endpoint: http://localhost:8321\nvector_store: agent-memory\n")

        client = await create_llamastack_client(config)
        assert isinstance(client, LlamaStackMemoryClient)

    @pytest.mark.asyncio
    async def test_returns_null_when_config_missing(self, tmp_path):
        client = await create_llamastack_client(tmp_path / "no-such-file.yaml")
        assert isinstance(client, NullMemoryClient)

    @pytest.mark.asyncio
    async def test_returns_null_when_config_invalid(self, tmp_path):
        bad_config = tmp_path / ".memory-llamastack.yaml"
        bad_config.write_text("[[[")
        client = await create_llamastack_client(bad_config)
        assert isinstance(client, NullMemoryClient)

    @pytest.mark.asyncio
    @patch("fipsagents.baseagent.memory_llamastack.httpx.AsyncClient")
    async def test_returns_null_on_connection_error(self, mock_client_cls, tmp_path):
        mock_client = _mock_factory_client()
        mock_client_cls.return_value = mock_client
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))

        config = tmp_path / ".memory-llamastack.yaml"
        config.write_text("endpoint: http://localhost:8321\nvector_store: agent-memory\n")

        client = await create_llamastack_client(config)
        assert isinstance(client, NullMemoryClient)


# ---------------------------------------------------------------------------
# TestLlamaStackMemoryClient — client operations
# ---------------------------------------------------------------------------


class TestLlamaStackMemoryClient:
    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        mock_http = _mock_httpx_client()
        search_resp = _mock_response(
            json_data={
                "data": [
                    {
                        "file_id": "file-1",
                        "content": [{"type": "text", "text": "hello world"}],
                        "score": 0.95,
                    }
                ]
            }
        )
        mock_http.post = AsyncMock(return_value=search_resp)

        client = _make_client(http=mock_http)
        results = await client.search("hello")

        assert results == [{"id": "file-1", "content": "hello world", "score": 0.95}]

    @pytest.mark.asyncio
    async def test_search_returns_empty_on_error(self):
        mock_http = _mock_httpx_client()
        mock_http.post = AsyncMock(side_effect=RuntimeError("network error"))

        client = _make_client(http=mock_http)
        results = await client.search("anything")

        assert results == []

    @pytest.mark.asyncio
    async def test_write_uploads_and_attaches(self):
        mock_http = _mock_httpx_client()

        upload_resp = _mock_response(json_data={"id": "file-abc"})
        attach_resp = _mock_response(json_data={"id": "file-abc"})
        mock_http.post = AsyncMock(side_effect=[upload_resp, attach_resp])

        client = _make_client(http=mock_http, vector_store_id="vs_test")
        result = await client.write("test content")

        assert result is not None
        assert result["id"] == "file-abc"
        assert result["content"] == "test content"

    @pytest.mark.asyncio
    async def test_write_returns_none_on_error(self):
        mock_http = _mock_httpx_client()
        mock_http.post = AsyncMock(side_effect=RuntimeError("upload failed"))

        client = _make_client(http=mock_http)
        result = await client.write("test content")

        assert result is None

    @pytest.mark.asyncio
    async def test_update_deletes_then_writes(self):
        mock_http = _mock_httpx_client()

        delete_resp = _mock_response(status_code=200, json_data={})
        upload_resp = _mock_response(json_data={"id": "file-new"})
        attach_resp = _mock_response(json_data={"id": "file-new"})

        mock_http.delete = AsyncMock(return_value=delete_resp)
        mock_http.post = AsyncMock(side_effect=[upload_resp, attach_resp])

        client = _make_client(http=mock_http, vector_store_id="vs_test")
        result = await client.update("file-old", "updated content")

        mock_http.delete.assert_called_once()
        assert result is not None
        assert result["content"] == "updated content"

    @pytest.mark.asyncio
    async def test_update_writes_even_if_delete_fails(self):
        mock_http = _mock_httpx_client()

        mock_http.delete = AsyncMock(side_effect=RuntimeError("404 not found"))
        upload_resp = _mock_response(json_data={"id": "file-recovered"})
        attach_resp = _mock_response(json_data={"id": "file-recovered"})
        mock_http.post = AsyncMock(side_effect=[upload_resp, attach_resp])

        client = _make_client(http=mock_http, vector_store_id="vs_test")
        result = await client.update("file-missing", "recovered content")

        assert result is not None

    @pytest.mark.asyncio
    async def test_report_contradiction_is_noop(self):
        mock_http = _mock_httpx_client()
        client = _make_client(http=mock_http)

        # Must not raise and must not make any HTTP calls
        await client.report_contradiction("file-1", "observed contrary behavior")

        mock_http.get.assert_not_called()
        mock_http.post.assert_not_called()
        mock_http.delete.assert_not_called()
