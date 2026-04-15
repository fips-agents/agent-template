"""Tests for fipsagents.baseagent.memory — optional MemoryHub integration."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fipsagents.baseagent.memory import (
    MemoryClient,
    MemoryClientBase,
    NullMemoryClient,
    create_memory_client,
)


# ── NullMemoryClient ──────────────────────────────────────────────────────


class TestNullMemoryClient:
    """Every method returns empty results or None — never raises."""

    @pytest.mark.asyncio
    async def test_search_returns_empty_list(self):
        client = NullMemoryClient()
        result = await client.search("anything")
        assert result == []

    @pytest.mark.asyncio
    async def test_write_returns_none(self):
        client = NullMemoryClient()
        result = await client.write("some content")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_returns_none(self):
        client = NullMemoryClient()
        result = await client.update("mem-123", "updated content")
        assert result is None

    @pytest.mark.asyncio
    async def test_report_contradiction_returns_none(self):
        client = NullMemoryClient()
        result = await client.report_contradiction("mem-123", "stale info")
        assert result is None

    @pytest.mark.asyncio
    async def test_search_accepts_kwargs(self):
        """Extra keyword arguments are silently ignored."""
        client = NullMemoryClient()
        result = await client.search("query", limit=10, scope="project")
        assert result == []

    @pytest.mark.asyncio
    async def test_write_accepts_kwargs(self):
        client = NullMemoryClient()
        result = await client.write("content", weight=0.8, scope="user")
        assert result is None

    def test_is_subclass_of_base(self):
        assert issubclass(NullMemoryClient, MemoryClientBase)


# ── MemoryClient with mocked SDK ──────────────────────────────────────────


class _FakeSDK:
    """Minimal stand-in for the MemoryHub SDK.

    Unlike MagicMock, this does not auto-create attributes, so
    ``getattr(sdk, "search", None)`` returns ``None`` and
    ``hasattr(sdk, "__aenter__")`` returns ``False`` — matching the
    attribute-probing logic in memory.py.
    """


def _make_sdk(**overrides: Any) -> _FakeSDK:
    """Create a fake MemoryHub SDK with async methods."""
    sdk = _FakeSDK()
    sdk.search_memory = AsyncMock(return_value=[{"id": "m1", "content": "found"}])
    sdk.write_memory = AsyncMock(return_value={"id": "m2", "content": "written"})
    sdk.update_memory = AsyncMock(return_value={"id": "m1", "content": "updated"})
    sdk.report_contradiction = AsyncMock(return_value=None)
    for key, value in overrides.items():
        setattr(sdk, key, value)
    return sdk


class TestMemoryClient:
    @pytest.mark.asyncio
    async def test_search_returns_list(self):
        sdk = _make_sdk()
        client = MemoryClient(sdk=sdk)
        results = await client.search("test query", limit=5)
        assert results == [{"id": "m1", "content": "found"}]
        sdk.search_memory.assert_awaited_once_with(query="test query", limit=5)

    @pytest.mark.asyncio
    async def test_search_unwraps_wrapper_object(self):
        """Some SDK versions return an object with a .memories attribute."""
        wrapper = SimpleNamespace(memories=[{"id": "m1"}])
        sdk = _make_sdk(search_memory=AsyncMock(return_value=wrapper))
        client = MemoryClient(sdk=sdk)
        results = await client.search("query")
        assert results == [{"id": "m1"}]

    @pytest.mark.asyncio
    async def test_write_returns_dict(self):
        sdk = _make_sdk()
        client = MemoryClient(sdk=sdk)
        result = await client.write("new memory", weight=0.9)
        assert result == {"id": "m2", "content": "written"}
        sdk.write_memory.assert_awaited_once_with(content="new memory", weight=0.9)

    @pytest.mark.asyncio
    async def test_update_returns_dict(self):
        sdk = _make_sdk()
        client = MemoryClient(sdk=sdk)
        result = await client.update("m1", "revised content", weight=1.0)
        assert result == {"id": "m1", "content": "updated"}
        sdk.update_memory.assert_awaited_once_with(
            memory_id="m1", content="revised content", weight=1.0
        )

    @pytest.mark.asyncio
    async def test_report_contradiction(self):
        sdk = _make_sdk()
        client = MemoryClient(sdk=sdk)
        await client.report_contradiction("m1", "outdated policy")
        sdk.report_contradiction.assert_awaited_once_with(
            memory_id="m1", description="outdated policy"
        )

    def test_is_subclass_of_base(self):
        assert issubclass(MemoryClient, MemoryClientBase)


# ── Graceful degradation on SDK errors ────────────────────────────────────


class TestGracefulDegradation:
    """When the MemoryHub server is unreachable, methods log and return empty."""

    @pytest.mark.asyncio
    async def test_search_error_returns_empty(self, caplog: pytest.LogCaptureFixture):
        sdk = _make_sdk(search_memory=AsyncMock(side_effect=ConnectionError("offline")))
        client = MemoryClient(sdk=sdk)
        with caplog.at_level(logging.WARNING):
            results = await client.search("query")
        assert results == []
        assert "MemoryHub search failed" in caplog.text

    @pytest.mark.asyncio
    async def test_write_error_returns_none(self, caplog: pytest.LogCaptureFixture):
        sdk = _make_sdk(write_memory=AsyncMock(side_effect=TimeoutError("timeout")))
        client = MemoryClient(sdk=sdk)
        with caplog.at_level(logging.WARNING):
            result = await client.write("content")
        assert result is None
        assert "MemoryHub write failed" in caplog.text

    @pytest.mark.asyncio
    async def test_update_error_returns_none(self, caplog: pytest.LogCaptureFixture):
        sdk = _make_sdk(update_memory=AsyncMock(side_effect=RuntimeError("boom")))
        client = MemoryClient(sdk=sdk)
        with caplog.at_level(logging.WARNING):
            result = await client.update("m1", "new")
        assert result is None
        assert "MemoryHub update failed" in caplog.text

    @pytest.mark.asyncio
    async def test_report_contradiction_error_is_swallowed(
        self, caplog: pytest.LogCaptureFixture,
    ):
        sdk = _make_sdk(
            report_contradiction=AsyncMock(side_effect=OSError("network")),
        )
        client = MemoryClient(sdk=sdk)
        with caplog.at_level(logging.WARNING):
            await client.report_contradiction("m1", "stale")
        assert "report_contradiction failed" in caplog.text

    @pytest.mark.asyncio
    async def test_write_non_dict_returns_none(self):
        """If the SDK returns a non-dict, we normalise to None."""
        sdk = _make_sdk(write_memory=AsyncMock(return_value="not-a-dict"))
        client = MemoryClient(sdk=sdk)
        result = await client.write("content")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_non_dict_returns_none(self):
        sdk = _make_sdk(update_memory=AsyncMock(return_value=42))
        client = MemoryClient(sdk=sdk)
        result = await client.update("m1", "content")
        assert result is None


# ── create_memory_client factory ──────────────────────────────────────────


class TestCreateMemoryClient:
    @pytest.mark.asyncio
    async def test_no_config_file_returns_null(self, tmp_path: Path):
        """When .memoryhub.yaml doesn't exist, returns NullMemoryClient."""
        client = await create_memory_client(tmp_path / ".memoryhub.yaml")
        assert isinstance(client, NullMemoryClient)

    @pytest.mark.asyncio
    async def test_missing_package_returns_null_with_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ):
        """Config exists but memoryhub isn't installed."""
        config_file = tmp_path / ".memoryhub.yaml"
        config_file.write_text("server_url: http://localhost:8080\n")

        with (
            patch.dict("sys.modules", {"memoryhub": None}),
            caplog.at_level(logging.WARNING),
        ):
            client = await create_memory_client(config_file)

        assert isinstance(client, NullMemoryClient)
        assert "memoryhub package is not installed" in caplog.text

    @pytest.mark.asyncio
    async def test_valid_config_creates_real_client(self, tmp_path: Path):
        """Config exists and memoryhub is importable — produces MemoryClient."""
        config_file = tmp_path / ".memoryhub.yaml"
        config_file.write_text("server_url: http://memory:8080\n")

        # Create a key file
        key_dir = tmp_path / ".config" / "memoryhub"
        key_dir.mkdir(parents=True)
        (key_dir / "api-key").write_text("  test-key-123  \n")

        mock_sdk_instance = _make_sdk()

        mock_memoryhub = MagicMock()
        mock_memoryhub.MemoryHubClient.return_value = mock_sdk_instance

        with (
            patch.dict("sys.modules", {"memoryhub": mock_memoryhub}),
            patch("fipsagents.baseagent.memory.Path.home", return_value=tmp_path),
        ):
            client = await create_memory_client(config_file)

        assert isinstance(client, MemoryClient)
        mock_memoryhub.MemoryHubClient.assert_called_once_with(
            api_key="test-key-123", server_url="http://memory:8080",
        )

    @pytest.mark.asyncio
    async def test_api_key_from_config(self, tmp_path: Path):
        """api_key in YAML takes precedence over the key file."""
        config_file = tmp_path / ".memoryhub.yaml"
        config_file.write_text(
            "api_key: yaml-key-456\nserver_url: http://memory:8080\n"
        )

        mock_sdk_instance = _make_sdk()
        mock_memoryhub = MagicMock()
        mock_memoryhub.MemoryHubClient.return_value = mock_sdk_instance

        with patch.dict("sys.modules", {"memoryhub": mock_memoryhub}):
            client = await create_memory_client(config_file)

        assert isinstance(client, MemoryClient)
        call_kwargs = mock_memoryhub.MemoryHubClient.call_args[1]
        assert call_kwargs["api_key"] == "yaml-key-456"

    @pytest.mark.asyncio
    async def test_sdk_init_failure_returns_null(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ):
        """If SDK initialisation blows up, fall back to NullMemoryClient."""
        config_file = tmp_path / ".memoryhub.yaml"
        config_file.write_text("server_url: http://memory:8080\n")

        mock_memoryhub = MagicMock()
        mock_memoryhub.MemoryHubClient.side_effect = RuntimeError("bad config")

        with (
            patch.dict("sys.modules", {"memoryhub": mock_memoryhub}),
            caplog.at_level(logging.WARNING),
        ):
            client = await create_memory_client(config_file)

        assert isinstance(client, NullMemoryClient)
        assert "Failed to initialise MemoryHub" in caplog.text

    @pytest.mark.asyncio
    async def test_register_session_called_when_available(self, tmp_path: Path):
        """If the SDK has register_session, the factory calls it."""
        config_file = tmp_path / ".memoryhub.yaml"
        config_file.write_text("server_url: http://memory:8080\n")

        mock_sdk_instance = _make_sdk()
        mock_sdk_instance.register_session = AsyncMock()

        mock_memoryhub = MagicMock()
        mock_memoryhub.MemoryHubClient.return_value = mock_sdk_instance

        with patch.dict("sys.modules", {"memoryhub": mock_memoryhub}):
            client = await create_memory_client(config_file)

        assert isinstance(client, MemoryClient)
        mock_sdk_instance.register_session.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_register_session_failure_falls_back(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ):
        """register_session failure triggers fallback to NullMemoryClient."""
        config_file = tmp_path / ".memoryhub.yaml"
        config_file.write_text("server_url: http://memory:8080\n")

        mock_sdk_instance = _make_sdk()
        mock_sdk_instance.register_session = AsyncMock(
            side_effect=ConnectionError("cannot reach server"),
        )

        mock_memoryhub = MagicMock()
        mock_memoryhub.MemoryHubClient.return_value = mock_sdk_instance

        with (
            patch.dict("sys.modules", {"memoryhub": mock_memoryhub}),
            caplog.at_level(logging.WARNING),
        ):
            client = await create_memory_client(config_file)

        assert isinstance(client, NullMemoryClient)
        assert "Failed to initialise MemoryHub" in caplog.text

    @pytest.mark.asyncio
    async def test_default_config_path(self, monkeypatch: pytest.MonkeyPatch):
        """Default config_path argument is .memoryhub.yaml in cwd."""
        # Point to a dir that definitely lacks the config file
        monkeypatch.chdir("/tmp")
        client = await create_memory_client()
        assert isinstance(client, NullMemoryClient)

    @pytest.mark.asyncio
    async def test_empty_config_file(self, tmp_path: Path):
        """An empty YAML file should not crash — just produce a client."""
        config_file = tmp_path / ".memoryhub.yaml"
        config_file.write_text("")

        mock_sdk_instance = _make_sdk()
        mock_memoryhub = MagicMock()
        mock_memoryhub.MemoryHubClient.return_value = mock_sdk_instance

        with patch.dict("sys.modules", {"memoryhub": mock_memoryhub}):
            client = await create_memory_client(config_file)

        assert isinstance(client, MemoryClient)
