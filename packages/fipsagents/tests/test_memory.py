"""Tests for fipsagents.baseagent.memory — NullMemoryClient, MemoryClient, factory."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fipsagents.baseagent.config import MemoryConfig
from fipsagents.baseagent.memory import (
    MemoryClient,
    MemoryClientBase,
    NullMemoryClient,
    create_memory_client,
)


# ---------------------------------------------------------------------------
# NullMemoryClient
# ---------------------------------------------------------------------------


class TestNullMemoryClient:
    @pytest.mark.asyncio
    async def test_search_returns_empty_list(self):
        client = NullMemoryClient()
        result = await client.search("any query")
        assert result == []

    @pytest.mark.asyncio
    async def test_write_returns_none(self):
        client = NullMemoryClient()
        result = await client.write("some memory")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_returns_none(self):
        client = NullMemoryClient()
        result = await client.update("mem-id-123", "updated content")
        assert result is None

    @pytest.mark.asyncio
    async def test_report_contradiction_returns_none(self):
        client = NullMemoryClient()
        result = await client.report_contradiction("mem-id-123", "contradicts X")
        assert result is None

    @pytest.mark.asyncio
    async def test_search_accepts_kwargs(self):
        client = NullMemoryClient()
        result = await client.search("query", limit=5, project="test")
        assert result == []


# ---------------------------------------------------------------------------
# MemoryClient — search
# ---------------------------------------------------------------------------


class TestMemoryClientSearch:
    @pytest.mark.asyncio
    async def test_search_returns_list_directly(self):
        sdk = MagicMock()
        sdk.search = AsyncMock(return_value=[{"id": "1", "content": "hello"}])
        client = MemoryClient(sdk=sdk)
        result = await client.search("hello")
        assert result == [{"id": "1", "content": "hello"}]

    @pytest.mark.asyncio
    async def test_search_with_results_attr(self):
        """SDK v0.5.0 returns SearchResult with .results list."""
        mem = MagicMock()
        mem.model_dump = MagicMock(return_value={"id": "2", "content": "world"})
        search_result = MagicMock()
        search_result.results = [mem]
        # Make it not a list so the isinstance(result, list) path is skipped
        del search_result.__class__.__iter__  # ensure it's not iterable as list check

        sdk = MagicMock()
        sdk.search = AsyncMock(return_value=search_result)
        client = MemoryClient(sdk=sdk)
        result = await client.search("world")
        assert len(result) == 1
        assert result[0]["content"] == "world"

    @pytest.mark.asyncio
    async def test_search_falls_back_to_search_memory(self):
        """Older SDK uses .search_memory() not .search()."""
        sdk = MagicMock(spec=[])  # empty spec — no .search attribute
        sdk.search_memory = AsyncMock(return_value=[{"id": "3"}])
        client = MemoryClient(sdk=sdk)
        result = await client.search("q")
        assert result == [{"id": "3"}]

    @pytest.mark.asyncio
    async def test_search_exception_returns_empty(self):
        sdk = MagicMock()
        sdk.search = AsyncMock(side_effect=RuntimeError("network error"))
        client = MemoryClient(sdk=sdk)
        result = await client.search("query")
        assert result == []


# ---------------------------------------------------------------------------
# MemoryClient — write
# ---------------------------------------------------------------------------


class TestMemoryClientWrite:
    @pytest.mark.asyncio
    async def test_write_returns_dict_directly(self):
        sdk = MagicMock()
        sdk.write = AsyncMock(return_value={"id": "new-mem"})
        client = MemoryClient(sdk=sdk)
        result = await client.write("some content")
        assert result == {"id": "new-mem"}

    @pytest.mark.asyncio
    async def test_write_pydantic_model_dump(self):
        write_result = MagicMock()
        write_result.model_dump = MagicMock(return_value={"id": "abc", "content": "x"})
        sdk = MagicMock()
        sdk.write = AsyncMock(return_value=write_result)
        client = MemoryClient(sdk=sdk)
        result = await client.write("x")
        assert result == {"id": "abc", "content": "x"}

    @pytest.mark.asyncio
    async def test_write_falls_back_to_write_memory(self):
        sdk = MagicMock(spec=[])
        sdk.write_memory = AsyncMock(return_value={"id": "old"})
        client = MemoryClient(sdk=sdk)
        result = await client.write("content")
        assert result == {"id": "old"}

    @pytest.mark.asyncio
    async def test_write_exception_returns_none(self):
        sdk = MagicMock()
        sdk.write = AsyncMock(side_effect=ConnectionError("unreachable"))
        client = MemoryClient(sdk=sdk)
        result = await client.write("data")
        assert result is None


# ---------------------------------------------------------------------------
# MemoryClient — update
# ---------------------------------------------------------------------------


class TestMemoryClientUpdate:
    @pytest.mark.asyncio
    async def test_update_returns_dict(self):
        sdk = MagicMock()
        sdk.update = AsyncMock(return_value={"id": "mem-1", "updated": True})
        client = MemoryClient(sdk=sdk)
        result = await client.update("mem-1", "new content")
        assert result == {"id": "mem-1", "updated": True}

    @pytest.mark.asyncio
    async def test_update_falls_back_to_update_memory(self):
        sdk = MagicMock(spec=[])
        sdk.update_memory = AsyncMock(return_value={"id": "mem-2"})
        client = MemoryClient(sdk=sdk)
        result = await client.update("mem-2", "content")
        assert result == {"id": "mem-2"}

    @pytest.mark.asyncio
    async def test_update_exception_returns_none(self):
        sdk = MagicMock()
        sdk.update = AsyncMock(side_effect=Exception("fail"))
        client = MemoryClient(sdk=sdk)
        result = await client.update("id", "content")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_non_dict_result_returns_none(self):
        sdk = MagicMock()
        sdk.update = AsyncMock(return_value="not a dict")
        client = MemoryClient(sdk=sdk)
        result = await client.update("id", "content")
        assert result is None


# ---------------------------------------------------------------------------
# MemoryClient — report_contradiction
# ---------------------------------------------------------------------------


class TestMemoryClientReportContradiction:
    @pytest.mark.asyncio
    async def test_report_contradiction_calls_sdk(self):
        sdk = MagicMock()
        sdk.report_contradiction = AsyncMock(return_value=None)
        client = MemoryClient(sdk=sdk)
        await client.report_contradiction("mem-1", "old fact contradicted")
        sdk.report_contradiction.assert_called_once_with(
            memory_id="mem-1", description="old fact contradicted"
        )

    @pytest.mark.asyncio
    async def test_report_contradiction_exception_swallowed(self):
        sdk = MagicMock()
        sdk.report_contradiction = AsyncMock(side_effect=RuntimeError("server down"))
        client = MemoryClient(sdk=sdk)
        # Should not raise
        await client.report_contradiction("mem-1", "desc")


# ---------------------------------------------------------------------------
# create_memory_client factory
# ---------------------------------------------------------------------------


class TestCreateMemoryClientFactory:
    @pytest.mark.asyncio
    async def test_returns_null_when_config_missing(self, tmp_path):
        client = await create_memory_client(tmp_path / "nonexistent.yaml")
        assert isinstance(client, NullMemoryClient)

    @pytest.mark.asyncio
    async def test_returns_null_when_memoryhub_not_installed(self, tmp_path):
        config_file = tmp_path / ".memoryhub.yaml"
        config_file.write_text("server_url: http://localhost:8000\n")

        # Simulate memoryhub not being importable
        with patch.dict(sys.modules, {"memoryhub": None}):
            client = await create_memory_client(config_file)
        assert isinstance(client, NullMemoryClient)

    @pytest.mark.asyncio
    async def test_returns_null_on_sdk_init_failure(self, tmp_path):
        config_file = tmp_path / ".memoryhub.yaml"
        config_file.write_text("server_url: http://localhost:8000\n")

        mock_memoryhub = MagicMock()
        mock_memoryhub.MemoryHubClient.side_effect = RuntimeError("cannot connect")

        with patch.dict(sys.modules, {"memoryhub": mock_memoryhub}):
            client = await create_memory_client(config_file)
        assert isinstance(client, NullMemoryClient)

    @pytest.mark.asyncio
    async def test_returns_null_when_config_has_no_server_url(self, tmp_path, caplog):
        """A stub `.memoryhub.yaml` (comment-only or missing server_url) must
        short-circuit to NullMemoryClient with a single info log — no
        traceback. Regression: the SDK raises MemoryHubError("url is
        required") and the generic except path logs exc_info=True, which
        spooks first-time readers of `make run-local`."""
        config_file = tmp_path / ".memoryhub.yaml"
        config_file.write_text("# MemoryHub configuration — see docs for details\n")

        mock_memoryhub = MagicMock()

        with caplog.at_level("DEBUG", logger="fipsagents.baseagent.memory"):
            with patch.dict(sys.modules, {"memoryhub": mock_memoryhub}):
                client = await create_memory_client(config_file)

        assert isinstance(client, NullMemoryClient)
        # Did not attempt to construct the SDK — that's the whole point.
        mock_memoryhub.MemoryHubClient.assert_not_called()
        # Only the friendly info line — no traceback / warning.
        records = [r for r in caplog.records if r.name == "fipsagents.baseagent.memory"]
        assert any(r.levelname == "INFO" and "no server_url" in r.message for r in records)
        assert not any(r.levelname == "WARNING" for r in records)

    @pytest.mark.asyncio
    async def test_returns_memory_client_when_configured(self, tmp_path):
        config_file = tmp_path / ".memoryhub.yaml"
        config_file.write_text("server_url: http://localhost:8000\n")

        mock_sdk_instance = MagicMock()
        # No __aenter__ to keep the code path simple
        del mock_sdk_instance.__aenter__
        if hasattr(mock_sdk_instance, "register_session"):
            del mock_sdk_instance.register_session

        mock_memoryhub = MagicMock()
        mock_memoryhub.MemoryHubClient.return_value = mock_sdk_instance

        with patch.dict(sys.modules, {"memoryhub": mock_memoryhub}):
            client = await create_memory_client(config_file)
        assert isinstance(client, MemoryClient)

    @pytest.mark.asyncio
    async def test_env_var_placeholders_are_substituted(self, tmp_path, monkeypatch):
        """.memoryhub.yaml may contain ${VAR:-default} placeholders; the
        factory expands them before instantiating the SDK client."""
        config_file = tmp_path / ".memoryhub.yaml"
        config_file.write_text(
            "server_url: ${TEST_MEMORYHUB_URL:-http://default:8000}\n"
            "api_key: ${TEST_MEMORYHUB_KEY:-fallback-key}\n"
        )
        monkeypatch.setenv("TEST_MEMORYHUB_URL", "http://from-env:9000")
        # TEST_MEMORYHUB_KEY intentionally unset — exercises the default branch.

        mock_sdk_instance = MagicMock()
        del mock_sdk_instance.__aenter__
        if hasattr(mock_sdk_instance, "register_session"):
            del mock_sdk_instance.register_session

        mock_memoryhub = MagicMock()
        mock_memoryhub.MemoryHubClient.return_value = mock_sdk_instance

        with patch.dict(sys.modules, {"memoryhub": mock_memoryhub}):
            await create_memory_client(config_file)

        # The SDK must have been invoked with the substituted values.
        kwargs = mock_memoryhub.MemoryHubClient.call_args.kwargs
        assert kwargs["server_url"] == "http://from-env:9000"
        assert kwargs["api_key"] == "fallback-key"


# ---------------------------------------------------------------------------
# Backend dispatch via MemoryConfig
# ---------------------------------------------------------------------------


class TestBackendDispatch:
    @pytest.mark.asyncio
    async def test_null_backend_returns_null_client(self):
        client = await create_memory_client(config=MemoryConfig(backend="null"))
        assert isinstance(client, NullMemoryClient)

    @pytest.mark.asyncio
    async def test_memoryhub_backend_dispatches_to_memoryhub(self, tmp_path):
        # Nonexistent config file causes _create_memoryhub_client to return Null,
        # but it proves dispatch reached the memoryhub path (not the sqlite/pgvector
        # paths) — if it had hit a different branch we would get a different log msg.
        nonexistent = tmp_path / "no-such-file.yaml"
        client = await create_memory_client(
            config=MemoryConfig(backend="memoryhub", config_path=str(nonexistent))
        )
        assert isinstance(client, NullMemoryClient)

    @pytest.mark.asyncio
    async def test_sqlite_backend_falls_back_when_module_missing(self):
        # memory_sqlite does not exist yet; ImportError must produce NullMemoryClient.
        with patch.dict(
            sys.modules,
            {"fipsagents.baseagent.memory_sqlite": None},
        ):
            client = await create_memory_client(config=MemoryConfig(backend="sqlite"))
        assert isinstance(client, NullMemoryClient)

    @pytest.mark.asyncio
    async def test_pgvector_backend_falls_back_when_module_missing(self):
        with patch.dict(
            sys.modules,
            {"fipsagents.baseagent.memory_pgvector": None},
        ):
            client = await create_memory_client(
                config=MemoryConfig(backend="pgvector")
            )
        assert isinstance(client, NullMemoryClient)

    @pytest.mark.asyncio
    async def test_custom_backend_without_class_returns_null(self):
        # backend_class defaults to None — factory must return Null immediately.
        client = await create_memory_client(config=MemoryConfig(backend="custom"))
        assert isinstance(client, NullMemoryClient)

    @pytest.mark.asyncio
    async def test_custom_backend_loads_valid_class(self):
        class MyMemClient(MemoryClientBase):
            async def search(self, query, **kwargs):
                return []

            async def write(self, content, **kwargs):
                return None

            async def update(self, memory_id, content, **kwargs):
                return None

            async def report_contradiction(self, memory_id, description):
                return None

        mock_module = MagicMock()
        mock_module.MyMemClient = MyMemClient

        with patch("importlib.import_module", return_value=mock_module):
            client = await create_memory_client(
                config=MemoryConfig(
                    backend="custom", backend_class="mymod.MyMemClient"
                )
            )

        assert isinstance(client, MyMemClient)

    @pytest.mark.asyncio
    async def test_custom_backend_rejects_non_subclass(self):
        class NotAMemClient:
            """Does NOT inherit from MemoryClientBase."""

        mock_module = MagicMock()
        mock_module.NotAMemClient = NotAMemClient

        with patch("importlib.import_module", return_value=mock_module):
            client = await create_memory_client(
                config=MemoryConfig(
                    backend="custom", backend_class="mymod.NotAMemClient"
                )
            )

        assert isinstance(client, NullMemoryClient)

    @pytest.mark.asyncio
    async def test_no_backend_falls_through_to_autodetect(self, tmp_path):
        # backend=None means auto-detect; nonexistent config_path yields Null.
        nonexistent = tmp_path / "no-memoryhub.yaml"
        client = await create_memory_client(
            config=MemoryConfig(config_path=str(nonexistent))
        )
        assert isinstance(client, NullMemoryClient)

    @pytest.mark.asyncio
    async def test_legacy_call_without_config_still_works(self, tmp_path):
        # Backward-compat: no config kwarg at all, positional path only.
        client = await create_memory_client(tmp_path / "nonexistent.yaml")
        assert isinstance(client, NullMemoryClient)


# ---------------------------------------------------------------------------
# MemoryConfig field validation
# ---------------------------------------------------------------------------


class TestMemoryConfigValidation:
    def test_empty_string_backend_coerced_to_none(self):
        cfg = MemoryConfig(backend="")
        assert cfg.backend is None

    def test_whitespace_backend_coerced_to_none(self):
        cfg = MemoryConfig(backend="  ")
        assert cfg.backend is None

    def test_valid_backend_preserved(self):
        cfg = MemoryConfig(backend="sqlite")
        assert cfg.backend == "sqlite"
