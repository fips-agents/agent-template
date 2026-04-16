"""Tests for fipsagents.baseagent.memory_markdown — Level 1 + Level 2."""

from __future__ import annotations

from pathlib import Path

import pytest

from fipsagents.baseagent.config import MemoryConfig
from fipsagents.baseagent.memory import NullMemoryClient, create_memory_client
from fipsagents.baseagent.memory_markdown import (
    MarkdownMemoryClient,
    _parse_sections,
    create_markdown_client,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _make_level1(tmp_path: Path) -> MarkdownMemoryClient:
    cfg = tmp_path / ".memory-markdown.yaml"
    cfg.write_text("file: memory.md\n")
    client = await create_markdown_client(cfg)
    assert isinstance(client, MarkdownMemoryClient)
    return client


async def _make_level2(tmp_path: Path) -> MarkdownMemoryClient:
    cfg = tmp_path / ".memory-markdown.yaml"
    cfg.write_text("dir: memories\n")
    client = await create_markdown_client(cfg)
    assert isinstance(client, MarkdownMemoryClient)
    return client


# ---------------------------------------------------------------------------
# TestParseSections
# ---------------------------------------------------------------------------


class TestParseSections:
    def test_empty_string_returns_empty(self):
        assert _parse_sections("") == []

    def test_content_before_first_heading_is_ignored(self):
        text = "preamble text\nmore preamble\n\n## first\n\nbody\n"
        assert _parse_sections(text) == [("first", "body")]

    def test_multiple_sections_preserve_order(self):
        text = "## a\n\nbody a\n\n## b\n\nbody b\n"
        assert _parse_sections(text) == [("a", "body a"), ("b", "body b")]

    def test_body_with_blank_lines_preserved(self):
        text = "## x\n\nline1\n\nline2\n\n## y\n\nbody y\n"
        sections = _parse_sections(text)
        assert sections[0] == ("x", "line1\n\nline2")

    def test_heading_whitespace_stripped(self):
        assert _parse_sections("## heading   \n\nbody\n") == [("heading", "body")]


# ---------------------------------------------------------------------------
# TestCreateMarkdownClient — factory
# ---------------------------------------------------------------------------


class TestCreateMarkdownClient:
    @pytest.mark.asyncio
    async def test_level1_creates_client(self, tmp_path):
        client = await _make_level1(tmp_path)
        assert client._file == tmp_path / "memory.md"
        assert client._dir is None

    @pytest.mark.asyncio
    async def test_level2_creates_client(self, tmp_path):
        client = await _make_level2(tmp_path)
        assert client._dir == tmp_path / "memories"
        assert client._file is None

    @pytest.mark.asyncio
    async def test_level1_creates_file_if_missing(self, tmp_path):
        await _make_level1(tmp_path)
        assert (tmp_path / "memory.md").exists()

    @pytest.mark.asyncio
    async def test_level2_creates_dir_if_missing(self, tmp_path):
        await _make_level2(tmp_path)
        assert (tmp_path / "memories").is_dir()

    @pytest.mark.asyncio
    async def test_missing_config_returns_null(self, tmp_path):
        client = await create_markdown_client(tmp_path / "no-such-file.yaml")
        assert isinstance(client, NullMemoryClient)

    @pytest.mark.asyncio
    async def test_malformed_config_returns_null(self, tmp_path):
        cfg = tmp_path / ".memory-markdown.yaml"
        cfg.write_text("[[[not valid yaml")
        client = await create_markdown_client(cfg)
        assert isinstance(client, NullMemoryClient)

    @pytest.mark.asyncio
    async def test_both_file_and_dir_returns_null(self, tmp_path):
        cfg = tmp_path / ".memory-markdown.yaml"
        cfg.write_text("file: a.md\ndir: b/\n")
        client = await create_markdown_client(cfg)
        assert isinstance(client, NullMemoryClient)

    @pytest.mark.asyncio
    async def test_neither_file_nor_dir_returns_null(self, tmp_path):
        cfg = tmp_path / ".memory-markdown.yaml"
        cfg.write_text("# no keys\n")
        client = await create_markdown_client(cfg)
        assert isinstance(client, NullMemoryClient)

    @pytest.mark.asyncio
    async def test_relative_paths_resolved_from_config_dir(self, tmp_path):
        subdir = tmp_path / "configs"
        subdir.mkdir()
        cfg = subdir / ".memory-markdown.yaml"
        cfg.write_text("file: ../notes.md\n")
        client = await create_markdown_client(cfg)
        assert isinstance(client, MarkdownMemoryClient)
        assert (tmp_path / "notes.md").exists()


# ---------------------------------------------------------------------------
# TestLevel1Write
# ---------------------------------------------------------------------------


class TestLevel1Write:
    @pytest.mark.asyncio
    async def test_write_appends_section(self, tmp_path):
        client = await _make_level1(tmp_path)
        await client.write("content here", memory_id="topic-a")
        text = (tmp_path / "memory.md").read_text()
        assert "## topic-a" in text
        assert "content here" in text

    @pytest.mark.asyncio
    async def test_write_returns_dict_with_id_and_content(self, tmp_path):
        client = await _make_level1(tmp_path)
        result = await client.write("hello", memory_id="greeting")
        assert result["id"] == "greeting"
        assert result["content"] == "hello"
        assert "created_at" in result

    @pytest.mark.asyncio
    async def test_write_without_memory_id_uses_timestamp(self, tmp_path):
        client = await _make_level1(tmp_path)
        result = await client.write("auto-headed")
        # Timestamp headings start with a year: 20XX-...
        assert result["id"].startswith("20")
        text = (tmp_path / "memory.md").read_text()
        assert f"## {result['id']}" in text

    @pytest.mark.asyncio
    async def test_write_multiple_preserves_order(self, tmp_path):
        client = await _make_level1(tmp_path)
        await client.write("first body", memory_id="first")
        await client.write("second body", memory_id="second")
        text = (tmp_path / "memory.md").read_text()
        assert text.index("## first") < text.index("## second")


# ---------------------------------------------------------------------------
# TestLevel1Search
# ---------------------------------------------------------------------------


class TestLevel1Search:
    @pytest.mark.asyncio
    async def test_search_empty_query_returns_every_section(self, tmp_path):
        client = await _make_level1(tmp_path)
        await client.write("body a", memory_id="a")
        await client.write("body b", memory_id="b")
        results = await client.search("")
        assert [r["id"] for r in results] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_search_empty_query_preserves_file_order(self, tmp_path):
        client = await _make_level1(tmp_path)
        for heading in ["first", "second", "third"]:
            await client.write(f"body {heading}", memory_id=heading)
        results = await client.search("")
        assert [r["id"] for r in results] == ["first", "second", "third"]

    @pytest.mark.asyncio
    async def test_search_substring_matches_case_insensitively(self, tmp_path):
        client = await _make_level1(tmp_path)
        await client.write("the user's favorite color is teal", memory_id="colors")
        await client.write("the user's cat is named Mochi", memory_id="pets")
        results = await client.search("TEAL")
        assert [r["id"] for r in results] == ["colors"]

    @pytest.mark.asyncio
    async def test_search_no_match_returns_empty(self, tmp_path):
        client = await _make_level1(tmp_path)
        await client.write("body", memory_id="entry")
        results = await client.search("nothing-here")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_on_empty_file_returns_empty(self, tmp_path):
        client = await _make_level1(tmp_path)
        assert await client.search("") == []
        assert await client.search("anything") == []


# ---------------------------------------------------------------------------
# TestLevel1Update
# ---------------------------------------------------------------------------


class TestLevel1Update:
    @pytest.mark.asyncio
    async def test_update_replaces_section_body(self, tmp_path):
        client = await _make_level1(tmp_path)
        await client.write("original body", memory_id="topic")
        result = await client.update("topic", "new body")
        assert result is not None
        assert result["content"] == "new body"
        results = await client.search("")
        assert results[0]["content"] == "new body"

    @pytest.mark.asyncio
    async def test_update_preserves_other_sections(self, tmp_path):
        client = await _make_level1(tmp_path)
        await client.write("body a", memory_id="a")
        await client.write("body b", memory_id="b")
        await client.update("a", "body a updated")
        ids = [r["id"] for r in await client.search("")]
        assert ids == ["a", "b"]
        a_result = await client.search("updated")
        assert len(a_result) == 1
        assert a_result[0]["id"] == "a"

    @pytest.mark.asyncio
    async def test_update_nonexistent_section_returns_none(self, tmp_path):
        client = await _make_level1(tmp_path)
        await client.write("body", memory_id="exists")
        result = await client.update("does-not-exist", "new")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_on_missing_file_returns_none(self, tmp_path):
        client = await _make_level1(tmp_path)
        (tmp_path / "memory.md").unlink()
        result = await client.update("anything", "content")
        assert result is None


# ---------------------------------------------------------------------------
# TestLevel2Write / Search / Update
# ---------------------------------------------------------------------------


class TestLevel2:
    @pytest.mark.asyncio
    async def test_write_creates_file_named_after_memory_id(self, tmp_path):
        client = await _make_level2(tmp_path)
        await client.write("body content", memory_id="preferences")
        assert (tmp_path / "memories" / "preferences.md").exists()

    @pytest.mark.asyncio
    async def test_write_rejects_unsafe_filename(self, tmp_path):
        client = await _make_level2(tmp_path)
        result = await client.write("body", memory_id="../evil")
        # Write catches ValueError from _safe_filename and returns None.
        assert result is None

    @pytest.mark.asyncio
    async def test_search_returns_one_result_per_file(self, tmp_path):
        client = await _make_level2(tmp_path)
        await client.write("body alpha", memory_id="alpha")
        await client.write("body beta", memory_id="beta")
        results = await client.search("")
        assert [r["id"] for r in results] == ["alpha", "beta"]

    @pytest.mark.asyncio
    async def test_search_filters_by_substring(self, tmp_path):
        client = await _make_level2(tmp_path)
        await client.write("the user likes teal", memory_id="colors")
        await client.write("cat is mochi", memory_id="pets")
        results = await client.search("mochi")
        assert [r["id"] for r in results] == ["pets"]

    @pytest.mark.asyncio
    async def test_update_rewrites_file(self, tmp_path):
        client = await _make_level2(tmp_path)
        await client.write("original", memory_id="topic")
        await client.update("topic", "replacement")
        assert (tmp_path / "memories" / "topic.md").read_text().strip() == "replacement"

    @pytest.mark.asyncio
    async def test_update_nonexistent_file_returns_none(self, tmp_path):
        client = await _make_level2(tmp_path)
        assert await client.update("missing", "content") is None

    @pytest.mark.asyncio
    async def test_sort_order_stable_across_writes(self, tmp_path):
        client = await _make_level2(tmp_path)
        # Deliberately out-of-alphabetical write order.
        await client.write("c", memory_id="charlie")
        await client.write("a", memory_id="alpha")
        await client.write("b", memory_id="bravo")
        results = await client.search("")
        assert [r["id"] for r in results] == ["alpha", "bravo", "charlie"]


# ---------------------------------------------------------------------------
# TestContradiction
# ---------------------------------------------------------------------------


class TestContradiction:
    @pytest.mark.asyncio
    async def test_report_contradiction_is_noop_and_logs(self, tmp_path, caplog):
        client = await _make_level1(tmp_path)
        import logging
        with caplog.at_level(logging.WARNING):
            await client.report_contradiction("id", "desc")
        assert any("Contradiction reported" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# TestDispatcherIntegration — entering via create_memory_client
# ---------------------------------------------------------------------------


class TestDispatcherIntegration:
    @pytest.mark.asyncio
    async def test_markdown_backend_via_memory_config(self, tmp_path):
        cfg_file = tmp_path / ".memory-markdown.yaml"
        cfg_file.write_text("file: memory.md\n")
        mc = MemoryConfig(backend="markdown", config_path=str(cfg_file))
        client = await create_memory_client(config=mc)
        assert isinstance(client, MarkdownMemoryClient)
