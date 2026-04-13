"""Tests for fipsagents.baseagent.tools — decorator, registry, schema generation, dispatch."""

from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path
from typing import Optional

import pytest

from fipsagents.baseagent.tools import (
    ToolCall,
    ToolMeta,
    ToolRegistry,
    ToolResult,
    _TOOL_MARKER,
    _is_optional,
    _type_to_schema,
    tool,
)

# We need pydantic for testing schema generation with model parameters.
from pydantic import BaseModel


# ── Fixtures & helpers ──────────────────────────────────────────────────────


class SampleModel(BaseModel):
    """A pydantic model used as a tool parameter in tests."""

    name: str
    count: int = 0


def _make_sync_tool(
    name: str = "sync_tool",
    visibility: str = "both",
    description: str = "A sync tool",
):
    """Factory for a simple sync tool."""

    @tool(description=description, visibility=visibility, name=name)
    def fn(x: str) -> str:
        return f"sync:{x}"

    return fn


def _make_async_tool(
    name: str = "async_tool",
    visibility: str = "both",
    description: str = "An async tool",
):
    """Factory for a simple async tool."""

    @tool(description=description, visibility=visibility, name=name)
    async def fn(x: str) -> str:
        return f"async:{x}"

    return fn


# ── @tool decorator ─────────────────────────────────────────────────────────


class TestToolDecorator:
    def test_attaches_metadata(self):
        @tool(description="Do stuff", visibility="agent_only")
        async def my_tool(query: str) -> str:
            """Docstring here."""
            return query

        assert hasattr(my_tool, _TOOL_MARKER)
        meta: ToolMeta = getattr(my_tool, _TOOL_MARKER)
        assert meta.name == "my_tool"
        assert "Do stuff" in meta.description
        assert meta.visibility == "agent_only"
        assert meta.is_async is True

    def test_name_override(self):
        @tool(description="desc", visibility="llm_only", name="custom_name")
        def fn() -> None:
            pass

        meta: ToolMeta = getattr(fn, _TOOL_MARKER)
        assert meta.name == "custom_name"

    def test_preserves_original_function(self):
        """The decorated function should still be directly callable."""

        @tool(description="desc", visibility="both")
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == 5

    def test_preserves_async_function(self):
        @tool(description="desc", visibility="both")
        async def greet(name: str) -> str:
            return f"hello {name}"

        result = asyncio.get_event_loop().run_until_complete(greet("world"))
        assert result == "hello world"

    def test_sync_function_detected(self):
        @tool(description="desc", visibility="both")
        def sync_fn() -> None:
            pass

        meta: ToolMeta = getattr(sync_fn, _TOOL_MARKER)
        assert meta.is_async is False

    def test_docstring_appended_to_description(self):
        @tool(description="Short desc", visibility="both")
        def fn() -> None:
            """Longer explanation from docstring."""
            pass

        meta: ToolMeta = getattr(fn, _TOOL_MARKER)
        assert "Short desc" in meta.description
        assert "Longer explanation from docstring" in meta.description

    def test_docstring_not_duplicated_when_matches_description(self):
        desc = "Exact match"

        @tool(description=desc, visibility="both")
        def fn() -> None:
            """Exact match"""
            pass

        meta: ToolMeta = getattr(fn, _TOOL_MARKER)
        # Should not repeat the description.
        assert meta.description == desc

    @pytest.mark.parametrize(
        "bad_vis",
        ["invisible", "public", "private", "", "AGENT_ONLY"],
        ids=["invisible", "public", "private", "empty", "wrong-case"],
    )
    def test_invalid_visibility_raises(self, bad_vis: str):
        with pytest.raises(ValueError, match="visibility must be one of"):
            @tool(description="desc", visibility=bad_vis)
            def fn() -> None:
                pass

    def test_parameters_extracted_from_signature(self):
        @tool(description="desc", visibility="both")
        def search(query: str, limit: int = 10) -> list:
            pass

        meta: ToolMeta = getattr(search, _TOOL_MARKER)
        props = meta.parameters.get("properties", {})
        assert "query" in props
        assert "limit" in props
        assert props["query"]["type"] == "string"
        assert props["limit"]["type"] == "integer"
        # query is required (no default), limit is not
        assert "query" in meta.parameters.get("required", [])
        assert "limit" not in meta.parameters.get("required", [])


# ── ToolCall and ToolResult models ──────────────────────────────────────────


class TestDataModels:
    def test_tool_call_defaults(self):
        tc = ToolCall(name="test")
        assert tc.name == "test"
        assert tc.arguments == {}
        assert len(tc.call_id) > 0

    def test_tool_call_with_arguments(self):
        tc = ToolCall(name="search", arguments={"q": "hello"}, call_id="abc123")
        assert tc.arguments["q"] == "hello"
        assert tc.call_id == "abc123"

    def test_tool_result_success(self):
        tr = ToolResult(call_id="x", name="test", result="ok")
        assert tr.is_error is False
        assert tr.result == "ok"

    def test_tool_result_error(self):
        tr = ToolResult(call_id="x", name="test", error="something broke")
        assert tr.is_error is True
        assert "something broke" in tr.error

    def test_tool_result_default_result_is_empty(self):
        tr = ToolResult(call_id="x", name="test")
        assert tr.result == ""
        assert tr.error is None


# ── ToolRegistry registration & retrieval ───────────────────────────────────


class TestToolRegistryRegistration:
    def test_register_and_get(self):
        registry = ToolRegistry()
        fn = _make_sync_tool(name="my_tool")
        meta = registry.register(fn)
        assert meta.name == "my_tool"
        assert registry.get("my_tool") is meta

    def test_get_returns_none_for_unknown(self):
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_duplicate_name_raises(self):
        registry = ToolRegistry()
        registry.register(_make_sync_tool(name="dup"))
        with pytest.raises(ValueError, match="already registered"):
            registry.register(_make_sync_tool(name="dup"))

    def test_non_decorated_function_raises(self):
        registry = ToolRegistry()

        def plain_fn():
            pass

        with pytest.raises(ValueError, match="not decorated with @tool"):
            registry.register(plain_fn)

    def test_get_all(self):
        registry = ToolRegistry()
        registry.register(_make_sync_tool(name="a"))
        registry.register(_make_async_tool(name="b"))
        all_tools = registry.get_all()
        names = {t.name for t in all_tools}
        assert names == {"a", "b"}


# ── Plane filtering ─────────────────────────────────────────────────────────


class TestPlaneFiltering:
    @pytest.fixture()
    def registry_with_all_planes(self):
        reg = ToolRegistry()
        reg.register(_make_sync_tool(name="agent", visibility="agent_only"))
        reg.register(_make_sync_tool(name="llm", visibility="llm_only"))
        reg.register(_make_sync_tool(name="shared", visibility="both"))
        return reg

    def test_get_llm_tools(self, registry_with_all_planes):
        names = {t.name for t in registry_with_all_planes.get_llm_tools()}
        assert names == {"llm", "shared"}

    def test_get_agent_tools(self, registry_with_all_planes):
        names = {t.name for t in registry_with_all_planes.get_agent_tools()}
        assert names == {"agent", "shared"}

    def test_agent_only_not_in_llm_tools(self, registry_with_all_planes):
        llm_names = {t.name for t in registry_with_all_planes.get_llm_tools()}
        assert "agent" not in llm_names

    def test_llm_only_not_in_agent_tools(self, registry_with_all_planes):
        agent_names = {t.name for t in registry_with_all_planes.get_agent_tools()}
        assert "llm" not in agent_names


# ── Auto-discovery from directory ───────────────────────────────────────────


class TestDiscovery:
    def test_discover_from_directory(self, tmp_path: Path):
        """Create temp .py files with @tool functions and discover them."""
        tool_file = tmp_path / "search.py"
        tool_file.write_text(textwrap.dedent("""\
            from fipsagents.baseagent.tools import tool

            @tool(description="Search things", visibility="llm_only")
            async def web_search(query: str) -> str:
                return f"results for {query}"

            @tool(description="Local search", visibility="agent_only")
            def local_search(path: str) -> str:
                return f"found in {path}"
        """))

        registry = ToolRegistry()
        found = registry.discover(tmp_path)
        names = {m.name for m in found}
        assert "web_search" in names
        assert "local_search" in names
        assert len(found) == 2

    def test_discover_skips_underscored_files(self, tmp_path: Path):
        (tmp_path / "_internal.py").write_text(textwrap.dedent("""\
            from fipsagents.baseagent.tools import tool

            @tool(description="hidden", visibility="both")
            def hidden() -> None:
                pass
        """))
        registry = ToolRegistry()
        found = registry.discover(tmp_path)
        assert len(found) == 0

    def test_discover_nonexistent_dir(self, tmp_path: Path):
        registry = ToolRegistry()
        found = registry.discover(tmp_path / "no_such_dir")
        assert found == []

    def test_discover_skips_files_with_errors(self, tmp_path: Path):
        (tmp_path / "broken.py").write_text("raise RuntimeError('boom')")
        (tmp_path / "good.py").write_text(textwrap.dedent("""\
            from fipsagents.baseagent.tools import tool

            @tool(description="works", visibility="both")
            def good_tool() -> str:
                return "ok"
        """))
        registry = ToolRegistry()
        found = registry.discover(tmp_path)
        assert len(found) == 1
        assert found[0].name == "good_tool"

    def test_discover_does_not_duplicate(self, tmp_path: Path):
        (tmp_path / "dup.py").write_text(textwrap.dedent("""\
            from fipsagents.baseagent.tools import tool

            @tool(description="dup", visibility="both")
            def my_fn() -> None:
                pass
        """))
        registry = ToolRegistry()
        registry.discover(tmp_path)
        # Discover again — same tool should not be added twice.
        found_second = registry.discover(tmp_path)
        assert len(found_second) == 0
        assert len(registry.get_all()) == 1


# ── Schema generation ───────────────────────────────────────────────────────


class TestTypeToSchema:
    """Data-driven tests for the internal ``_type_to_schema`` helper."""

    @pytest.mark.parametrize(
        "annotation, expected_type",
        [
            (str, "string"),
            (int, "integer"),
            (float, "number"),
            (bool, "boolean"),
        ],
        ids=["str", "int", "float", "bool"],
    )
    def test_primitive_types(self, annotation, expected_type):
        schema = _type_to_schema(annotation)
        assert schema["type"] == expected_type

    def test_list_of_str(self):
        schema = _type_to_schema(list[str])
        assert schema["type"] == "array"
        assert schema["items"]["type"] == "string"

    def test_bare_list(self):
        schema = _type_to_schema(list)
        assert schema["type"] == "array"

    def test_bare_dict(self):
        schema = _type_to_schema(dict)
        assert schema["type"] == "object"

    def test_dict_str_any(self):
        schema = _type_to_schema(dict[str, int])
        assert schema["type"] == "object"

    def test_optional_str(self):
        schema = _type_to_schema(Optional[str])
        assert schema["type"] == "string"

    def test_optional_int(self):
        schema = _type_to_schema(Optional[int])
        assert schema["type"] == "integer"

    def test_pydantic_model(self):
        schema = _type_to_schema(SampleModel)
        assert "properties" in schema
        assert "name" in schema["properties"]

    def test_none_type(self):
        schema = _type_to_schema(type(None))
        assert schema["type"] == "null"


class TestIsOptional:
    def test_optional_str_is_optional(self):
        assert _is_optional(Optional[str]) is True

    def test_plain_str_is_not_optional(self):
        assert _is_optional(str) is False

    def test_plain_int_is_not_optional(self):
        assert _is_optional(int) is False


class TestSchemaGeneration:
    def test_basic_schema_generation(self):
        @tool(description="Search", visibility="llm_only")
        async def search(query: str, limit: int = 10) -> str:
            pass

        registry = ToolRegistry()
        registry.register(search)
        schemas = registry.generate_schemas()

        assert len(schemas) == 1
        s = schemas[0]
        assert s["type"] == "function"
        assert s["function"]["name"] == "search"
        assert "Search" in s["function"]["description"]
        params = s["function"]["parameters"]
        assert params["properties"]["query"]["type"] == "string"
        assert params["properties"]["limit"]["type"] == "integer"
        assert "query" in params["required"]

    def test_agent_only_tool_excluded_from_schemas(self):
        registry = ToolRegistry()
        registry.register(_make_sync_tool(name="hidden", visibility="agent_only"))
        schemas = registry.generate_schemas()
        assert len(schemas) == 0

    def test_both_visibility_included_in_schemas(self):
        registry = ToolRegistry()
        registry.register(_make_sync_tool(name="shared", visibility="both"))
        schemas = registry.generate_schemas()
        assert len(schemas) == 1

    def test_pydantic_model_parameter_schema(self):
        @tool(description="Process data", visibility="llm_only")
        async def process(data: SampleModel) -> str:
            return data.name

        registry = ToolRegistry()
        registry.register(process)
        schemas = registry.generate_schemas()
        params = schemas[0]["function"]["parameters"]
        data_schema = params["properties"]["data"]
        # Should contain the pydantic model's JSON schema (has 'properties' key).
        assert "properties" in data_schema

    def test_optional_parameter_not_required(self):
        @tool(description="Maybe filter", visibility="llm_only")
        async def query(text: str, filter_by: Optional[str] = None) -> str:
            return text

        registry = ToolRegistry()
        registry.register(query)
        schemas = registry.generate_schemas()
        params = schemas[0]["function"]["parameters"]
        required = params.get("required", [])
        assert "text" in required
        assert "filter_by" not in required

    def test_docstring_used_in_schema_description(self):
        @tool(description="Main desc", visibility="llm_only")
        async def documented(x: int) -> int:
            """Extended explanation of what this does."""
            return x

        registry = ToolRegistry()
        registry.register(documented)
        schemas = registry.generate_schemas()
        desc = schemas[0]["function"]["description"]
        assert "Main desc" in desc
        assert "Extended explanation" in desc

    def test_multiple_tools_generate_multiple_schemas(self):
        registry = ToolRegistry()
        registry.register(_make_sync_tool(name="t1", visibility="llm_only"))
        registry.register(_make_async_tool(name="t2", visibility="llm_only"))
        registry.register(_make_sync_tool(name="t3", visibility="both"))
        schemas = registry.generate_schemas()
        assert len(schemas) == 3
        names = {s["function"]["name"] for s in schemas}
        assert names == {"t1", "t2", "t3"}


# ── Tool execution ──────────────────────────────────────────────────────────


class TestExecuteTool:
    @pytest.mark.asyncio
    async def test_execute_async_tool(self):
        @tool(description="echo", visibility="both")
        async def echo(msg: str) -> str:
            return f"echo:{msg}"

        registry = ToolRegistry()
        registry.register(echo)
        result = await registry.execute("echo", msg="hello")
        assert result.result == "echo:hello"
        assert result.error is None
        assert result.name == "echo"

    @pytest.mark.asyncio
    async def test_execute_sync_tool(self):
        @tool(description="add", visibility="both")
        def add(a: int, b: int) -> int:
            return a + b

        registry = ToolRegistry()
        registry.register(add)
        result = await registry.execute("add", a=2, b=3)
        assert result.result == "5"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        registry = ToolRegistry()
        result = await registry.execute("does_not_exist")
        assert result.is_error
        assert "Unknown tool" in result.error

    @pytest.mark.asyncio
    async def test_execute_catches_exception(self):
        @tool(description="boom", visibility="both")
        async def explode() -> str:
            raise RuntimeError("kaboom")

        registry = ToolRegistry()
        registry.register(explode)
        result = await registry.execute("explode")
        assert result.is_error
        assert "RuntimeError" in result.error
        assert "kaboom" in result.error

    @pytest.mark.asyncio
    async def test_execute_sync_exception(self):
        @tool(description="sync boom", visibility="both")
        def sync_explode() -> str:
            raise ValueError("bad value")

        registry = ToolRegistry()
        registry.register(sync_explode)
        result = await registry.execute("sync_explode")
        assert result.is_error
        assert "ValueError" in result.error
        assert "bad value" in result.error

    @pytest.mark.asyncio
    async def test_execute_returns_empty_string_for_none(self):
        @tool(description="void", visibility="both")
        async def void_tool() -> None:
            pass

        registry = ToolRegistry()
        registry.register(void_tool)
        result = await registry.execute("void_tool")
        assert result.result == ""
        assert result.error is None

    @pytest.mark.asyncio
    async def test_execute_result_has_call_id(self):
        @tool(description="t", visibility="both")
        async def t() -> str:
            return "ok"

        registry = ToolRegistry()
        registry.register(t)
        result = await registry.execute("t")
        assert len(result.call_id) > 0


# ── ToolMeta.matches_plane ──────────────────────────────────────────────────


class TestMatchesPlane:
    @pytest.mark.parametrize(
        "visibility, plane, expected",
        [
            ("agent_only", "agent_only", True),
            ("agent_only", "llm_only", False),
            ("llm_only", "llm_only", True),
            ("llm_only", "agent_only", False),
            ("both", "agent_only", True),
            ("both", "llm_only", True),
        ],
        ids=[
            "agent-sees-agent_only",
            "llm-cannot-see-agent_only",
            "llm-sees-llm_only",
            "agent-cannot-see-llm_only",
            "agent-sees-both",
            "llm-sees-both",
        ],
    )
    def test_matches_plane(self, visibility, plane, expected):
        meta = ToolMeta(
            name="test",
            description="test",
            visibility=visibility,
            fn=lambda: None,
            is_async=False,
        )
        assert meta.matches_plane(plane) is expected
