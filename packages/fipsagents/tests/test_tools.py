"""Tests for fipsagents.baseagent.tools — @tool decorator, ToolRegistry, schema gen."""

from __future__ import annotations

import asyncio
import tempfile
import textwrap
from pathlib import Path
from typing import Optional

import pytest
from pydantic import BaseModel

from fipsagents.baseagent.tools import (
    ToolCall,
    ToolMeta,
    ToolRegistry,
    ToolResult,
    _params_from_signature,
    _type_to_schema,
    tool,
)


# ---------------------------------------------------------------------------
# @tool decorator
# ---------------------------------------------------------------------------


class TestToolDecorator:
    def test_attaches_tool_meta(self):
        @tool(description="Does something", visibility="llm_only")
        async def my_tool(x: str) -> str:
            return x

        meta = getattr(my_tool, "__base_agent_tool__", None)
        assert meta is not None
        assert isinstance(meta, ToolMeta)

    def test_default_name_is_function_name(self):
        @tool(description="Foo", visibility="both")
        def my_func(x: int) -> int:
            return x

        assert my_func.__base_agent_tool__.name == "my_func"

    def test_custom_name_override(self):
        @tool(description="Bar", visibility="agent_only", name="custom_name")
        def original_name() -> None:
            pass

        assert original_name.__base_agent_tool__.name == "custom_name"

    def test_description_stored(self):
        @tool(description="My description", visibility="both")
        def some_tool() -> None:
            pass

        assert "My description" in some_tool.__base_agent_tool__.description

    def test_docstring_appended_to_description(self):
        @tool(description="Short desc", visibility="llm_only")
        def documented_tool() -> None:
            """This is a docstring."""
            pass

        meta = documented_tool.__base_agent_tool__
        assert "Short desc" in meta.description
        assert "This is a docstring." in meta.description

    def test_visibility_set(self):
        @tool(description="x", visibility="agent_only")
        def agent_tool() -> None:
            pass

        assert agent_tool.__base_agent_tool__.visibility == "agent_only"

    def test_is_async_detected(self):
        @tool(description="async", visibility="both")
        async def async_tool() -> str:
            return "x"

        assert async_tool.__base_agent_tool__.is_async is True

    def test_is_sync_detected(self):
        @tool(description="sync", visibility="both")
        def sync_tool() -> str:
            return "x"

        assert sync_tool.__base_agent_tool__.is_async is False

    def test_invalid_visibility_raises(self):
        with pytest.raises(ValueError, match="visibility"):
            @tool(description="x", visibility="invalid_plane")  # type: ignore[arg-type]
            def bad_tool() -> None:
                pass


# ---------------------------------------------------------------------------
# ToolMeta.matches_plane
# ---------------------------------------------------------------------------


class TestToolMetaMatchesPlane:
    def _make_meta(self, visibility: str) -> ToolMeta:
        return ToolMeta(
            name="t",
            description="d",
            visibility=visibility,  # type: ignore[arg-type]
            fn=lambda: None,
            is_async=False,
        )

    def test_both_matches_llm_only(self):
        meta = self._make_meta("both")
        assert meta.matches_plane("llm_only") is True

    def test_both_matches_agent_only(self):
        meta = self._make_meta("both")
        assert meta.matches_plane("agent_only") is True

    def test_llm_only_matches_llm_only(self):
        meta = self._make_meta("llm_only")
        assert meta.matches_plane("llm_only") is True

    def test_llm_only_does_not_match_agent_only(self):
        meta = self._make_meta("llm_only")
        assert meta.matches_plane("agent_only") is False

    def test_agent_only_matches_agent_only(self):
        meta = self._make_meta("agent_only")
        assert meta.matches_plane("agent_only") is True

    def test_agent_only_does_not_match_llm_only(self):
        meta = self._make_meta("agent_only")
        assert meta.matches_plane("llm_only") is False


# ---------------------------------------------------------------------------
# ToolCall and ToolResult models
# ---------------------------------------------------------------------------


class TestToolCall:
    def test_default_call_id_generated(self):
        call = ToolCall(name="my_tool")
        assert call.call_id
        assert len(call.call_id) > 0

    def test_call_id_unique(self):
        c1 = ToolCall(name="t")
        c2 = ToolCall(name="t")
        assert c1.call_id != c2.call_id

    def test_arguments_default_empty(self):
        call = ToolCall(name="t")
        assert call.arguments == {}

    def test_custom_arguments(self):
        call = ToolCall(name="t", arguments={"x": 1})
        assert call.arguments == {"x": 1}


class TestToolResult:
    def test_is_error_false_when_no_error(self):
        result = ToolResult(call_id="abc", name="t", result="ok")
        assert result.is_error is False

    def test_is_error_true_when_error_set(self):
        result = ToolResult(call_id="abc", name="t", error="oops")
        assert result.is_error is True

    def test_result_defaults_to_empty_string(self):
        result = ToolResult(call_id="abc", name="t")
        assert result.result == ""


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class TestToolRegistryRegister:
    def test_registers_decorated_function(self):
        registry = ToolRegistry()

        @tool(description="x", visibility="both")
        def my_fn() -> None:
            pass

        meta = registry.register(my_fn)
        assert meta.name == "my_fn"

    def test_rejects_undecorated_function(self):
        registry = ToolRegistry()

        def plain_fn() -> None:
            pass

        with pytest.raises(ValueError, match="@tool"):
            registry.register(plain_fn)

    def test_rejects_duplicate_name(self):
        registry = ToolRegistry()

        @tool(description="x", visibility="both")
        def dup_tool() -> None:
            pass

        registry.register(dup_tool)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(dup_tool)


class TestToolRegistryFiltering:
    def setup_method(self):
        self.registry = ToolRegistry()

        @tool(description="llm", visibility="llm_only")
        def llm_fn() -> None:
            pass

        @tool(description="agent", visibility="agent_only")
        def agent_fn() -> None:
            pass

        @tool(description="both", visibility="both")
        def both_fn() -> None:
            pass

        self.registry.register(llm_fn)
        self.registry.register(agent_fn)
        self.registry.register(both_fn)

    def test_get_llm_tools_excludes_agent_only(self):
        names = {t.name for t in self.registry.get_llm_tools()}
        assert "agent_fn" not in names
        assert "llm_fn" in names
        assert "both_fn" in names

    def test_get_agent_tools_excludes_llm_only(self):
        names = {t.name for t in self.registry.get_agent_tools()}
        assert "llm_fn" not in names
        assert "agent_fn" in names
        assert "both_fn" in names

    def test_get_all_returns_all(self):
        names = {t.name for t in self.registry.get_all()}
        assert names == {"llm_fn", "agent_fn", "both_fn"}


class TestToolRegistryGenerateSchemas:
    def test_schemas_are_openai_format(self):
        registry = ToolRegistry()

        @tool(description="Search the web", visibility="llm_only")
        def web_search(query: str) -> str:
            return ""

        registry.register(web_search)
        schemas = registry.generate_schemas()
        assert len(schemas) == 1
        schema = schemas[0]
        assert schema["type"] == "function"
        assert "function" in schema
        assert schema["function"]["name"] == "web_search"

    def test_agent_only_tools_excluded_from_schemas(self):
        registry = ToolRegistry()

        @tool(description="Internal", visibility="agent_only")
        def internal_fn() -> None:
            pass

        registry.register(internal_fn)
        schemas = registry.generate_schemas()
        assert schemas == []


class TestToolRegistryExecute:
    @pytest.mark.asyncio
    async def test_executes_async_tool(self):
        registry = ToolRegistry()

        @tool(description="x", visibility="both")
        async def adder(a: int, b: int) -> int:
            return a + b

        registry.register(adder)
        result = await registry.execute("adder", a=2, b=3)
        assert result.is_error is False
        assert result.result == "5"

    @pytest.mark.asyncio
    async def test_executes_sync_tool(self):
        registry = ToolRegistry()

        @tool(description="x", visibility="both")
        def multiplier(x: int, y: int) -> int:
            return x * y

        registry.register(multiplier)
        result = await registry.execute("multiplier", x=3, y=4)
        assert result.is_error is False
        assert result.result == "12"

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        registry = ToolRegistry()
        result = await registry.execute("no_such_tool")
        assert result.is_error is True
        assert "no_such_tool" in result.error

    @pytest.mark.asyncio
    async def test_exception_in_tool_returns_error(self):
        registry = ToolRegistry()

        @tool(description="x", visibility="both")
        async def exploding_tool() -> str:
            raise ValueError("boom")

        registry.register(exploding_tool)
        result = await registry.execute("exploding_tool")
        assert result.is_error is True
        assert "ValueError" in result.error
        assert "boom" in result.error


class TestToolRegistryDiscover:
    def test_discovers_tools_from_directory(self, tmp_path):
        tool_file = tmp_path / "my_tools.py"
        tool_file.write_text(
            textwrap.dedent("""\
                from fipsagents.baseagent.tools import tool

                @tool(description="Greet", visibility="both")
                async def greet(name: str) -> str:
                    return f"Hello, {name}"
            """)
        )
        registry = ToolRegistry()
        discovered = registry.discover(tmp_path)
        assert len(discovered) == 1
        assert discovered[0].name == "greet"

    def test_skips_underscore_files(self, tmp_path):
        (tmp_path / "_private.py").write_text("x = 1")
        registry = ToolRegistry()
        discovered = registry.discover(tmp_path)
        assert discovered == []

    def test_nonexistent_directory_returns_empty(self):
        registry = ToolRegistry()
        result = registry.discover("/nonexistent/path/that/does/not/exist")
        assert result == []


# ---------------------------------------------------------------------------
# _type_to_schema
# ---------------------------------------------------------------------------


class TestTypeToSchema:
    @pytest.mark.parametrize(
        "annotation, expected_type",
        [
            (str, "string"),
            (int, "integer"),
            (float, "number"),
            (bool, "boolean"),
        ],
    )
    def test_primitive_types(self, annotation, expected_type):
        assert _type_to_schema(annotation) == {"type": expected_type}

    def test_list_of_str(self):
        from typing import List
        schema = _type_to_schema(List[str])
        assert schema == {"type": "array", "items": {"type": "string"}}

    def test_bare_list(self):
        assert _type_to_schema(list) == {"type": "array"}

    def test_dict(self):
        assert _type_to_schema(dict) == {"type": "object"}

    def test_optional_int(self):
        schema = _type_to_schema(Optional[int])
        assert schema == {"type": "integer"}

    def test_pydantic_model(self):
        class MyModel(BaseModel):
            x: int

        schema = _type_to_schema(MyModel)
        assert "properties" in schema or "$defs" in schema or "x" in str(schema)

    def test_none_type(self):
        assert _type_to_schema(type(None)) == {"type": "null"}


# ---------------------------------------------------------------------------
# _params_from_signature
# ---------------------------------------------------------------------------


class TestParamsFromSignature:
    def test_required_param(self):
        def fn(x: str) -> None:
            pass

        schema = _params_from_signature(fn)
        assert "x" in schema["properties"]
        assert "x" in schema["required"]

    def test_optional_param_not_required(self):
        def fn(x: Optional[str] = None) -> None:
            pass

        schema = _params_from_signature(fn)
        assert "x" in schema["properties"]
        assert "required" not in schema or "x" not in schema.get("required", [])

    def test_param_with_default_not_required(self):
        def fn(x: str = "default") -> None:
            pass

        schema = _params_from_signature(fn)
        assert "required" not in schema or "x" not in schema.get("required", [])

    def test_skips_self(self):
        class MyClass:
            def method(self, x: str) -> None:
                pass

        schema = _params_from_signature(MyClass.method)
        assert "self" not in schema["properties"]

    def test_unannotated_param_gets_empty_schema(self):
        def fn(x) -> None:
            pass

        schema = _params_from_signature(fn)
        assert "x" in schema["properties"]
        assert schema["properties"]["x"] == {}
