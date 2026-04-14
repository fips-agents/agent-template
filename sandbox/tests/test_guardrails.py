"""Tests for sandbox.guardrails.validate_code."""

import pytest

from sandbox.guardrails import validate_code


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean(source: str) -> list[str]:
    """Strip leading indentation from triple-quoted test snippets."""
    import textwrap

    return validate_code(textwrap.dedent(source).strip())


# ---------------------------------------------------------------------------
# Allowed code — expect no violations
# ---------------------------------------------------------------------------

ALLOWED_CASES = [
    pytest.param("import math; print(math.sqrt(4))", id="import_math"),
    pytest.param(
        "from collections import Counter; print(Counter([1, 1, 2]))",
        id="from_collections",
    ),
    pytest.param('import json; json.dumps({"a": 1})', id="import_json"),
    pytest.param("x = [i**2 for i in range(10)]", id="list_comprehension"),
    pytest.param("import math, statistics", id="multi_import"),
    pytest.param("from datetime import datetime", id="from_datetime"),
    pytest.param(
        "from typing import Optional, List\ndef foo(x: Optional[int]) -> List[int]: ...",
        id="typing_annotations",
    ),
    pytest.param(
        "import functools\n@functools.lru_cache()\ndef fib(n): return n if n < 2 else fib(n-1)+fib(n-2)",
        id="functools_decorator",
    ),
]


@pytest.mark.parametrize("source", ALLOWED_CASES)
def test_allowed(source):
    violations = validate_code(source)
    assert violations == [], f"Expected no violations but got: {violations}"


# ---------------------------------------------------------------------------
# Blocked imports — expect at least one violation
# ---------------------------------------------------------------------------

BLOCKED_IMPORT_CASES = [
    pytest.param("import os", id="import_os"),
    pytest.param("import subprocess", id="import_subprocess"),
    pytest.param("import socket", id="import_socket"),
    pytest.param("from os import path", id="from_os"),
    pytest.param("import importlib", id="import_importlib"),
    pytest.param("import urllib", id="import_urllib"),
    pytest.param("import http", id="import_http"),
    pytest.param("import requests", id="import_requests"),
    pytest.param("import sys", id="import_sys"),
    pytest.param("import builtins", id="import_builtins"),
    pytest.param("from os import *", id="star_import_os"),
]


@pytest.mark.parametrize("source", BLOCKED_IMPORT_CASES)
def test_blocked_import(source):
    violations = validate_code(source)
    assert len(violations) >= 1, f"Expected a violation for: {source!r}"
    # The violation message should mention the import.
    assert any("import" in v for v in violations), violations


# ---------------------------------------------------------------------------
# Blocked calls — expect at least one violation
# ---------------------------------------------------------------------------

BLOCKED_CALL_CASES = [
    pytest.param('eval("1+1")', id="eval"),
    pytest.param('exec("x=1")', id="exec"),
    pytest.param('compile("x", "", "exec")', id="compile"),
    pytest.param('__import__("os")', id="dunder_import"),
    pytest.param('open("file.txt")', id="open_read"),
    pytest.param('open("file.txt", "w")', id="open_write"),
    pytest.param('getattr(type, "__subclasses__")()', id="getattr_bypass"),
    pytest.param('setattr(obj, "x", 1)', id="setattr"),
    pytest.param('delattr(obj, "x")', id="delattr"),
    pytest.param("breakpoint()", id="breakpoint"),
    pytest.param("input('prompt')", id="input"),
]


@pytest.mark.parametrize("source", BLOCKED_CALL_CASES)
def test_blocked_call(source):
    violations = validate_code(source)
    assert len(violations) >= 1, f"Expected a violation for: {source!r}"


# ---------------------------------------------------------------------------
# Blocked attribute / module patterns — expect at least one violation
# ---------------------------------------------------------------------------

BLOCKED_PATTERN_CASES = [
    pytest.param('import os; os.system("ls")', id="os_system"),
    pytest.param('import os; os.popen("ls")', id="os_popen"),
    pytest.param('import subprocess; subprocess.run(["ls"])', id="subprocess_run"),
    pytest.param("().__class__.__subclasses__()", id="subclasses"),
    pytest.param("x.__globals__", id="dunder_globals"),
    pytest.param("x.__builtins__", id="dunder_builtins"),
]


@pytest.mark.parametrize("source", BLOCKED_PATTERN_CASES)
def test_blocked_pattern(source):
    violations = validate_code(source)
    assert len(violations) >= 1, f"Expected a violation for: {source!r}"


# ---------------------------------------------------------------------------
# Syntax errors — expect exactly one parse-error violation
# ---------------------------------------------------------------------------


def test_syntax_error():
    violations = validate_code("def foo(")
    assert len(violations) == 1
    assert "SyntaxError" in violations[0]


# ---------------------------------------------------------------------------
# Multiple violations — all returned in one pass
# ---------------------------------------------------------------------------


def test_multiple_violations():
    source = 'import os; eval("1"); exec("2")'
    violations = validate_code(source)
    # 3 violations: os import, eval call, exec call
    assert len(violations) >= 3, f"Expected >=3 violations, got: {violations}"

    text = "\n".join(violations)
    assert "os" in text
    assert "eval" in text
    assert "exec" in text


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_source():
    assert validate_code("") == []


def test_allowed_import_submodule_blocked():
    # datetime.timezone is fine — still the datetime top-level package
    violations = validate_code("from datetime import timezone")
    assert violations == []


def test_subprocess_attr_access_blocked():
    # Accessing subprocess.PIPE without a call still involves an import,
    # so the import violation is caught.
    violations = validate_code("import subprocess; x = subprocess.PIPE")
    assert any("subprocess" in v for v in violations)


def test_dunder_subclasses_in_call():
    # Access via method call — __subclasses__ dunder must be caught.
    violations = validate_code("type.__subclasses__(type)")
    assert any("__subclasses__" in v for v in violations)
