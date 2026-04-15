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


# ---------------------------------------------------------------------------
# Credential / secret detection — expect at least one violation
# ---------------------------------------------------------------------------

CREDENTIAL_DETECTION_CASES = [
    pytest.param(
        'x = "AKIAIOSFODNN7EXAMPLE"',
        id="aws_access_key",
    ),
    pytest.param(
        'key = "-----BEGIN RSA PRIVATE KEY-----\\nMIIEpAIBAAKCAQEA"',
        id="pem_private_key",
    ),
    pytest.param(
        'key = "-----BEGIN PRIVATE KEY-----\\nMIIEpAIBAAKCAQEA"',
        id="pem_private_key_generic",
    ),
    pytest.param(
        r'''config = "api_key = 'sk-abcdefghijklmnop1234567890'"''',
        id="generic_api_key_in_string",
    ),
    pytest.param(
        'x = "aabbccdd11223344aabbccdd11223344"',
        id="high_entropy_hex_32",
    ),
]

@pytest.mark.parametrize("source", CREDENTIAL_DETECTION_CASES)
def test_credential_detected(source):
    violations = validate_code(source)
    assert len(violations) >= 1, f"Expected a credential violation for: {source!r}"
    assert any("string literal matches" in v for v in violations), violations


CREDENTIAL_SAFE_CASES = [
    pytest.param('x = "hello"', id="short_string"),
    pytest.param('msg = "This is a test"', id="normal_string"),
    pytest.param("y = 42", id="integer_constant"),
    pytest.param('x = "deadbeef" * 10', id="short_hex_no_match"),
]


@pytest.mark.parametrize("source", CREDENTIAL_SAFE_CASES)
def test_credential_safe(source):
    violations = validate_code(source)
    cred_violations = [v for v in violations if "string literal matches" in v]
    assert cred_violations == [], f"False positive credential detection: {cred_violations}"


# ---------------------------------------------------------------------------
# Unsafe deserialization — expect at least one violation
# ---------------------------------------------------------------------------

UNSAFE_DESER_CASES = [
    pytest.param("pickle.loads(data)", id="pickle_loads"),
    pytest.param("pickle.load(f)", id="pickle_load"),
    pytest.param("pickle.Unpickler(f)", id="pickle_unpickler"),
    pytest.param("yaml.unsafe_load(data)", id="yaml_unsafe_load"),
    pytest.param("yaml.load(data)", id="yaml_load"),
    pytest.param("marshal.loads(data)", id="marshal_loads"),
    pytest.param("marshal.load(f)", id="marshal_load"),
    pytest.param("shelve.open('db')", id="shelve_open"),
]


@pytest.mark.parametrize("source", UNSAFE_DESER_CASES)
def test_unsafe_deser(source):
    violations = validate_code(source)
    assert len(violations) >= 1, f"Expected a violation for: {source!r}"
    assert any("unsafe deserialization" in v for v in violations), violations


# ---------------------------------------------------------------------------
# SQL injection — expect at least one violation
# ---------------------------------------------------------------------------

SQL_INJECTION_CASES = [
    pytest.param(
        'query = f"SELECT * FROM users WHERE id = {user_id}"',
        id="fstring_select",
    ),
    pytest.param(
        'query = f"DELETE FROM sessions WHERE token = {tok}"',
        id="fstring_delete",
    ),
    pytest.param(
        'query = "SELECT * FROM users WHERE id = %s" % user_id',
        id="percent_format_select",
    ),
    pytest.param(
        'query = "INSERT INTO logs VALUES (%s)" % val',
        id="percent_format_insert",
    ),
    pytest.param(
        '"SELECT * FROM users WHERE id = {}".format(user_id)',
        id="dot_format_select",
    ),
]


@pytest.mark.parametrize("source", SQL_INJECTION_CASES)
def test_sql_injection(source):
    violations = validate_code(source)
    assert len(violations) >= 1, f"Expected a SQL injection violation for: {source!r}"
    assert any("SQL injection" in v for v in violations), violations


SQL_SAFE_CASES = [
    pytest.param('x = "SELECT"', id="bare_keyword_short"),
    pytest.param('query = "SELECT * FROM users"', id="static_query_no_interp"),
]


@pytest.mark.parametrize("source", SQL_SAFE_CASES)
def test_sql_safe(source):
    violations = validate_code(source)
    sql_violations = [v for v in violations if "SQL injection" in v]
    assert sql_violations == [], f"False positive SQL detection: {sql_violations}"


# ---------------------------------------------------------------------------
# Weak cryptography — expect at least one violation
# ---------------------------------------------------------------------------

WEAK_CRYPTO_CASES = [
    pytest.param("hashlib.md5(data)", id="md5"),
    pytest.param("hashlib.sha1(data)", id="sha1"),
]


@pytest.mark.parametrize("source", WEAK_CRYPTO_CASES)
def test_weak_crypto(source):
    violations = validate_code(source)
    assert len(violations) >= 1, f"Expected a weak crypto violation for: {source!r}"
    assert any("weak cryptography" in v for v in violations), violations


# ---------------------------------------------------------------------------
# Path traversal — expect at least one violation
# ---------------------------------------------------------------------------

PATH_TRAVERSAL_CASES = [
    pytest.param('path = "../../etc/passwd"', id="double_traversal"),
    pytest.param('f = "../../../secret.txt"', id="triple_traversal"),
]


@pytest.mark.parametrize("source", PATH_TRAVERSAL_CASES)
def test_path_traversal(source):
    violations = validate_code(source)
    assert len(violations) >= 1, f"Expected a path traversal violation for: {source!r}"
    assert any("path traversal" in v for v in violations), violations


PATH_TRAVERSAL_SAFE_CASES = [
    pytest.param('x = "hello"', id="no_traversal"),
    pytest.param('x = "a/b/c"', id="normal_path"),
    pytest.param('x = "../short"', id="single_traversal_short_string"),
]


@pytest.mark.parametrize("source", PATH_TRAVERSAL_SAFE_CASES)
def test_path_traversal_safe(source):
    violations = validate_code(source)
    traversal_violations = [v for v in violations if "path traversal" in v]
    assert traversal_violations == [], f"False positive path traversal: {traversal_violations}"


# ---------------------------------------------------------------------------
# allowed_imports parameter — profile-aware import checking
# ---------------------------------------------------------------------------


def test_custom_allowed_imports_permits_numpy():
    # numpy is not in the default (minimal) allowlist, but passing it explicitly
    # should suppress the import violation.
    violations = validate_code(
        "import numpy",
        allowed_imports=frozenset({"math", "numpy"}),
    )
    assert violations == [], (
        f"numpy should be allowed with custom allowed_imports, got: {violations}"
    )


def test_default_allowed_imports_blocks_numpy():
    # Without an explicit allowed_imports argument the minimal allowlist applies,
    # so numpy must still be rejected.
    violations = validate_code("import numpy")
    assert any("numpy" in v for v in violations), (
        f"expected numpy to be blocked by default allowlist, got: {violations}"
    )
