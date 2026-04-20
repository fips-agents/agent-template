"""Tests for sandbox.executor."""

import pytest

from sandbox.executor import execute_code


@pytest.mark.asyncio
async def test_happy_path():
    result = await execute_code('print("hello")')
    assert result.stdout == "hello\n", f"unexpected stdout: {result.stdout!r}"
    assert result.exit_code == 0, f"unexpected exit_code: {result.exit_code}"
    assert result.stderr == "", f"unexpected stderr: {result.stderr!r}"
    assert result.timed_out is False


@pytest.mark.asyncio
async def test_math_computation():
    result = await execute_code("import math; print(math.pi)")
    assert result.exit_code == 0, f"unexpected exit_code: {result.exit_code}"
    assert "3.14" in result.stdout, f"unexpected stdout: {result.stdout!r}"


@pytest.mark.asyncio
async def test_stderr_output():
    result = await execute_code(
        'import sys; print("err", file=sys.stderr)',
        runtime_restrict=False,
    )
    assert result.stderr == "err\n", f"unexpected stderr: {result.stderr!r}"
    assert result.exit_code == 0


@pytest.mark.asyncio
async def test_nonzero_exit():
    result = await execute_code("import sys; sys.exit(1)", runtime_restrict=False)
    assert result.exit_code == 1, f"unexpected exit_code: {result.exit_code}"


@pytest.mark.asyncio
async def test_runtime_error():
    result = await execute_code("1/0")
    assert result.exit_code != 0, "expected non-zero exit for ZeroDivisionError"
    assert "ZeroDivisionError" in result.stderr, (
        f"expected ZeroDivisionError in stderr, got: {result.stderr!r}"
    )


@pytest.mark.asyncio
async def test_timeout():
    result = await execute_code("import time; time.sleep(30)", timeout=1.0)
    assert result.timed_out is True, "expected timed_out=True"
    assert result.exit_code != 0, f"expected non-zero exit_code, got: {result.exit_code}"
    assert "timed out" in result.stderr, (
        f"expected timeout message in stderr, got: {result.stderr!r}"
    )


@pytest.mark.asyncio
async def test_empty_code():
    result = await execute_code("")
    assert result.exit_code == 0, f"unexpected exit_code: {result.exit_code}"
    assert result.stdout == ""
    assert result.timed_out is False


@pytest.mark.asyncio
async def test_multiline_code():
    result = await execute_code("x = 5\ny = 10\nprint(x + y)")
    assert result.stdout == "15\n", f"unexpected stdout: {result.stdout!r}"
    assert result.exit_code == 0


@pytest.mark.asyncio
async def test_unicode_output():
    result = await execute_code('print("hello 世界")')
    assert "世界" in result.stdout, f"unexpected stdout: {result.stdout!r}"
    assert result.exit_code == 0
