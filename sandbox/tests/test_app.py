"""Tests for sandbox.app — the FastAPI HTTP layer."""

import pytest
from httpx import ASGITransport, AsyncClient

from sandbox.app import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# /healthz
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_healthz(client: AsyncClient):
    resp = await client.get("/healthz")
    assert resp.status_code == 200, f"unexpected status: {resp.status_code}"
    assert resp.json() == {"status": "ok"}, f"unexpected body: {resp.json()}"


# ---------------------------------------------------------------------------
# /execute — happy paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_valid_code(client: AsyncClient):
    resp = await client.post("/execute", json={"code": "print(42)"})
    assert resp.status_code == 200, f"unexpected status: {resp.status_code}, body: {resp.text}"
    body = resp.json()
    assert body["stdout"] == "42\n", f"unexpected stdout: {body['stdout']!r}"
    assert body["exit_code"] == 0, f"unexpected exit_code: {body['exit_code']}"
    assert body["timed_out"] is False


@pytest.mark.asyncio
async def test_custom_timeout(client: AsyncClient):
    resp = await client.post("/execute", json={"code": "print(1)", "timeout": 5})
    assert resp.status_code == 200, f"unexpected status: {resp.status_code}, body: {resp.text}"
    body = resp.json()
    assert body["exit_code"] == 0
    assert body["timed_out"] is False


@pytest.mark.asyncio
async def test_runtime_error(client: AsyncClient):
    resp = await client.post("/execute", json={"code": "1/0"})
    assert resp.status_code == 200, f"unexpected status: {resp.status_code}, body: {resp.text}"
    body = resp.json()
    assert body["exit_code"] != 0, f"expected non-zero exit_code, got: {body['exit_code']}"
    assert "ZeroDivisionError" in body["stderr"], (
        f"expected ZeroDivisionError in stderr: {body['stderr']!r}"
    )


@pytest.mark.asyncio
async def test_timeout_execution(client: AsyncClient):
    # Busy-loop using only allowed builtins so guardrails pass, but execution hits the timeout.
    busy_loop = "x = 0\nwhile True:\n    x += 1"
    resp = await client.post(
        "/execute",
        json={"code": busy_loop, "timeout": 1},
        timeout=10.0,
    )
    assert resp.status_code == 200, f"unexpected status: {resp.status_code}, body: {resp.text}"
    body = resp.json()
    assert body["timed_out"] is True, f"expected timed_out=True, got: {body['timed_out']}"


# ---------------------------------------------------------------------------
# /execute — validation failures (400)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_code(client: AsyncClient):
    resp = await client.post("/execute", json={"code": ""})
    assert resp.status_code == 400, f"unexpected status: {resp.status_code}"
    body = resp.json()
    assert "error" in body, f"expected 'error' key, got: {body}"
    assert body["error"] == "No code provided", f"unexpected error: {body['error']!r}"


@pytest.mark.asyncio
async def test_blocked_import(client: AsyncClient):
    resp = await client.post("/execute", json={"code": "import os"})
    assert resp.status_code == 400, f"unexpected status: {resp.status_code}"
    body = resp.json()
    assert body.get("error") == "Code validation failed", f"unexpected body: {body}"
    assert isinstance(body.get("violations"), list), f"expected violations list: {body}"
    assert len(body["violations"]) > 0, "violations list should not be empty"


@pytest.mark.asyncio
async def test_syntax_error(client: AsyncClient):
    resp = await client.post("/execute", json={"code": "def foo("})
    assert resp.status_code == 400, f"unexpected status: {resp.status_code}"
    body = resp.json()
    assert body.get("error") == "Code validation failed", f"unexpected body: {body}"
    assert isinstance(body.get("violations"), list), f"expected violations list: {body}"
    assert any("SyntaxError" in v for v in body["violations"]), (
        f"expected SyntaxError in violations: {body['violations']}"
    )


# ---------------------------------------------------------------------------
# /execute — Pydantic validation failures (422)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeout_too_large(client: AsyncClient):
    resp = await client.post("/execute", json={"code": "x=1", "timeout": 60})
    assert resp.status_code == 422, (
        f"expected 422 for timeout > 30, got {resp.status_code}: {resp.text}"
    )


@pytest.mark.asyncio
async def test_negative_timeout(client: AsyncClient):
    resp = await client.post("/execute", json={"code": "x=1", "timeout": -1})
    assert resp.status_code == 422, (
        f"expected 422 for negative timeout, got {resp.status_code}: {resp.text}"
    )
