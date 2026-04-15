"""Code execution tool — sends Python code to the sandbox sidecar."""

import os

import httpx

from fipsagents.baseagent.tools import tool

SANDBOX_URL = os.environ.get("SANDBOX_URL", "http://localhost:8000")


@tool(
    description="Execute Python code in an isolated sandbox and return the output. "
    "Use this for any computation, math, data processing, or logic that "
    "benefits from exact results. The code runs in a restricted environment "
    "with access to: math, statistics, itertools, functools, re, datetime, "
    "collections, json, csv, string, textwrap, decimal, fractions, random, "
    "operator, typing. Use print() to produce output.",
    visibility="llm_only",
)
async def code_executor(code: str, timeout: float = 10.0) -> str:
    """Execute Python code in the sandbox sidecar.

    Args:
        code: Python source code to execute. Must use print() for output.
        timeout: Maximum execution time in seconds (1-30).
    """
    timeout = max(1.0, min(timeout, 30.0))

    async with httpx.AsyncClient(timeout=timeout + 5) as client:
        try:
            resp = await client.post(
                f"{SANDBOX_URL}/execute",
                json={"code": code, "timeout": timeout},
            )
        except httpx.ConnectError:
            return (
                "ERROR: Cannot connect to sandbox sidecar at "
                f"{SANDBOX_URL}. Is it running?"
            )
        except httpx.TimeoutException:
            return "ERROR: Request to sandbox timed out."

    data = resp.json()

    if resp.status_code == 400:
        if "violations" in data:
            violations = "\n".join(f"  - {v}" for v in data["violations"])
            return f"CODE BLOCKED by sandbox guardrails:\n{violations}"
        return f"ERROR: {data.get('error', 'Unknown error')}"

    stdout = data.get("stdout", "").strip()
    stderr = data.get("stderr", "").strip()
    exit_code = data.get("exit_code", -1)
    timed_out = data.get("timed_out", False)

    if timed_out:
        return f"TIMEOUT: Code exceeded {timeout}s limit.\nPartial output:\n{stdout}"

    parts = []
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(f"STDERR:\n{stderr}")
    if exit_code != 0:
        parts.append(f"(exit code {exit_code})")

    return "\n".join(parts) if parts else "(no output — did you forget print()?)"
