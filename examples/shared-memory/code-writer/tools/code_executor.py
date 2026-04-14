"""Code executor tool — LLM-callable (plane 2).

Sends Python code to the sandbox sidecar for safe execution. The sidecar
validates the code with AST-based guardrails (import allowlist, blocked
patterns) and runs it in an isolated subprocess with a timeout.

Available modules: math, statistics, itertools, functools, re, datetime,
collections, json, csv, string, textwrap, decimal, fractions, random,
operator, typing.

The sidecar URL defaults to http://localhost:8000 (pod-local sidecar).
Override by setting the SANDBOX_URL environment variable.
"""

import os

import httpx

from fipsagents.baseagent.tools import tool

_AVAILABLE_MODULES = (
    "math, statistics, itertools, functools, re, datetime, collections, "
    "json, csv, string, textwrap, decimal, fractions, random, operator, typing"
)


@tool(
    description=(
        "Execute Python code in a secure sandbox. "
        f"Available modules: {_AVAILABLE_MODULES}. "
        "No network access, no filesystem writes, no subprocess calls. "
        "Use for calculations, data transformations, and logic that benefits "
        "from code rather than prose."
    ),
    visibility="llm_only",
)
async def code_executor(code: str, timeout: float = 10.0) -> str:
    """Execute Python code in the sandbox sidecar and return the output.

    Args:
        code: The Python code to execute. Only standard-library modules from
            the allowlist are available. Print to stdout to return results.
        timeout: Maximum execution time in seconds. Must be between 1 and 30.
            Defaults to 10.0.

    Returns:
        Formatted string containing stdout, stderr (if non-empty), and exit
        code (if non-zero), or an error description if execution failed.
    """
    timeout = max(1.0, min(30.0, timeout))
    sandbox_url = os.environ.get("SANDBOX_URL", "http://localhost:8000")

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{sandbox_url}/execute",
                json={"code": code, "timeout": timeout},
                timeout=timeout + 5,
            )
    except httpx.ConnectError:
        return (
            "Error: sandbox sidecar is unavailable. "
            f"Could not connect to {sandbox_url}. "
            "Ensure the sandbox container is running alongside this agent."
        )
    except httpx.HTTPError as exc:
        return f"Error: HTTP request to sandbox failed: {exc}"

    try:
        data = resp.json()
    except Exception:
        return f"Error: sandbox returned status {resp.status_code}: {resp.text[:500]}"

    if not resp.is_success:
        error = data.get("error", resp.text)
        violations = data.get("violations", [])
        if violations:
            violation_lines = "\n".join(f"  - {v}" for v in violations)
            return f"Validation error: {error}\nViolations:\n{violation_lines}"
        return f"Error: {error}"

    parts = []
    stdout = data.get("stdout", "").rstrip()
    stderr = data.get("stderr", "").rstrip()
    exit_code = data.get("exit_code", 0)

    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(f"stderr:\n{stderr}")
    if exit_code != 0:
        parts.append(f"exit code: {exit_code}")
    if not parts:
        parts.append("(no output)")

    return "\n".join(parts)
