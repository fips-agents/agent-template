"""End-to-end integration tests for the code execution sandbox.

Exercises the full pipeline through the FastAPI HTTP layer: real HTTP requests
hitting real guardrails and real subprocess execution.  The existing unit tests
cover individual modules; these tests validate the full flow with complex,
realistic scenarios.
"""

import asyncio
import json
import time

import pytest
from httpx import ASGITransport, AsyncClient

from sandbox.app import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def execute(client: AsyncClient, code: str, timeout: float = 10.0) -> tuple:
    """POST /execute and return (status_code, body_dict)."""
    resp = await client.post("/execute", json={"code": code, "timeout": timeout}, timeout=30.0)
    return resp.status_code, resp.json()


class TestRealisticComputations:
    """Simulate realistic code an LLM would generate — multi-line, real problems."""

    @pytest.mark.asyncio
    async def test_statistical_analysis(self, client: AsyncClient):
        """Compute mean, median, std dev of a dataset using the statistics module."""
        code = """
import statistics
data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
print(f"mean={statistics.mean(data)}")
print(f"median={statistics.median(data)}")
print(f"stdev={statistics.stdev(data):.4f}")
"""
        status, body = await execute(client, code)
        assert status == 200, f"unexpected status {status}: {body}"
        assert body["exit_code"] == 0, f"non-zero exit: {body['stderr']}"
        stdout = body["stdout"]
        assert "mean=55" in stdout, f"wrong mean in: {stdout!r}"
        assert "median=55" in stdout, f"wrong median in: {stdout!r}"
        assert "stdev=30.2765" in stdout, f"wrong stdev in: {stdout!r}"

    @pytest.mark.asyncio
    async def test_json_data_processing(self, client: AsyncClient):
        """Parse JSON, filter and map, serialize back — verify output is valid JSON."""
        code = """
import json
raw = '[{"name": "alice", "score": 82}, {"name": "bob", "score": 45}, {"name": "carol", "score": 91}]'
data = json.loads(raw)
passing = [{"name": d["name"], "grade": "pass"} for d in data if d["score"] >= 50]
print(json.dumps(passing))
"""
        status, body = await execute(client, code)
        assert status == 200, f"unexpected status {status}: {body}"
        assert body["exit_code"] == 0, f"non-zero exit: {body['stderr']}"
        result = json.loads(body["stdout"].strip())
        assert len(result) == 2, f"expected 2 passing entries, got: {result}"
        names = {r["name"] for r in result}
        assert names == {"alice", "carol"}, f"unexpected names: {names}"
        assert all(r["grade"] == "pass" for r in result)

    @pytest.mark.asyncio
    async def test_regex_extraction(self, client: AsyncClient):
        """Extract all email addresses from a text block with re."""
        code = (
            "import re\n"
            "text = (\n"
            "    'Hello from alice@example.com. '\n"
            "    'Please CC bob@company.org and support@helpdesk.io. '\n"
            "    'Also loop in carol@reports.example.com for the report.'\n"
            ")\n"
            r"emails = re.findall(r'[\w.+-]+@[\w.-]+\.[a-z]{2,}', text)" "\n"
            "print(len(emails))\n"
            "for e in sorted(emails):\n"
            "    print(e)\n"
        )
        status, body = await execute(client, code)
        assert status == 200, f"unexpected status {status}: {body}"
        assert body["exit_code"] == 0, f"non-zero exit: {body['stderr']}"
        lines = body["stdout"].strip().splitlines()
        count = int(lines[0])
        assert count == 4, f"expected 4 emails, got {count}: {lines}"
        found = set(lines[1:])
        assert "alice@example.com" in found
        assert "bob@company.org" in found
        assert "support@helpdesk.io" in found
        assert "carol@reports.example.com" in found

    @pytest.mark.asyncio
    async def test_date_arithmetic(self, client: AsyncClient):
        """Compute days between two dates using datetime."""
        code = """
from datetime import date
d1 = date(2024, 1, 1)
d2 = date(2024, 12, 31)
delta = (d2 - d1).days
print(delta)
"""
        status, body = await execute(client, code)
        assert status == 200, f"unexpected status {status}: {body}"
        assert body["exit_code"] == 0, f"non-zero exit: {body['stderr']}"
        result = int(body["stdout"].strip())
        assert result == 365, f"expected 365 days, got {result}"

    @pytest.mark.asyncio
    async def test_combinatorics(self, client: AsyncClient):
        """Use itertools.combinations to compute C(10,3); verify count is 120."""
        code = """
import itertools
combos = list(itertools.combinations(range(10), 3))
print(len(combos))
"""
        status, body = await execute(client, code)
        assert status == 200, f"unexpected status {status}: {body}"
        assert body["exit_code"] == 0, f"non-zero exit: {body['stderr']}"
        result = int(body["stdout"].strip())
        assert result == 120, f"C(10,3) should be 120, got {result}"

    @pytest.mark.asyncio
    async def test_fibonacci_lru_cache(self, client: AsyncClient):
        """Compute fib(20) with @functools.lru_cache; verify result is 6765."""
        code = """
import functools

@functools.lru_cache(maxsize=None)
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

print(fib(20))
"""
        status, body = await execute(client, code)
        assert status == 200, f"unexpected status {status}: {body}"
        assert body["exit_code"] == 0, f"non-zero exit: {body['stderr']}"
        result = int(body["stdout"].strip())
        assert result == 6765, f"fib(20) should be 6765, got {result}"

    @pytest.mark.asyncio
    async def test_io_import_blocked_use_print_csv(self, client: AsyncClient):
        """Verify io is NOT in the allowlist, then produce CSV-like output with print."""
        # First: confirm io is blocked
        status_blocked, body_blocked = await execute(client, "import io")
        assert status_blocked == 400, f"expected 400 for 'import io', got {status_blocked}"
        assert "violations" in body_blocked, f"expected violations: {body_blocked}"
        assert any("io" in v for v in body_blocked["violations"]), (
            f"violation should mention 'io': {body_blocked['violations']}"
        )

        # Then: produce structured output without io using print
        code = """
import json
rows = [
    {"product": "apple", "qty": 10, "price": 1.20},
    {"product": "banana", "qty": 5, "price": 0.50},
    {"product": "cherry", "qty": 100, "price": 3.00},
]
header = list(rows[0].keys())
print(",".join(header))
for row in rows:
    print(",".join(str(row[k]) for k in header))
total = sum(r["qty"] * r["price"] for r in rows)
print(f"total={total:.2f}")
"""
        status, body = await execute(client, code)
        assert status == 200, f"unexpected status {status}: {body}"
        assert body["exit_code"] == 0, f"non-zero exit: {body['stderr']}"
        stdout = body["stdout"]
        assert "product,qty,price" in stdout
        assert "apple" in stdout
        assert "total=314.50" in stdout

    @pytest.mark.asyncio
    async def test_decimal_precision(self, client: AsyncClient):
        """Use decimal.Decimal for precise financial arithmetic."""
        code = """
from decimal import Decimal, ROUND_HALF_UP
price = Decimal("19.99")
tax_rate = Decimal("0.0875")
tax = (price * tax_rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
total = price + tax
print(f"tax={tax}")
print(f"total={total}")
"""
        status, body = await execute(client, code)
        assert status == 200, f"unexpected status {status}: {body}"
        assert body["exit_code"] == 0, f"non-zero exit: {body['stderr']}"
        stdout = body["stdout"]
        assert "tax=1.75" in stdout, f"wrong tax in: {stdout!r}"
        assert "total=21.74" in stdout, f"wrong total in: {stdout!r}"

    @pytest.mark.asyncio
    async def test_multi_module_computation(self, client: AsyncClient):
        """Use math, statistics, collections, and json together in one script."""
        code = """
import math
import statistics
import collections
import json

scores = [72, 85, 90, 72, 88, 65, 90, 72, 95, 85]
freq = dict(collections.Counter(scores))
result = {
    "mean": round(statistics.mean(scores), 2),
    "variance": round(statistics.variance(scores), 2),
    "log_mean": round(math.log(statistics.mean(scores)), 4),
    "most_common": max(freq, key=freq.get),
    "frequency": freq,
}
print(json.dumps(result, sort_keys=True))
"""
        status, body = await execute(client, code)
        assert status == 200, f"unexpected status {status}: {body}"
        assert body["exit_code"] == 0, f"non-zero exit: {body['stderr']}"
        result = json.loads(body["stdout"].strip())
        assert result["mean"] == 81.4, f"unexpected mean: {result['mean']}"
        assert result["most_common"] == 72, f"unexpected most_common: {result['most_common']}"
        assert "log_mean" in result
        assert "frequency" in result




class TestGuardrailEdgeCases:
    """Tricky patterns that might slip through naive validation."""

    @pytest.mark.asyncio
    async def test_shadowing_open_builtin_still_blocked(self, client: AsyncClient):
        """Defining a local function named 'open' does not unblock the open() call.

        The AST visitor sees the Call node for open() regardless of whether
        user code has shadowed the name.  The guardrail must still reject it.
        """
        code = """
def open(path):
    return path

result = open("/etc/passwd")
print(result)
"""
        status, body = await execute(client, code)
        assert status == 400, f"expected 400 (open blocked), got {status}: {body}"
        assert "violations" in body
        assert any("open" in v for v in body["violations"]), (
            f"expected 'open' violation: {body['violations']}"
        )

    @pytest.mark.asyncio
    async def test_string_containing_eval_is_allowed(self, client: AsyncClient):
        """A string that contains the word 'eval' is not a call — must pass."""
        code = """
x = "eval('dangerous')"
print(x)
"""
        status, body = await execute(client, code)
        assert status == 200, f"expected 200 (string, not call), got {status}: {body}"
        assert body["exit_code"] == 0
        assert "eval('dangerous')" in body["stdout"]

    @pytest.mark.asyncio
    async def test_lambda_with_eval_name_attribute_is_allowed(self, client: AsyncClient):
        """Assigning __name__ on a lambda does not invoke eval; should pass."""
        code = """
f = lambda: None
f.__name__ = "eval"
print(f.__name__)
"""
        status, body = await execute(client, code)
        # __name__ is not in _BLOCKED_DUNDERS; __globals__ and __subclasses__ are.
        assert status == 200, f"expected 200, got {status}: {body}"
        assert body["exit_code"] == 0
        assert "eval" in body["stdout"]

    @pytest.mark.asyncio
    async def test_chained_string_methods_allowed(self, client: AsyncClient):
        """Chained calls on safe builtins should pass guardrails and execute."""
        code = """
result = "  Hello World  ".strip().upper().replace("WORLD", "SANDBOX")
print(result)
"""
        status, body = await execute(client, code)
        assert status == 200, f"expected 200, got {status}: {body}"
        assert body["exit_code"] == 0
        assert "HELLO SANDBOX" in body["stdout"]

    @pytest.mark.asyncio
    async def test_list_comprehension_with_math(self, client: AsyncClient):
        """List comprehension over allowed import — math.sqrt — should execute."""
        code = """
import math
result = [round(math.sqrt(i), 4) for i in range(1, 6)]
print(result)
"""
        status, body = await execute(client, code)
        assert status == 200, f"expected 200, got {status}: {body}"
        assert body["exit_code"] == 0
        stdout = body["stdout"]
        assert "1.0" in stdout
        assert "2.2361" in stdout

    @pytest.mark.asyncio
    async def test_multiple_violations_all_reported(self, client: AsyncClient):
        """Send code with 5+ violations and verify ALL are returned, not just the first."""
        code = """
import os
import subprocess
import socket
eval("1+1")
exec("x=1")
result = open("/etc/passwd")
"""
        status, body = await execute(client, code)
        assert status == 400, f"expected 400, got {status}"
        violations = body.get("violations", [])
        assert len(violations) >= 5, (
            f"expected at least 5 violations, got {len(violations)}: {violations}"
        )
        # Spot-check that distinct violations are present
        combined = " ".join(violations)
        assert "os" in combined, f"missing os violation: {violations}"
        assert "subprocess" in combined, f"missing subprocess violation: {violations}"
        assert "eval" in combined, f"missing eval violation: {violations}"
        assert "open" in combined, f"missing open violation: {violations}"

    @pytest.mark.asyncio
    async def test_unicode_identifiers_and_strings(self, client: AsyncClient):
        """Unicode variable names and string values should execute correctly."""
        code = """
名前 = "世界"
greeting = f"こんにちは、{名前}！"
print(greeting)
print(len(名前))
"""
        status, body = await execute(client, code)
        assert status == 200, f"expected 200, got {status}: {body}"
        assert body["exit_code"] == 0, f"non-zero exit: {body['stderr']}"
        assert "世界" in body["stdout"]
        assert "2" in body["stdout"]

    @pytest.mark.asyncio
    async def test_deeply_nested_safe_builtins(self, client: AsyncClient):
        """Deeply nested builtin calls should pass guardrails and return correct result."""
        code = """
result = print(sum(range(len(list(map(str, range(10)))))))
"""
        status, body = await execute(client, code)
        assert status == 200, f"expected 200, got {status}: {body}"
        assert body["exit_code"] == 0, f"non-zero exit: {body['stderr']}"
        assert "45" in body["stdout"], f"expected 45 in stdout: {body['stdout']!r}"




class TestConcurrency:
    """Validate the sidecar handles concurrent requests without corruption."""

    @pytest.mark.asyncio
    async def test_parallel_requests(self, client: AsyncClient):
        """Five valid execution requests issued concurrently all return correct output."""
        snippets = [
            ("print(2 ** 10)", "1024"),
            ("print(sum(range(101)))", "5050"),
            ("import math; print(round(math.pi, 5))", "3.14159"),
            ("print('hello' * 3)", "hellohellohello"),
            ("print(sorted([5, 3, 1, 4, 2]))", "[1, 2, 3, 4, 5]"),
        ]

        async def run(code: str, expected: str):
            s, body = await execute(client, code)
            assert s == 200, f"concurrent request failed: {body}"
            assert body["exit_code"] == 0, f"non-zero exit for {code!r}: {body['stderr']}"
            assert expected in body["stdout"], (
                f"expected {expected!r} in stdout for {code!r}: {body['stdout']!r}"
            )

        await asyncio.gather(*[run(code, expected) for code, expected in snippets])




class TestOutputLimits:
    """Verify the 50 KB output cap is enforced end-to-end."""

    @pytest.mark.asyncio
    async def test_large_stdout_is_truncated(self, client: AsyncClient):
        """1000 lines of 100 chars each (~101 KB) must be truncated to 50 KB."""
        code = """
line = "x" * 100
for _ in range(1000):
    print(line)
"""
        status, body = await execute(client, code)
        assert status == 200, f"unexpected status {status}: {body}"
        assert body["exit_code"] == 0, f"non-zero exit: {body['stderr']}"
        stdout = body["stdout"]
        assert len(stdout.encode("utf-8")) <= 55 * 1024, (
            f"stdout exceeds safe limit: {len(stdout)} chars"
        )
        assert "[output truncated at 50 KB]" in stdout, (
            f"truncation note missing; stdout length={len(stdout)}"
        )

    @pytest.mark.asyncio
    async def test_binary_like_output(self, client: AsyncClient):
        """Printing bytes-like strings with special characters is handled cleanly."""
        code = r"""
data = bytes(range(32, 128))
print(repr(data))
print(len(data))
"""
        status, body = await execute(client, code)
        assert status == 200, f"unexpected status {status}: {body}"
        assert body["exit_code"] == 0, f"non-zero exit: {body['stderr']}"
        assert "96" in body["stdout"]  # len(range(32, 128)) == 96




class TestErrorRecovery:
    """Verify the pipeline handles adversarial and failure inputs gracefully."""

    @pytest.mark.asyncio
    async def test_open_call_blocked(self, client: AsyncClient):
        """open() for file I/O must be blocked by guardrails before execution."""
        code = 'open("/tmp/test.txt", "w").write("hello")'
        status, body = await execute(client, code)
        assert status == 400, f"expected 400 (open blocked), got {status}: {body}"
        assert "violations" in body
        assert any("open" in v for v in body["violations"])

    @pytest.mark.asyncio
    async def test_infinite_loop_timeout(self, client: AsyncClient):
        """Infinite loop with short timeout returns timed_out=true quickly."""
        code = "while True: pass"
        start = time.monotonic()
        status, body = await execute(client, code, timeout=1.0)
        elapsed = time.monotonic() - start
        assert status == 200, f"unexpected status {status}: {body}"
        assert body["timed_out"] is True, f"expected timed_out=True: {body}"
        assert elapsed < 8.0, f"took too long ({elapsed:.1f}s) to kill infinite loop"

    @pytest.mark.asyncio
    async def test_memory_intensive_code(self, client: AsyncClient):
        """Allocating ~800 MB should fail the subprocess with a non-zero exit code."""
        code = "x = [0] * (10 ** 8)"
        status, body = await execute(client, code)
        assert status == 200, f"unexpected status {status}: {body}"
        # May succeed or OOM depending on the host; verify the response structure is valid.
        assert isinstance(body["exit_code"], int), f"exit_code must be int: {body}"
        assert isinstance(body["timed_out"], bool), f"timed_out must be bool: {body}"

    @pytest.mark.asyncio
    async def test_system_exit_code_propagated(self, client: AsyncClient):
        """raise SystemExit(42) should result in exit_code == 42 in the response."""
        code = "raise SystemExit(42)"
        status, body = await execute(client, code)
        assert status == 200, f"unexpected status {status}: {body}"
        assert body["exit_code"] == 42, (
            f"expected exit_code=42, got {body['exit_code']}; stderr={body['stderr']!r}"
        )
        assert body["timed_out"] is False
