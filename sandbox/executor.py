"""Subprocess-based Python code executor with timeout and output capping.

Executes validated Python code in an isolated subprocess using ``python3 -I``
(isolated mode: no user site-packages, PYTHON* env vars ignored).  Code is
written to a temporary file under ``/tmp`` and cleaned up unconditionally.
"""

import asyncio
import dataclasses
import os
import tempfile

_MAX_OUTPUT_BYTES = 50 * 1024  # 50 KB per stream
_TRUNCATION_NOTE = "\n[output truncated at 50 KB]"


def _build_preamble() -> str:
    """Build a runtime preamble that restricts imports in the subprocess.

    This is defense-in-depth: even if AST guardrails are bypassed via
    dynamic string construction (chr(), bytes.decode(), etc.), the
    runtime import restriction prevents importing dangerous modules.

    Uses a denylist rather than an allowlist because allowed stdlib
    modules have transitive dependencies (e.g. ``random`` → ``os``,
    ``statistics`` → ``numbers``) that would break with an allowlist.
    The AST guardrails enforce the allowlist; this layer blocks modules
    that are never needed by allowed modules' dependency chains.
    """
    # Modules that are dangerous AND not in any allowed module's
    # transitive dependency chain.  os, importlib, and marshal are
    # excluded from this list because they ARE needed internally.
    return (
        "def __sandbox_setup__():\n"
        "    import sys as _sys\n"
        "    _denied = frozenset({\n"
        "        'os', 'sys', 'builtins',\n"
        "        'subprocess', 'socket', 'ctypes', 'multiprocessing',\n"
        "        'pty', 'shutil', 'signal', 'mmap',\n"
        "        'http', 'urllib', 'xmlrpc', 'ftplib', 'smtplib',\n"
        "        'webbrowser', 'antigravity',\n"
        "        'pickle', 'marshal', 'shelve',\n"
        "        'code', 'codeop',\n"
        "    })\n"
        "    # Purge the most critical module reference: random._os → os.\n"
        "    # Other references (statistics.sys, etc.) are caught by AST.\n"
        "    # Generic purge is too destructive — collections._sys breaks namedtuple.\n"
        "    try:\n"
        "        import random as _random\n"
        "        if hasattr(_random, '_os'):\n"
        "            delattr(_random, '_os')\n"
        "    except ImportError:\n"
        "        pass\n"
        "    _bi = __builtins__\n"
        "    _d = _bi if isinstance(_bi, dict) else _bi.__dict__\n"
        "    _orig = _d['__import__']\n"
        "    def _rimp(name, gl=None, lo=None, fromlist=(), level=0):\n"
        "        top = name.split('.')[0]\n"
        "        if level == 0 and top in _denied:\n"
        "            caller = (gl or {}).get('__name__', '__main__')\n"
        "            if caller == '__main__':\n"
        "                raise ImportError(f\"import of '{name}' blocked by sandbox\")\n"  # noqa: Q003
        "        return _orig(name, gl, lo, fromlist, level)\n"
        "    _d['__import__'] = _rimp\n"
        "    del _sys\n"
        "__sandbox_setup__()\n"
        "del __sandbox_setup__\n"
    )


@dataclasses.dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False


async def execute_code(
    code: str,
    timeout: float = 10.0,
    *,
    runtime_restrict: bool = True,
) -> ExecutionResult:
    """Execute *code* in an isolated Python subprocess and return the result.

    Args:
        code: Python source code to run.
        timeout: Wall-clock seconds before the process is killed.
        runtime_restrict: If True (default), a runtime preamble is prepended
            that blocks imports of dangerous modules (subprocess, socket,
            ctypes, etc.).  Defense-in-depth against AST bypasses.

    Returns:
        An :class:`ExecutionResult` with captured stdout, stderr, exit code,
        and a flag indicating whether the process was killed due to timeout.
    """
    if runtime_restrict:
        code = _build_preamble() + code

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".py", dir="/tmp", delete=False, mode="w", encoding="utf-8"
        ) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        process = await asyncio.create_subprocess_exec(
            "python3", "-I", tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            raw_stdout, raw_stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            # Drain whatever was buffered before the kill
            try:
                raw_stdout, raw_stderr = await asyncio.wait_for(
                    process.communicate(), timeout=5.0
                )
            except asyncio.TimeoutError:
                raw_stdout, raw_stderr = b"", b""
            return ExecutionResult(
                stdout=_decode(raw_stdout),
                stderr=f"Execution timed out after {timeout}s",
                exit_code=process.returncode if process.returncode is not None else -1,
                timed_out=True,
            )

        return ExecutionResult(
            stdout=_decode(raw_stdout),
            stderr=_decode(raw_stderr),
            exit_code=process.returncode,
        )
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass


def _decode(raw: bytes) -> str:
    # Truncate bytes before decoding to enforce a consistent size cap
    # regardless of character encoding.
    if len(raw) > _MAX_OUTPUT_BYTES:
        raw = raw[:_MAX_OUTPUT_BYTES]
        return raw.decode("utf-8", errors="replace") + _TRUNCATION_NOTE
    return raw.decode("utf-8", errors="replace")
