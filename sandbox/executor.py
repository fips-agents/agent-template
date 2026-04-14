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


@dataclasses.dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False


async def execute_code(code: str, timeout: float = 10.0) -> ExecutionResult:
    """Execute *code* in an isolated Python subprocess and return the result.

    Args:
        code: Python source code to run.
        timeout: Wall-clock seconds before the process is killed.

    Returns:
        An :class:`ExecutionResult` with captured stdout, stderr, exit code,
        and a flag indicating whether the process was killed due to timeout.
    """
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
