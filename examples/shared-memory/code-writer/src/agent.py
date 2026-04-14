"""Code Writer agent — generates validated Python code with MemoryHub context.

Implements a four-phase workflow in step():
  1. Search MemoryHub for relevant context
  2. Generate code via LLM using that context
  3. Validate in the sandbox sidecar
  4. Extract any [MEMORY] tags and persist them
"""

import json
import logging
import os
import re

import httpx

from fipsagents.baseagent import BaseAgent, StepResult

logger = logging.getLogger(__name__)

_MEMORY_PATTERN = re.compile(r"\[MEMORY\]\s*(.*?)\s*\[/MEMORY\]", re.DOTALL)
_CODE_PATTERN = re.compile(r"```python\s*(.*?)```", re.DOTALL)

PROJECT_ID = "agent-template-demo"


def _sandbox_url() -> str:
    return os.environ.get("SANDBOX_URL", "http://localhost:8000")


class CodeWriter(BaseAgent):
    """Code generator with sandbox validation and MemoryHub context."""

    async def step(self) -> StepResult:
        """Four-phase workflow: fetch context -> generate -> validate -> output."""
        # Find the user's request (last user message)
        request = ""
        for msg in reversed(self.messages):
            if msg.get("role") == "user":
                request = msg.get("content", "")
                break

        if not request:
            return StepResult.done(result=json.dumps({
                "response": "No request provided.",
                "code": "",
                "validation_passed": False,
                "validation_output": "",
                "context_used": [],
                "memories_written": [],
            }))

        # Phase 1: Fetch context from MemoryHub
        context_items = await self._fetch_context(request)
        context_text = (
            "\n".join(f"- {c}" for c in context_items)
            if context_items
            else "No prior context available."
        )
        logger.info("Loaded %d memories for context", len(context_items))

        # Phase 2: Generate code using the generate prompt
        raw_output = await self._generate_code(request, context_text)

        # Extract code from the ```python ... ``` block
        code_match = _CODE_PATTERN.search(raw_output)
        code = code_match.group(1).strip() if code_match else raw_output.strip()

        # Phase 3: Validate in sandbox
        validation_passed, validation_result = await self._validate_code(code)

        # Phase 4: Persist any [MEMORY] tags found in the response
        memories_written = await self._persist_memories(raw_output)

        # Build formatted output
        output_parts = [f"```python\n{code}\n```"]
        if validation_passed:
            output_parts.append(
                f"\n**Validation passed.** Output:\n```\n{validation_result.strip()}\n```"
            )
        elif validation_result:
            output_parts.append(
                f"\n**Validation failed:**\n```\n{validation_result.strip()}\n```"
            )
        if context_items:
            output_parts.append(
                f"\n*Used {len(context_items)} memories for context.*"
            )

        clean_output = "\n".join(output_parts)
        self.add_message("assistant", clean_output)

        return StepResult.done(result=json.dumps({
            "response": clean_output,
            "code": code,
            "validation_passed": validation_passed,
            "validation_output": validation_result,
            "context_used": context_items,
            "memories_written": memories_written,
        }))

    async def _fetch_context(self, request: str) -> list[str]:
        """Search MemoryHub and return a list of relevant content strings."""
        if not self.memory:
            return []
        try:
            memories = await self.memory.search(
                query=request,
                owner_id="",  # search across all owners for shared memories
                max_results=5,
            )
            return [
                mem.get("content", "")
                for mem in memories
                if mem.get("content", "")
            ]
        except Exception:
            logger.warning("MemoryHub search failed — continuing without context", exc_info=True)
            return []

    async def _generate_code(self, request: str, context_text: str) -> str:
        """Call the LLM with the generate prompt and return the raw response."""
        prompt_text = self.prompts.render(
            "generate",
            request=request,
            context=context_text,
        )

        # Build the system prompt with context injected
        system_text = self.prompts.render("system", context=context_text)

        gen_messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": prompt_text},
        ]
        response = await self.call_model(messages=gen_messages, include_tools=False)
        return response.content or ""

    async def _validate_code(self, code: str) -> tuple[bool, str]:
        """Submit code to the sandbox sidecar and return (passed, output)."""
        if not code:
            return False, "No code to validate."

        sandbox = _sandbox_url()
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{sandbox}/execute",
                    json={"code": code, "timeout": 10},
                    timeout=15,
                )
                data = resp.json()
        except httpx.ConnectError:
            msg = f"Sandbox unavailable at {sandbox} — skipping validation."
            logger.warning(msg)
            return False, msg
        except Exception as exc:
            msg = f"Sandbox request failed: {exc}"
            logger.warning(msg)
            return False, msg

        if not resp.is_success:
            error = data.get("error", resp.text)
            violations = data.get("violations", [])
            detail = error
            if violations:
                detail += "\n" + "\n".join(f"  - {v}" for v in violations)
            return False, detail

        exit_code = data.get("exit_code", 0)
        stdout = data.get("stdout", "").rstrip()
        stderr = data.get("stderr", "").rstrip()

        parts = []
        if stdout:
            parts.append(stdout)
        if stderr:
            parts.append(f"stderr:\n{stderr}")
        if exit_code != 0:
            parts.append(f"exit code: {exit_code}")

        result_text = "\n".join(parts) if parts else "(no output)"
        return exit_code == 0, result_text

    async def _persist_memories(self, raw_output: str) -> list[str]:
        """Extract [MEMORY]...[/MEMORY] tags and write them to MemoryHub."""
        written: list[str] = []
        if not self.memory:
            return written

        for match in _MEMORY_PATTERN.finditer(raw_output):
            memory_text = match.group(1).strip()
            if not memory_text:
                continue
            try:
                result = await self.memory.write(
                    content=memory_text,
                    scope="user",
                    weight=0.8,
                )
                if result:
                    written.append(memory_text)
                    logger.info("Wrote memory: %s", memory_text[:80])
            except Exception:
                logger.warning(
                    "Failed to write memory: %s", memory_text[:80], exc_info=True
                )

        return written
