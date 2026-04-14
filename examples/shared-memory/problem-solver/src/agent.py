"""Problem Solver agent — answers analytical questions with code verification."""

import json
import logging
import re

from fipsagents.baseagent import BaseAgent, StepResult

logger = logging.getLogger(__name__)

# Pattern to extract memory content from LLM response
_MEMORY_PATTERN = re.compile(r"\[MEMORY\]\s*(.*?)\s*\[/MEMORY\]", re.DOTALL)

PROJECT_ID = "agent-template-demo"


class ProblemSolver(BaseAgent):
    """Analytical problem solver with code execution and memory."""

    async def step(self) -> StepResult:
        # Insert system prompt on first step
        if not self.messages or self.messages[0].get("role") != "system":
            system_prompt = self.build_system_prompt()
            self.messages.insert(0, {"role": "system", "content": system_prompt})

        # Call the model
        response = await self.call_model()

        # Handle tool calls
        while response.tool_calls:
            self.messages.append({
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in response.tool_calls
                ],
            })
            for tc in response.tool_calls:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                result = await self.tools.execute(tc.function.name, **args)
                self.messages.append({
                    "role": "tool",
                    "content": result.result if not result.is_error else f"Error: {result.error}",
                    "tool_call_id": tc.id,
                })
            response = await self.call_model()

        content = response.content or ""

        # Extract and write memories
        memories_written = []
        for match in _MEMORY_PATTERN.finditer(content):
            memory_text = match.group(1).strip()
            if memory_text and self.memory:
                result = await self.memory.write(
                    content=memory_text,
                    scope="user",
                    weight=0.8,
                )
                if result:
                    memories_written.append(memory_text)
                    logger.info("Wrote memory: %s", memory_text[:80])

        # Strip memory tags from the user-visible response
        clean_content = _MEMORY_PATTERN.sub("", content).strip()

        self.add_message("assistant", clean_content)
        return StepResult.done(result=json.dumps({
            "response": clean_content,
            "memories_written": memories_written,
        }))
