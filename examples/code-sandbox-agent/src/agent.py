"""Code Sandbox Agent — answers questions by writing and executing Python."""

import json

from fipsagents.baseagent import BaseAgent, StepResult


class CodeSandboxAgent(BaseAgent):
    """Agent that writes and executes Python code to answer questions.

    Uses the code_executor tool to run LLM-generated code in an isolated
    sandbox sidecar, then returns the results.
    """

    async def step(self) -> StepResult:
        response = await self.call_model()

        # Tool-call loop: the LLM may call code_executor one or more times.
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
                    "content": result.result if not result.is_error else f"ERROR: {result.error}",
                    "tool_call_id": tc.id,
                })

            response = await self.call_model()

        return StepResult.done(response.content)
