"""Research Assistant — example agent subclass demonstrating BaseAgent patterns.

Takes a research query, searches the web, validates relevance, and produces
a structured report with citations.  Shows all three model-calling patterns:
``call_model``, ``call_model_json``, and ``call_model_validated``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field

from base_agent import BaseAgent, ModelResponse, StepResult


class ResearchReport(BaseModel):
    """Structured output schema for the final research report."""

    answer: str = Field(description="The research answer in Markdown")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    citations: list[str] = Field(default_factory=list, description="Source URLs")


class ResearchAssistant(BaseAgent):
    """A research assistant that searches, evaluates, and reports."""

    async def step(self) -> StepResult:
        # 1. Set system prompt once at the start of the conversation.
        #    Guard prevents duplicate system messages when run() calls
        #    step() multiple times.
        if not self.messages or self.messages[0].get("role") != "system":
            system_prompt = self.build_system_prompt()
            self.messages.insert(0, {"role": "system", "content": system_prompt})

        # 2. Call the model with LLM-visible tools (e.g. web_search).
        #    The LLM decides whether to search.
        response = await self.call_model()

        # 3. Handle any tool calls the LLM made (search, follow-ups).
        #    Include tool_call_id — required by the OpenAI-compatible API.
        while response.tool_calls:
            for tc in response.tool_calls:
                fn = tc.function
                args = json.loads(fn.arguments) if fn.arguments else {}
                result = await self.tools.execute(fn.name, **args)
                self.messages.append({
                    "role": "tool",
                    "content": result.result,
                    "tool_call_id": tc.id,
                })
            response = await self.call_model()

        # 4. Produce structured output via call_model_json
        report_messages = self.messages + [
            {
                "role": "user",
                "content": (
                    "Based on the research above, produce a structured "
                    "research report as JSON."
                ),
            },
        ]
        report = await self.call_model_json(
            ResearchReport, messages=report_messages
        )

        # 5. Validate relevance via call_model_validated
        query = next(
            (m["content"] for m in self.messages if m["role"] == "user"),
            "",
        )

        def validate_relevance(resp: ModelResponse) -> str:
            """Check the model confirms the report addresses the query."""
            text = (resp.content or "").lower()
            if "not relevant" in text or "does not address" in text:
                raise ValueError("Report does not address the original query")
            return resp.content or ""

        relevance_check = await self.call_model_validated(
            validate_relevance,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Does this report address the query '{query}'? "
                        f"Answer: {report.answer[:200]}"
                    ),
                },
            ],
        )
        logger.debug("Relevance validation passed: %s", relevance_check[:80])

        # 6. Format citations using the agent-only tool (plane 1)
        if report.citations:
            cite_result = await self.use_tool(
                "format_citations",
                urls=report.citations,
                titles=["Source"] * len(report.citations),
            )
            if not cite_result.is_error:
                report.citations = cite_result.result.splitlines()

        return StepResult.done(result=report)
