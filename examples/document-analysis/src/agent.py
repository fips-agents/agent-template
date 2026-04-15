"""Document analysis workflow demonstrating mixed node types and conditional routing.

Pipeline: classify -> (extract | summarize | fallback) -> validate -> format_report -> END

ClassifyNode (BaseNode) uses structural signals to route the document to the
appropriate processing node. ExtractSpecsNode and SummarizeNode (AgentNode) use
the LLM for heavy lifting. ValidateNode and FormatReportNode (BaseNode) handle
post-processing without LLM calls.
"""

from __future__ import annotations

import logging
import re

from fipsagents.workflow import (
    END,
    AgentNode,
    BaseNode,
    Graph,
    WorkflowRunner,
    WorkflowState,
    node,
)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class DocumentState(WorkflowState):
    """State flowing through the document analysis pipeline."""

    document: str                       # Input document text
    document_type: str = ""             # "technical", "narrative", or "unknown"
    extracted_specs: str = ""           # Structured extraction from technical docs
    summary: str = ""                   # Summary of narrative docs
    validation_errors: list[str] = []   # Issues found during validation
    report: str = ""                    # Final formatted report


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


@node()
class ClassifyNode(BaseNode):
    """Pure-logic classifier. Routes based on structural signals in the document."""

    async def process(self, state: DocumentState) -> DocumentState:
        indicators = 0
        text = state.document

        if re.search(r"```", text):
            indicators += 1
        if re.search(r"\d+\.\s", text):
            indicators += 1
        if re.search(
            r"\b(requirement|specification|version|API|endpoint|parameter|schema|config)\b",
            text,
            re.I,
        ):
            indicators += 1
        if re.search(r"\b[A-Z]{4,}\b", text):
            indicators += 1  # ALL-CAPS words (4+ chars to skip acronyms like CEO, AI)
        if "|" in text and "-" in text:
            indicators += 1  # table-like

        if indicators >= 2:
            doc_type = "technical"
        elif len(text.split()) > 50 and indicators == 0:
            doc_type = "narrative"
        else:
            doc_type = "unknown"

        self.logger.info(
            "Classified document as %r (%d indicators)", doc_type, indicators
        )
        return state.model_copy(update={"document_type": doc_type})


@node()
class ExtractSpecsNode(AgentNode):
    """Uses the LLM to extract structured specs from technical documents."""

    async def process(self, state: DocumentState) -> DocumentState:
        prompt = self.prompts.render("extract", document=state.document)
        self.add_message("user", prompt)
        response = await self.call_model(include_tools=False)
        return state.model_copy(update={"extracted_specs": response.content or ""})


@node()
class SummarizeNode(AgentNode):
    """Uses the LLM to produce a concise summary of narrative documents."""

    async def process(self, state: DocumentState) -> DocumentState:
        prompt = self.prompts.render("summarize", document=state.document)
        self.add_message("user", prompt)
        response = await self.call_model(include_tools=False)
        return state.model_copy(update={"summary": response.content or ""})


@node()
class FallbackNode(BaseNode):
    """Handles documents that could not be classified."""

    async def process(self, state: DocumentState) -> DocumentState:
        return state.model_copy(
            update={
                "summary": (
                    f"Document could not be classified. "
                    f"Raw text length: {len(state.document)} characters."
                )
            }
        )


@node()
class ValidateNode(BaseNode):
    """Checks that LLM-produced output meets minimum quality thresholds."""

    async def process(self, state: DocumentState) -> DocumentState:
        errors: list[str] = []
        if state.document_type == "technical" and len(state.extracted_specs.strip()) < 20:
            errors.append("Technical extraction produced insufficient output")
        if state.document_type == "narrative" and len(state.summary.strip()) < 20:
            errors.append("Summary produced insufficient output")
        if errors:
            self.logger.warning("Validation issues: %s", errors)
        return state.model_copy(update={"validation_errors": errors})


@node()
class FormatReportNode(BaseNode):
    """Assembles a final Markdown report from all processed state fields."""

    async def process(self, state: DocumentState) -> DocumentState:
        sections = [
            "# Document Analysis Report\n",
            f"**Type:** {state.document_type}\n",
        ]
        if state.extracted_specs:
            sections.append(f"## Extracted Specifications\n\n{state.extracted_specs}\n")
        if state.summary:
            sections.append(f"## Summary\n\n{state.summary}\n")
        if state.validation_errors:
            sections.append("## Validation Issues\n")
            for err in state.validation_errors:
                sections.append(f"- {err}\n")
        report = "\n".join(sections)
        return state.model_copy(update={"report": report})


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------


def build_graph() -> Graph:
    """Construct the document analysis workflow graph."""
    graph = Graph(state_type=DocumentState)

    graph.add_node("classify", ClassifyNode())
    graph.add_node("extract", ExtractSpecsNode())
    graph.add_node("summarize", SummarizeNode())
    graph.add_node("fallback", FallbackNode())
    graph.add_node("validate", ValidateNode())
    graph.add_node("format_report", FormatReportNode())

    graph.set_entry_point("classify")

    # Conditional routing based on document type
    graph.add_conditional_edge(
        "classify",
        lambda s: {
            "technical": "extract",
            "narrative": "summarize",
        }.get(s.document_type, "fallback"),
    )

    # All paths converge at validate -> format_report -> END
    graph.add_edge("extract", "validate")
    graph.add_edge("summarize", "validate")
    graph.add_edge("fallback", "format_report")  # Skip validation for fallback
    graph.add_edge("validate", "format_report")
    graph.add_edge("format_report", END)

    return graph


# ---------------------------------------------------------------------------
# Sample documents
# ---------------------------------------------------------------------------

TECHNICAL_DOC = """
# API Specification v2.3

## Authentication Endpoint

POST /api/v1/auth/token

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| client_id | string | yes | OAuth2 client identifier |
| client_secret | string | yes | OAuth2 client secret |
| grant_type | string | yes | Must be "client_credentials" |

### Response Schema

```json
{
    "access_token": "string",
    "token_type": "bearer",
    "expires_in": 3600
}
```

### Requirements

1. All tokens expire after 3600 seconds
2. Rate limit: 100 requests per minute per client
3. TLS 1.2 minimum required
""".strip()

NARRATIVE_DOC = """
The quarterly board meeting revealed several interesting developments in
the company's strategic direction. The CEO emphasized that artificial
intelligence would play a central role in the next phase of growth,
noting that early investments in machine learning infrastructure had
already begun to pay dividends. Revenue from AI-powered features grew
forty-two percent compared to the same quarter last year, outpacing
the company's overall growth rate by a significant margin.

Several board members raised concerns about the pace of regulatory
change in the AI space, particularly in the European Union where new
compliance requirements could affect product timelines. The legal team
presented a detailed analysis of the proposed regulations and their
potential impact on the company's roadmap. Despite these challenges,
the consensus was that the company's proactive approach to responsible
AI development positioned it well for whatever regulatory framework
emerges.
""".strip()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    """Run the document analysis workflow with sample documents."""
    logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")

    for name, doc in [("Technical", TECHNICAL_DOC), ("Narrative", NARRATIVE_DOC)]:
        print(f"\n{'=' * 60}")
        print(f"Processing: {name} Document")
        print(f"{'=' * 60}\n")

        graph = build_graph()
        runner = WorkflowRunner(graph, max_steps=10)
        result = await runner.start(DocumentState(document=doc))

        print(result.report)
        print(f"\nDocument type: {result.document_type}")
        if result.validation_errors:
            print(f"Validation issues: {result.validation_errors}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
