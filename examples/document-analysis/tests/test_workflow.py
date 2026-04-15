"""Unit tests for the document analysis workflow.

Tests cover individual nodes, graph structure, and full end-to-end flows
with mocked LLM calls.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from fipsagents.baseagent.config import AgentConfig, BackoffConfig, LLMConfig, LoopConfig
from fipsagents.baseagent.llm import LLMClient, ModelResponse
from fipsagents.baseagent.memory import NullMemoryClient
from fipsagents.workflow import END, AgentNode, Graph, WorkflowRunner

# Make src/ importable so we can import the example's agent module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent import (
    NARRATIVE_DOC,
    TECHNICAL_DOC,
    ClassifyNode,
    DocumentState,
    ExtractSpecsNode,
    FallbackNode,
    FormatReportNode,
    SummarizeNode,
    ValidateNode,
    build_graph,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config() -> AgentConfig:
    return AgentConfig(
        model=LLMConfig(endpoint="http://fake:8080/v1", name="test-model"),
        loop=LoopConfig(max_iterations=10, backoff=BackoffConfig()),
    )


def _wire_agent(agent: AgentNode) -> None:
    """Manually wire subsystems so no real services are needed."""
    agent.config = _make_config()
    agent.llm = MagicMock(spec=LLMClient)
    agent.memory = NullMemoryClient()
    agent._setup_done = True


def _mock_response(content: str) -> MagicMock:
    """Create a MagicMock that looks like a ModelResponse."""
    resp = MagicMock(spec=ModelResponse)
    resp.content = content
    resp.tool_calls = None
    return resp


# ---------------------------------------------------------------------------
# ClassifyNode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestClassifyNode:
    async def test_technical_doc_classified(self):
        node = ClassifyNode()
        result = await node.process(DocumentState(document=TECHNICAL_DOC))
        assert result.document_type == "technical"

    async def test_narrative_doc_classified(self):
        node = ClassifyNode()
        result = await node.process(DocumentState(document=NARRATIVE_DOC))
        assert result.document_type == "narrative"

    async def test_short_ambiguous_text_is_unknown(self):
        node = ClassifyNode()
        result = await node.process(DocumentState(document="Hello world."))
        assert result.document_type == "unknown"

    async def test_technical_needs_two_indicators(self):
        """A document with only one indicator should not be classified as technical."""
        node = ClassifyNode()
        # Only one indicator: a numbered list
        doc = "1. First item\n2. Second item\nnothing else here"
        result = await node.process(DocumentState(document=doc))
        assert result.document_type != "technical"

    async def test_does_not_mutate_original_state(self):
        node = ClassifyNode()
        original = DocumentState(document=TECHNICAL_DOC)
        result = await node.process(original)
        assert original.document_type == ""  # unchanged
        assert result.document_type == "technical"


# ---------------------------------------------------------------------------
# FallbackNode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFallbackNode:
    async def test_produces_summary_with_character_count(self):
        node = FallbackNode()
        doc = "Some short text"
        result = await node.process(DocumentState(document=doc))
        assert str(len(doc)) in result.summary
        assert "could not be classified" in result.summary


# ---------------------------------------------------------------------------
# ValidateNode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestValidateNode:
    async def test_passes_with_good_technical_data(self):
        node = ValidateNode()
        state = DocumentState(
            document="x",
            document_type="technical",
            extracted_specs="This is a sufficiently long extraction result for validation.",
        )
        result = await node.process(state)
        assert result.validation_errors == []

    async def test_catches_insufficient_extraction(self):
        node = ValidateNode()
        state = DocumentState(
            document="x",
            document_type="technical",
            extracted_specs="short",
        )
        result = await node.process(state)
        assert len(result.validation_errors) == 1
        assert "extraction" in result.validation_errors[0].lower()

    async def test_catches_insufficient_summary(self):
        node = ValidateNode()
        state = DocumentState(
            document="x",
            document_type="narrative",
            summary="tiny",
        )
        result = await node.process(state)
        assert len(result.validation_errors) == 1
        assert "summary" in result.validation_errors[0].lower()

    async def test_passes_with_good_narrative_data(self):
        node = ValidateNode()
        state = DocumentState(
            document="x",
            document_type="narrative",
            summary="A detailed multi-sentence summary of the document content.",
        )
        result = await node.process(state)
        assert result.validation_errors == []

    async def test_no_errors_for_unknown_type(self):
        node = ValidateNode()
        state = DocumentState(document="x", document_type="unknown")
        result = await node.process(state)
        assert result.validation_errors == []


# ---------------------------------------------------------------------------
# FormatReportNode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFormatReportNode:
    async def test_includes_type(self):
        node = FormatReportNode()
        state = DocumentState(document="x", document_type="technical")
        result = await node.process(state)
        assert "**Type:** technical" in result.report

    async def test_includes_extracted_specs_section(self):
        node = FormatReportNode()
        state = DocumentState(
            document="x",
            document_type="technical",
            extracted_specs="POST /api/v1/auth",
        )
        result = await node.process(state)
        assert "## Extracted Specifications" in result.report
        assert "POST /api/v1/auth" in result.report

    async def test_includes_summary_section(self):
        node = FormatReportNode()
        state = DocumentState(
            document="x",
            document_type="narrative",
            summary="The board met and discussed strategy.",
        )
        result = await node.process(state)
        assert "## Summary" in result.report
        assert "board met" in result.report

    async def test_includes_validation_issues(self):
        node = FormatReportNode()
        state = DocumentState(
            document="x",
            document_type="technical",
            validation_errors=["Output too short", "Missing schema"],
        )
        result = await node.process(state)
        assert "## Validation Issues" in result.report
        assert "- Output too short" in result.report
        assert "- Missing schema" in result.report

    async def test_omits_empty_sections(self):
        node = FormatReportNode()
        state = DocumentState(document="x", document_type="unknown")
        result = await node.process(state)
        assert "## Extracted Specifications" not in result.report
        assert "## Summary" not in result.report
        assert "## Validation Issues" not in result.report


# ---------------------------------------------------------------------------
# Graph structure
# ---------------------------------------------------------------------------


class TestGraphStructure:
    def test_validates_without_error(self):
        graph = build_graph()
        graph.validate()  # should not raise

    def test_correct_entry_point(self):
        graph = build_graph()
        assert graph.entry_point == "classify"

    def test_all_expected_nodes_registered(self):
        graph = build_graph()
        expected = {"classify", "extract", "summarize", "fallback", "validate", "format_report"}
        assert set(graph.nodes.keys()) == expected

    def test_classify_has_conditional_edge(self):
        graph = build_graph()
        assert "classify" in graph.conditional_edges

    def test_linear_edges_wired(self):
        graph = build_graph()
        assert graph.edges["extract"] == "validate"
        assert graph.edges["summarize"] == "validate"
        assert graph.edges["fallback"] == "format_report"
        assert graph.edges["validate"] == "format_report"
        assert graph.edges["format_report"] == END

    def test_conditional_edge_routes_technical(self):
        graph = build_graph()
        edge_fn = graph.conditional_edges["classify"]
        state = DocumentState(document="x", document_type="technical")
        assert edge_fn(state) == "extract"

    def test_conditional_edge_routes_narrative(self):
        graph = build_graph()
        edge_fn = graph.conditional_edges["classify"]
        state = DocumentState(document="x", document_type="narrative")
        assert edge_fn(state) == "summarize"

    def test_conditional_edge_routes_unknown_to_fallback(self):
        graph = build_graph()
        edge_fn = graph.conditional_edges["classify"]
        state = DocumentState(document="x", document_type="unknown")
        assert edge_fn(state) == "fallback"


# ---------------------------------------------------------------------------
# Full workflow (mocked LLM)
#
# Strategy: create subclasses of ExtractSpecsNode/SummarizeNode with
# no-op setup/shutdown, wire in a mock LLM, and run the complete graph.
# ---------------------------------------------------------------------------


class MockExtractSpecsNode(ExtractSpecsNode):
    """ExtractSpecsNode with no-op lifecycle for testing."""

    async def setup(self):
        pass

    async def shutdown(self):
        pass


class MockSummarizeNode(SummarizeNode):
    """SummarizeNode with no-op lifecycle for testing."""

    async def setup(self):
        pass

    async def shutdown(self):
        pass


def _build_test_graph(
    extract_node: AgentNode,
    summarize_node: AgentNode,
) -> Graph:
    """Build the same graph topology as build_graph() but with injectable agent nodes."""
    graph = Graph(state_type=DocumentState)

    graph.add_node("classify", ClassifyNode())
    graph.add_node("extract", extract_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("fallback", FallbackNode())
    graph.add_node("validate", ValidateNode())
    graph.add_node("format_report", FormatReportNode())

    graph.set_entry_point("classify")

    graph.add_conditional_edge(
        "classify",
        lambda s: {
            "technical": "extract",
            "narrative": "summarize",
        }.get(s.document_type, "fallback"),
    )

    graph.add_edge("extract", "validate")
    graph.add_edge("summarize", "validate")
    graph.add_edge("fallback", "format_report")
    graph.add_edge("validate", "format_report")
    graph.add_edge("format_report", END)

    return graph


@pytest.mark.asyncio
class TestFullWorkflowTechnical:
    """Technical doc -> classify -> extract -> validate -> format_report."""

    async def test_technical_end_to_end(self):
        extract = MockExtractSpecsNode(config=_make_config())
        summarize = MockSummarizeNode(config=_make_config())
        _wire_agent(extract)
        _wire_agent(summarize)

        # Load prompts so extract.prompts.render("extract", ...) works
        prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
        extract.prompts.load_all(prompts_dir)

        extract_content = (
            "## POST /api/v1/auth/token\n\n"
            "**Parameters:** client_id, client_secret, grant_type\n"
            "**Requirements:** Token TTL 3600s, rate limit 100 req/min, TLS 1.2+"
        )
        extract.llm.call_model = AsyncMock(
            return_value=_mock_response(extract_content)
        )

        graph = _build_test_graph(extract, summarize)
        runner = WorkflowRunner(graph, max_steps=10)
        result = await runner.start(DocumentState(document=TECHNICAL_DOC))

        assert result.document_type == "technical"
        assert "POST /api/v1/auth/token" in result.extracted_specs
        assert result.validation_errors == []
        assert "## Extracted Specifications" in result.report
        assert result.summary == ""  # summarize path not taken

    async def test_technical_with_validation_failure(self):
        extract = MockExtractSpecsNode(config=_make_config())
        summarize = MockSummarizeNode(config=_make_config())
        _wire_agent(extract)
        _wire_agent(summarize)

        prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
        extract.prompts.load_all(prompts_dir)

        # Return insufficient output to trigger validation error
        extract.llm.call_model = AsyncMock(
            return_value=_mock_response("N/A")
        )

        graph = _build_test_graph(extract, summarize)
        runner = WorkflowRunner(graph, max_steps=10)
        result = await runner.start(DocumentState(document=TECHNICAL_DOC))

        assert result.document_type == "technical"
        assert len(result.validation_errors) == 1
        assert "## Validation Issues" in result.report


@pytest.mark.asyncio
class TestFullWorkflowNarrative:
    """Narrative doc -> classify -> summarize -> validate -> format_report."""

    async def test_narrative_end_to_end(self):
        extract = MockExtractSpecsNode(config=_make_config())
        summarize = MockSummarizeNode(config=_make_config())
        _wire_agent(extract)
        _wire_agent(summarize)

        prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
        summarize.prompts.load_all(prompts_dir)

        summary_content = (
            "- AI is central to the company's growth strategy\n"
            "- AI revenue grew 42% year-over-year\n"
            "- EU regulatory risks flagged by board members\n"
            "- Company's proactive AI approach seen as a competitive advantage"
        )
        summarize.llm.call_model = AsyncMock(
            return_value=_mock_response(summary_content)
        )

        graph = _build_test_graph(extract, summarize)
        runner = WorkflowRunner(graph, max_steps=10)
        result = await runner.start(DocumentState(document=NARRATIVE_DOC))

        assert result.document_type == "narrative"
        assert "42%" in result.summary
        assert result.validation_errors == []
        assert "## Summary" in result.report
        assert result.extracted_specs == ""  # extract path not taken


@pytest.mark.asyncio
class TestFullWorkflowUnknown:
    """Unknown doc -> classify -> fallback -> format_report (skips validate)."""

    async def test_unknown_end_to_end(self):
        extract = MockExtractSpecsNode(config=_make_config())
        summarize = MockSummarizeNode(config=_make_config())
        _wire_agent(extract)
        _wire_agent(summarize)

        graph = _build_test_graph(extract, summarize)
        runner = WorkflowRunner(graph, max_steps=10)
        short_doc = "Hello world."
        result = await runner.start(DocumentState(document=short_doc))

        assert result.document_type == "unknown"
        assert "could not be classified" in result.summary
        assert str(len(short_doc)) in result.summary
        assert result.extracted_specs == ""
        # Fallback path skips validation
        assert result.validation_errors == []
        assert "# Document Analysis Report" in result.report
