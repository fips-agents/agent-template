"""Run the document analysis workflow with full per-node tracing.

Wraps each node's process() method to capture state snapshots before and
after execution, then writes the complete trace to outputs/ as JSON.

Usage:
    PYTHONPATH=src python trace.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from agent import (
    NARRATIVE_DOC,
    TECHNICAL_DOC,
    DocumentState,
    WorkflowRunner,
    build_graph,
)

OUTPUT_DIR = Path(__file__).parent / "outputs"


@dataclass
class NodeTrace:
    node_name: str
    input_state: dict
    output_state: dict
    duration_ms: float
    error: str | None = None


@dataclass
class WorkflowTrace:
    document_name: str
    document_type_detected: str
    nodes_executed: list[NodeTrace] = field(default_factory=list)
    final_report: str = ""
    total_duration_ms: float = 0.0


def _wrap_nodes_for_tracing(graph, trace: WorkflowTrace) -> None:
    """Monkey-patch each node's process() to record before/after state."""
    for name, node in graph.nodes.items():
        original_process = node.process

        async def traced_process(state, _name=name, _orig=original_process):
            input_snapshot = state.model_dump()
            start = time.monotonic()
            error = None
            try:
                result = await _orig(state)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                duration_ms = (time.monotonic() - start) * 1000
                output_snapshot = result.model_dump() if error is None else input_snapshot
                trace.nodes_executed.append(NodeTrace(
                    node_name=_name,
                    input_state=input_snapshot,
                    output_state=output_snapshot,
                    duration_ms=round(duration_ms, 2),
                    error=error,
                ))

            return result

        node.process = traced_process


async def run_traced(name: str, document: str) -> WorkflowTrace:
    """Run a single document through the workflow with full tracing."""
    trace = WorkflowTrace(document_name=name, document_type_detected="")

    graph = build_graph()
    _wrap_nodes_for_tracing(graph, trace)

    runner = WorkflowRunner(graph, max_steps=10)

    start = time.monotonic()
    result = await runner.start(DocumentState(document=document))
    trace.total_duration_ms = round((time.monotonic() - start) * 1000, 2)

    trace.document_type_detected = result.document_type
    trace.final_report = result.report

    return trace


def trace_to_dict(trace: WorkflowTrace) -> dict:
    """Convert a WorkflowTrace to a JSON-serializable dict."""
    return {
        "document_name": trace.document_name,
        "document_type_detected": trace.document_type_detected,
        "total_duration_ms": trace.total_duration_ms,
        "nodes_executed": [
            {
                "step": i + 1,
                "node_name": nt.node_name,
                "duration_ms": nt.duration_ms,
                "error": nt.error,
                "state_before": nt.input_state,
                "state_after": nt.output_state,
            }
            for i, nt in enumerate(trace.nodes_executed)
        ],
        "final_report": trace.final_report,
    }


UNKNOWN_DOC = "Hello world"


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")

    OUTPUT_DIR.mkdir(exist_ok=True)

    documents = [
        ("technical", TECHNICAL_DOC),
        ("narrative", NARRATIVE_DOC),
        ("unknown", UNKNOWN_DOC),
    ]

    for name, doc in documents:
        print(f"\n{'=' * 60}")
        print(f"Tracing: {name}")
        print(f"{'=' * 60}")

        trace = await run_traced(name, doc)
        output = trace_to_dict(trace)

        out_path = OUTPUT_DIR / f"trace_{name}.json"
        out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
        print(f"  Type detected: {trace.document_type_detected}")
        print(f"  Nodes executed: {[nt.node_name for nt in trace.nodes_executed]}")
        print(f"  Duration: {trace.total_duration_ms:.0f}ms")
        print(f"  Trace written to: {out_path}")

        # Also write the final report as readable markdown
        report_path = OUTPUT_DIR / f"report_{name}.md"
        report_path.write_text(trace.final_report)
        print(f"  Report written to: {report_path}")

    print(f"\nAll traces written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
