"""Lightweight eval runner for workflow agents.

Loads eval cases from ``evals.yaml``, builds the workflow graph, runs each
case through the WorkflowRunner, checks assertions against the output state,
and prints a pass/fail report.

Usage::

    # Dry-run — list cases without executing
    python -m evals.run_evals --dry-run

    # Run all cases (mock LLM, default)
    python -m evals.run_evals

    # Run a single case
    python -m evals.run_evals --case simple_query_direct_summarize

    # Run with a real LLM (requires configured endpoint)
    python -m evals.run_evals --real-llm

    # Filter by tag
    python -m evals.run_evals --tag smoke

The runner is intentionally minimal.  It handles the assertion types defined
in ``evals.yaml`` and is designed to be replaced or augmented by external
eval frameworks (Braintrust, Promptfoo, etc.) that consume the same YAML.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import yaml

from evals import _EVALS_DIR, _FIXTURES_DIR
from evals.assertions import Assertion, AssertionResult, check_assertion
from evals.discovery import _discover_build_graph, _discover_state_class
from evals.mock_factory import _build_mock_responses, create_mock_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EvalCase:
    """One eval case loaded from evals.yaml."""

    name: str
    description: str
    input: str
    expected_behavior: str
    tags: list[str] = field(default_factory=list)
    assertions: list[Assertion] = field(default_factory=list)


@dataclass
class CaseResult:
    """Outcome of running one eval case."""

    case: EvalCase
    passed: bool
    skipped: bool = False
    error: str | None = None
    assertion_results: list[AssertionResult] = field(default_factory=list)
    tool_calls_log: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def load_eval_cases(path: Path | None = None) -> list[EvalCase]:
    """Load eval cases from a YAML file."""
    yaml_path = path or (_EVALS_DIR / "evals.yaml")
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    cases: list[EvalCase] = []
    for entry in raw.get("cases", []):
        assertions = [
            Assertion(type=a.pop("type"), params=a)
            for a in (entry.get("assertions") or [])
        ]
        cases.append(
            EvalCase(
                name=entry["name"],
                description=entry.get("description", ""),
                input=entry["input"],
                expected_behavior=entry.get("expected_behavior", ""),
                tags=entry.get("tags", []),
                assertions=assertions,
            )
        )
    return cases


# ---------------------------------------------------------------------------
# Fixture loader
# ---------------------------------------------------------------------------


def load_fixture(name: str) -> Any:
    """Load a JSON fixture file from the fixtures/ directory."""
    path = _FIXTURES_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Fixture not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Case runner
# ---------------------------------------------------------------------------


async def run_case(
    case: EvalCase,
    *,
    use_real_llm: bool = False,
) -> CaseResult:
    """Execute a single eval case and return its result.

    Builds the workflow graph, creates initial state from the case input,
    and runs the workflow through the WorkflowRunner.
    """
    tool_calls_log: list[str] = []

    try:
        from workflow import WorkflowRunner

        build_graph = _discover_build_graph()
        state_cls = _discover_state_class()

        graph = build_graph()
        runner = WorkflowRunner(graph, max_steps=10)

        # Create initial state from the eval case input.
        initial_state = state_cls(query=case.input)

        if use_real_llm:
            result_state = await runner.start(initial_state)
        else:
            # For mock mode, we run the workflow but mock the LLM calls
            # on any AgentNode instances.
            side_effects, summary_text = _build_mock_responses(case.input)

            # Patch AgentNode instances in the graph to use mock LLM.
            from workflow.agent_node import AgentNode
            from fipsagents.baseagent.llm import LLMClient

            effect_iter = iter(side_effects)
            for node_obj in graph._nodes.values():
                if isinstance(node_obj, AgentNode):
                    node_obj._config = create_mock_config()
                    node_obj.llm = MagicMock(spec=LLMClient)
                    node_obj.llm.call_model = AsyncMock(
                        side_effect=lambda *a, **kw: next(effect_iter, side_effects[-1])
                    )
                    node_obj.llm.call_model_json = AsyncMock(return_value=None)
                    node_obj.llm.call_model_validated = AsyncMock(
                        return_value="Validation passed."
                    )
                    node_obj._setup_done = True

            result_state = await runner.start(initial_state)

            # Collect tool calls from mock side effects.
            for resp in side_effects:
                if hasattr(resp, "tool_calls") and resp.tool_calls:
                    for tc in resp.tool_calls:
                        tool_calls_log.append(tc.function.name)

        # Evaluate assertions against the final state.
        assertion_results = [
            check_assertion(a, result_state, tool_calls_log)
            for a in case.assertions
        ]
        all_passed = all(ar.passed for ar in assertion_results)

        return CaseResult(
            case=case,
            passed=all_passed,
            assertion_results=assertion_results,
            tool_calls_log=tool_calls_log,
        )

    except Exception as exc:
        return CaseResult(
            case=case,
            passed=False,
            error=f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------


def print_report(results: list[CaseResult]) -> None:
    """Print a human-readable eval report to stdout."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed and not r.skipped)
    skipped = sum(1 for r in results if r.skipped)

    print()
    print("=" * 60)
    print("  EVAL RESULTS")
    print("=" * 60)

    for r in results:
        if r.skipped:
            status = "SKIP"
        elif r.passed:
            status = "PASS"
        else:
            status = "FAIL"

        print(f"\n  [{status}] {r.case.name}")
        if r.case.tags:
            print(f"         tags: {', '.join(r.case.tags)}")

        if r.error:
            print(f"         error: {r.error}")

        for ar in r.assertion_results:
            mark = "ok" if ar.passed else "FAIL"
            print(f"         [{mark}] {ar.assertion.type}: {ar.detail}")

    print()
    print("-" * 60)
    print(f"  Total: {total}  Passed: {passed}  Failed: {failed}  Skipped: {skipped}")
    print("-" * 60)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run eval cases for the workflow.",
        prog="python -m evals.run_evals",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List eval cases without executing them.",
    )
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="Run only the named eval case.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Run only cases matching this tag.",
    )
    parser.add_argument(
        "--real-llm",
        action="store_true",
        help="Use a real LLM endpoint instead of mocks.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--evals-file",
        type=str,
        default=None,
        help="Path to evals YAML file (default: evals/evals.yaml).",
    )
    return parser


async def async_main(args: argparse.Namespace) -> int:
    """Run evals and return an exit code (0 = all passed)."""
    evals_path = Path(args.evals_file) if args.evals_file else None
    cases = load_eval_cases(evals_path)

    # Filter by --case.
    if args.case:
        cases = [c for c in cases if c.name == args.case]
        if not cases:
            print(f"No eval case named '{args.case}' found.", file=sys.stderr)
            return 1

    # Filter by --tag.
    if args.tag:
        cases = [c for c in cases if args.tag in c.tags]
        if not cases:
            print(f"No eval cases with tag '{args.tag}' found.", file=sys.stderr)
            return 1

    # Dry run: just list.
    if args.dry_run:
        print(f"\n  {len(cases)} eval case(s):\n")
        for c in cases:
            tags = f"  [{', '.join(c.tags)}]" if c.tags else ""
            print(f"    - {c.name}{tags}")
            print(f"      {c.description.strip()[:80]}")
        print()
        return 0

    # Execute.
    results: list[CaseResult] = []
    for case in cases:
        result = await run_case(case, use_real_llm=args.real_llm)
        results.append(result)

    print_report(results)

    return 0 if all(r.passed for r in results) else 1


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    exit_code = asyncio.run(async_main(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
