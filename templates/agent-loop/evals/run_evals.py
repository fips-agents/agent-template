"""Lightweight eval runner for BaseAgent agents.

Loads eval cases from ``evals.yaml``, creates an agent instance, runs each
case through the agent's ``step()`` method, checks assertions against the
output, and prints a pass/fail report.

Usage::

    # Dry-run — list cases without executing
    python -m evals.run_evals --dry-run

    # Run all cases (mock LLM, default)
    python -m evals.run_evals

    # Run a single case
    python -m evals.run_evals --case basic_research_query

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
import importlib
import inspect
import json
import logging
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import yaml

# ---------------------------------------------------------------------------
# Resolve paths relative to the template root (one level up from evals/)
# ---------------------------------------------------------------------------

_EVALS_DIR = Path(__file__).resolve().parent
_TEMPLATE_ROOT = _EVALS_DIR.parent
_FIXTURES_DIR = _EVALS_DIR / "fixtures"

# Ensure the src/ directory is importable.
sys.path.insert(0, str(_TEMPLATE_ROOT / "src"))
sys.path.insert(0, str(_TEMPLATE_ROOT))

from base_agent.agent import BaseAgent, StepOutcome  # noqa: E402
from base_agent.config import AgentConfig, BackoffConfig, LLMConfig, LoopConfig  # noqa: E402
from base_agent.llm import LLMClient, ModelResponse  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dynamic agent / model discovery
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _discover_agent_class() -> type:
    """Find the single BaseAgent subclass in agent.py.

    Raises RuntimeError with an actionable message when zero or multiple
    subclasses are found.
    """
    agent_module = importlib.import_module("agent")
    candidates = [
        obj
        for _name, obj in inspect.getmembers(agent_module, inspect.isclass)
        if issubclass(obj, BaseAgent) and obj is not BaseAgent
    ]
    if len(candidates) == 0:
        raise RuntimeError(
            "No BaseAgent subclass found in agent.py. "
            "Run /create-agent to generate your agent."
        )
    if len(candidates) > 1:
        names = [c.__name__ for c in candidates]
        raise RuntimeError(
            f"Multiple BaseAgent subclasses in agent.py: {names}. "
            "The eval runner expects exactly one."
        )
    return candidates[0]


@lru_cache(maxsize=1)
def _discover_output_model() -> type | None:
    """Find a Pydantic BaseModel subclass in agent.py for structured output.

    Returns None if the agent does not define one (i.e. does not use
    structured output).
    """
    from pydantic import BaseModel as PydanticBaseModel

    agent_module = importlib.import_module("agent")
    candidates = [
        obj
        for _name, obj in inspect.getmembers(agent_module, inspect.isclass)
        if issubclass(obj, PydanticBaseModel) and obj is not PydanticBaseModel
    ]
    if not candidates:
        return None
    # Most agents define a single output schema; return the first.
    return candidates[0]


def _build_mock_instance(model_class: type) -> Any:
    """Create a plausible mock instance of a Pydantic model from its schema.

    Uses ``model_json_schema()`` to inspect fields and populate them with
    type-appropriate placeholder values.
    """
    schema = model_class.model_json_schema()
    props = schema.get("properties", {})
    mock_data: dict[str, Any] = {}
    for field_name, field_schema in props.items():
        field_type = field_schema.get("type", "string")
        if field_type == "string":
            mock_data[field_name] = f"Mock {field_name} for eval"
        elif field_type in ("number", "integer"):
            mock_data[field_name] = 0.85
        elif field_type == "boolean":
            mock_data[field_name] = True
        elif field_type == "array":
            mock_data[field_name] = ["https://example.com/eval-source"]
        else:
            mock_data[field_name] = f"mock_{field_name}"
    return model_class(**mock_data)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Assertion:
    """A single check to run against agent output."""

    type: str
    params: dict[str, Any] = field(default_factory=dict)


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
class AssertionResult:
    """Outcome of checking a single assertion."""

    assertion: Assertion
    passed: bool
    detail: str = ""


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
# Mock LLM factory
# ---------------------------------------------------------------------------


def _build_mock_litellm_response(
    content: str | None = None,
    tool_calls: list[Any] | None = None,
) -> Any:
    """Construct a fake litellm response matching ModelResponse expectations."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _make_tool_call_obj(name: str, arguments: dict[str, Any]) -> Any:
    """Build a fake tool-call object in OpenAI format."""
    return SimpleNamespace(
        id=f"call_eval_{name}",
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments),
        ),
    )


def _build_mock_responses(
    query: str,
    fixture_data: dict[str, Any] | None = None,
) -> tuple[list[Any], Any | None, str]:
    """Produce the sequence of mock LLM responses for a single eval case.

    Returns (call_model_side_effects, report_or_none, validation_text).

    The report is built from whatever Pydantic model the agent defines in
    agent.py.  If no model is found (the agent does not use structured
    output), *report* is None and callers should skip call_model_json
    mocking.
    """
    # Detect multi-step queries (comparisons, "vs", multiple topics).
    multi_step_keywords = {"compare", "vs", "versus", "difference", "between"}
    is_multi_step = any(kw in query.lower() for kw in multi_step_keywords)

    side_effects: list[Any] = []

    if is_multi_step:
        # Simulate multiple search rounds.
        tc1 = _make_tool_call_obj("web_search", {"query": query})
        side_effects.append(
            ModelResponse(_build_mock_litellm_response(tool_calls=[tc1]))
        )
        tc2 = _make_tool_call_obj(
            "web_search", {"query": f"{query} detailed comparison"}
        )
        side_effects.append(
            ModelResponse(_build_mock_litellm_response(tool_calls=[tc2]))
        )
        side_effects.append(
            ModelResponse(
                _build_mock_litellm_response(
                    content=f"After thorough research on '{query}', here are the findings."
                )
            )
        )
    else:
        # Single search round.
        search_tc = _make_tool_call_obj("web_search", {"query": query})
        side_effects.append(
            ModelResponse(_build_mock_litellm_response(tool_calls=[search_tc]))
        )
        side_effects.append(
            ModelResponse(
                _build_mock_litellm_response(
                    content=f"Based on my research about '{query}', here are the findings."
                )
            )
        )

    # Build a mock report from the agent's Pydantic output model.
    output_model = _discover_output_model()
    report: Any | None = None
    if output_model is not None:
        report = _build_mock_instance(output_model)

    validation_text = "The report addresses the query."

    return side_effects, report, validation_text


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


async def create_agent(*, use_real_llm: bool = False) -> Any:
    """Create an agent instance by discovering the BaseAgent subclass in agent.py.

    When *use_real_llm* is False (the default), the LLM client is replaced
    with mocks so evals run without a live model endpoint.
    """
    agent_cls = _discover_agent_class()

    config = AgentConfig(
        model=LLMConfig(
            endpoint="http://eval-mock:8321/v1",
            name="eval-mock-model",
            temperature=0.0,
            max_tokens=1024,
        ),
        loop=LoopConfig(
            max_iterations=5,
            backoff=BackoffConfig(initial=0.01, max=0.05, multiplier=2.0),
        ),
    )
    agent = agent_cls(config=config, base_dir=_TEMPLATE_ROOT)

    if use_real_llm:
        await agent.setup()
    else:
        # Partial setup: real tools/prompts/rules/skills, mock LLM.
        agent.config = config
        agent.llm = MagicMock(spec=LLMClient)
        agent.tools.discover(_TEMPLATE_ROOT / "tools")
        prompts_dir = _TEMPLATE_ROOT / "prompts"
        if prompts_dir.is_dir():
            agent.prompts.load_all(prompts_dir)
        rules_dir = _TEMPLATE_ROOT / "rules"
        if rules_dir.is_dir():
            agent.rules.load_all(rules_dir)
        skills_dir = _TEMPLATE_ROOT / "skills"
        if skills_dir.is_dir():
            agent.skills.load_all(skills_dir)
        agent._setup_done = True

    return agent


# ---------------------------------------------------------------------------
# Assertion checkers
# ---------------------------------------------------------------------------


def check_assertion(
    assertion: Assertion,
    result: Any,
    tool_calls_log: list[str],
) -> AssertionResult:
    """Evaluate one assertion against the agent's output.

    *result* is expected to be a Pydantic model or ``None`` if the agent
    failed.
    """
    atype = assertion.type
    params = assertion.params

    if result is None:
        return AssertionResult(
            assertion=assertion,
            passed=False,
            detail="Agent returned no result",
        )

    if atype == "field_exists":
        fld = params["field"]
        exists = hasattr(result, fld) and getattr(result, fld) is not None
        return AssertionResult(
            assertion=assertion,
            passed=exists,
            detail=f"field '{fld}' {'exists' if exists else 'missing'}",
        )

    if atype == "contains":
        fld = params["field"]
        expected = params["value"]
        actual = str(getattr(result, fld, ""))
        found = expected.lower() in actual.lower()
        return AssertionResult(
            assertion=assertion,
            passed=found,
            detail=(
                f"'{expected}' {'found' if found else 'not found'} "
                f"in {fld} (length {len(actual)})"
            ),
        )

    if atype == "not_contains":
        fld = params["field"]
        unexpected = params["value"]
        actual = str(getattr(result, fld, ""))
        absent = unexpected.lower() not in actual.lower()
        return AssertionResult(
            assertion=assertion,
            passed=absent,
            detail=(
                f"'{unexpected}' {'absent' if absent else 'present'} "
                f"in {fld}"
            ),
        )

    if atype == "field_gte":
        fld = params["field"]
        threshold = float(params["value"])
        actual = float(getattr(result, fld, 0))
        ok = actual >= threshold
        return AssertionResult(
            assertion=assertion,
            passed=ok,
            detail=f"{fld}={actual} {'>=':} {threshold}",
        )

    if atype == "field_lte":
        fld = params["field"]
        threshold = float(params["value"])
        actual = float(getattr(result, fld, 0))
        ok = actual <= threshold
        return AssertionResult(
            assertion=assertion,
            passed=ok,
            detail=f"{fld}={actual} {'<=':} {threshold}",
        )

    if atype == "tool_called":
        tool_name = params["tool"]
        min_calls = int(params.get("min_calls", 1))
        count = tool_calls_log.count(tool_name)
        ok = count >= min_calls
        return AssertionResult(
            assertion=assertion,
            passed=ok,
            detail=(
                f"tool '{tool_name}' called {count} time(s), "
                f"expected >= {min_calls}"
            ),
        )

    if atype == "custom":
        return AssertionResult(
            assertion=assertion,
            passed=False,
            detail=(
                "Custom assertions must be registered via an external "
                "eval harness.  Skipping in the built-in runner."
            ),
        )

    return AssertionResult(
        assertion=assertion,
        passed=False,
        detail=f"Unknown assertion type: {atype}",
    )


# ---------------------------------------------------------------------------
# Case runner
# ---------------------------------------------------------------------------


async def run_case(
    case: EvalCase,
    *,
    use_real_llm: bool = False,
) -> CaseResult:
    """Execute a single eval case and return its result."""
    tool_calls_log: list[str] = []

    try:
        agent = await create_agent(use_real_llm=use_real_llm)
        agent.add_message("user", case.input)

        if not use_real_llm:
            # Wire up mock responses.
            side_effects, report, validation_text = _build_mock_responses(
                case.input
            )
            agent.llm.call_model = AsyncMock(side_effect=side_effects)
            if report is not None:
                agent.llm.call_model_json = AsyncMock(return_value=report)
            agent.llm.call_model_validated = AsyncMock(
                return_value=validation_text
            )

        # Run one step.
        step_result = await agent.step()

        # Collect tool call information from mock interactions.
        if not use_real_llm and agent.llm.call_model.call_count > 0:
            # The first mock response contains a web_search tool call.
            # Record it so tool_called assertions work.
            for resp in side_effects:
                if resp.tool_calls:
                    for tc in resp.tool_calls:
                        tool_calls_log.append(tc.function.name)

        # Evaluate assertions.
        output = step_result.result if step_result else None
        assertion_results = [
            check_assertion(a, output, tool_calls_log)
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
        description="Run eval cases for the agent.",
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
