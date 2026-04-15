"""Multi-stage code execution pipeline.

Orchestrates pre-execution validation stages, code execution, and
post-execution review stages.  Each profile defines which stages run.

Usage::

    from sandbox.pipeline import run_pipeline
    from sandbox.profiles import get_active_profile

    profile = get_active_profile()
    result = await run_pipeline(code, profile, timeout=10.0)
    if result.rejected:
        # surface result.violations to the LLM
        ...
"""

from __future__ import annotations

import dataclasses

from sandbox.executor import ExecutionResult, execute_code
from sandbox.guardrails import blocklist_audit, validate_code
from sandbox.profiles import Profile


@dataclasses.dataclass
class PipelineResult:
    """Outcome of the code execution pipeline."""

    rejected: bool
    violations: list[str] = dataclasses.field(default_factory=list)
    result: ExecutionResult | None = None


# Registry of pre-execution stage functions.
# Each takes (source, profile) and returns a list of violation strings.
_PRE_STAGES: dict[str, object] = {
    "ast_scan": lambda source, profile: validate_code(
        source, profile.allowed_imports,
    ),
    "blocklist_audit": lambda source, profile: blocklist_audit(
        source, profile.blocklist,
    ),
}

# Registry of post-execution stage functions.
# Each takes (source, result, profile) and returns a list of violation strings.
_POST_STAGES: dict[str, object] = {}


async def run_pipeline(
    code: str,
    profile: Profile,
    timeout: float = 10.0,
) -> PipelineResult:
    """Run the full code execution pipeline for the given profile.

    Stages execute in order.  The pipeline short-circuits on the first
    stage that returns violations.

    Args:
        code: Python source code to validate and execute.
        profile: Active sandbox profile (determines which stages run).
        timeout: Execution timeout in seconds (capped by profile max).

    Returns:
        A :class:`PipelineResult` indicating whether the code was rejected
        or executed, with violations or execution output.
    """
    timeout = min(timeout, profile.resources.timeout_max)

    # Pre-execution stages.
    for stage_name in profile.scan_stages.pre:
        stage_fn = _PRE_STAGES.get(stage_name)
        if stage_fn is None:
            return PipelineResult(
                rejected=True,
                violations=[f"Unknown pre-execution stage: {stage_name}"],
            )
        violations = stage_fn(code, profile)
        if violations:
            return PipelineResult(rejected=True, violations=violations)

    # Execute.
    result = await execute_code(code, timeout)

    # Post-execution stages.
    for stage_name in profile.scan_stages.post:
        stage_fn = _POST_STAGES.get(stage_name)
        if stage_fn is None:
            continue
        violations = stage_fn(code, result, profile)
        if violations:
            return PipelineResult(
                rejected=True, violations=violations, result=result,
            )

    return PipelineResult(rejected=False, result=result)
