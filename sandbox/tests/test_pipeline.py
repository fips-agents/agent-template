"""Tests for sandbox.pipeline.run_pipeline and sandbox.guardrails.blocklist_audit."""

import textwrap

import pytest

from sandbox.guardrails import blocklist_audit
from sandbox.pipeline import run_pipeline
from sandbox.profiles import load_profile


# ---------------------------------------------------------------------------
# blocklist_audit — unit tests
# ---------------------------------------------------------------------------


def test_blocklist_audit_empty_blocklist():
    violations = blocklist_audit("import numpy; numpy.array([1, 2])", [])
    assert violations == [], f"empty blocklist should produce no violations: {violations}"


def test_blocklist_audit_catches_numpy_ctypeslib():
    source = "numpy.ctypeslib.as_array(ptr)"
    violations = blocklist_audit(source, [("numpy", "ctypeslib")])
    assert len(violations) >= 1, f"expected violation for numpy.ctypeslib: {violations}"
    assert any("numpy.ctypeslib" in v for v in violations), violations


def test_blocklist_audit_catches_pandas_read_pickle():
    source = "df = pandas.read_pickle('data.pkl')"
    violations = blocklist_audit(source, [("pandas", "read_pickle")])
    assert len(violations) >= 1, f"expected violation for pandas.read_pickle: {violations}"
    assert any("pandas.read_pickle" in v for v in violations), violations


def test_blocklist_audit_allows_pandas_dataframe():
    source = "df = pandas.DataFrame({'a': [1, 2]})"
    violations = blocklist_audit(source, [("pandas", "read_pickle"), ("pandas", "read_sql")])
    assert violations == [], (
        f"pandas.DataFrame should not be blocked, got: {violations}"
    )


def test_blocklist_audit_instance_method_not_caught():
    """Instance-level method calls (df.to_pickle) cannot be caught by the
    blocklist audit because AST analysis sees the variable name ('df'), not
    the type ('DataFrame').  This test documents the limitation."""
    source = "df.to_pickle('output.pkl')"
    violations = blocklist_audit(source, [("DataFrame", "to_pickle")])
    assert violations == [], (
        "blocklist_audit uses AST names, not types — instance methods "
        "are not caught (by design)"
    )


# ---------------------------------------------------------------------------
# run_pipeline — minimal profile
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_pipeline_minimal_allowed_code():
    profile = load_profile("minimal")
    result = await run_pipeline("import math; x = math.sqrt(4)", profile, timeout=5.0)
    assert not result.rejected, (
        f"allowed code should not be rejected, violations: {result.violations}"
    )
    assert result.result is not None
    assert result.result.exit_code == 0


@pytest.mark.asyncio
async def test_run_pipeline_minimal_blocked_import():
    profile = load_profile("minimal")
    result = await run_pipeline("import os", profile, timeout=5.0)
    assert result.rejected, "blocked import should be rejected"
    assert any("os" in v for v in result.violations), (
        f"expected 'os' in violation messages: {result.violations}"
    )


# ---------------------------------------------------------------------------
# run_pipeline — data-science profile
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_pipeline_data_science_numpy_allowed():
    profile = load_profile("data-science")
    # numpy is in the allowed imports — ast_scan should pass
    source = textwrap.dedent("""\
        import numpy
        arr = numpy.array([1, 2, 3])
    """)
    result = await run_pipeline(source, profile, timeout=5.0)
    # numpy may or may not be installed in the test environment; we care that
    # the *guardrail* passes, not that numpy runs successfully at runtime.
    # A rejected result means guardrails blocked it — that's the failure case.
    assert not result.rejected, (
        f"numpy import should not be rejected by guardrails, violations: {result.violations}"
    )


@pytest.mark.asyncio
async def test_run_pipeline_data_science_numpy_ctypeslib_rejected():
    profile = load_profile("data-science")
    source = textwrap.dedent("""\
        import numpy
        ptr = numpy.ctypeslib.as_array(None)
    """)
    result = await run_pipeline(source, profile, timeout=5.0)
    assert result.rejected, "numpy.ctypeslib should be rejected by blocklist_audit"
    assert any("ctypeslib" in v for v in result.violations), (
        f"expected 'ctypeslib' in violation messages: {result.violations}"
    )


@pytest.mark.asyncio
async def test_run_pipeline_data_science_subprocess_rejected():
    profile = load_profile("data-science")
    result = await run_pipeline("import subprocess", profile, timeout=5.0)
    assert result.rejected, "subprocess should still be rejected by data-science profile"
    assert any("subprocess" in v for v in result.violations), (
        f"expected 'subprocess' in violation messages: {result.violations}"
    )


# ---------------------------------------------------------------------------
# run_pipeline — unknown stage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_pipeline_unknown_stage_returns_error(monkeypatch):
    """A profile with an unknown pre-execution stage name should return a rejection."""
    from sandbox.profiles import Profile, ProfileResources, ScanStages

    bad_profile = Profile(
        name="test-bad-stage",
        allowed_imports=frozenset({"math"}),
        blocklist=[],
        resources=ProfileResources(),
        scan_stages=ScanStages(pre=["ast_scan", "does_not_exist"], post=[]),
    )
    result = await run_pipeline("import math", bad_profile, timeout=5.0)
    assert result.rejected, "unknown stage should produce a rejected result"
    assert any("does_not_exist" in v for v in result.violations), (
        f"expected stage name in violation messages: {result.violations}"
    )
