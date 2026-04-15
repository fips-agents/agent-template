"""Tests for sandbox.profiles — profile loading and inheritance."""

import pytest

from sandbox.profiles import get_active_profile, load_profile


# ---------------------------------------------------------------------------
# Minimal profile
# ---------------------------------------------------------------------------

_MINIMAL_EXPECTED_IMPORTS = frozenset(
    {
        "math",
        "statistics",
        "itertools",
        "functools",
        "re",
        "datetime",
        "collections",
        "json",
        "csv",
        "string",
        "textwrap",
        "decimal",
        "fractions",
        "random",
        "operator",
        "typing",
    }
)


def test_minimal_name():
    profile = load_profile("minimal")
    assert profile.name == "minimal", f"unexpected name: {profile.name!r}"


def test_minimal_allowed_imports():
    profile = load_profile("minimal")
    assert profile.allowed_imports == _MINIMAL_EXPECTED_IMPORTS, (
        f"unexpected imports: {profile.allowed_imports}"
    )


def test_minimal_blocklist_empty():
    profile = load_profile("minimal")
    assert profile.blocklist == [], (
        f"expected empty blocklist, got: {profile.blocklist}"
    )


def test_minimal_pre_stages_contains_ast_scan():
    profile = load_profile("minimal")
    assert "ast_scan" in profile.scan_stages.pre, (
        f"expected 'ast_scan' in pre stages: {profile.scan_stages.pre}"
    )


def test_minimal_resources_memory():
    profile = load_profile("minimal")
    assert profile.resources.memory == "256Mi", (
        f"unexpected memory: {profile.resources.memory!r}"
    )


# ---------------------------------------------------------------------------
# Data-science profile
# ---------------------------------------------------------------------------

_DS_EXTRA_IMPORTS = {"numpy", "pandas", "scipy"}


def test_data_science_name():
    profile = load_profile("data-science")
    assert profile.name == "data-science", f"unexpected name: {profile.name!r}"


def test_data_science_inherits_minimal_imports():
    profile = load_profile("data-science")
    assert _MINIMAL_EXPECTED_IMPORTS.issubset(profile.allowed_imports), (
        "data-science profile should include all minimal imports"
    )


def test_data_science_adds_extra_imports():
    profile = load_profile("data-science")
    assert _DS_EXTRA_IMPORTS.issubset(profile.allowed_imports), (
        f"data-science profile missing extra imports: {_DS_EXTRA_IMPORTS - profile.allowed_imports}"
    )


def test_data_science_total_import_count():
    profile = load_profile("data-science")
    expected_count = len(_MINIMAL_EXPECTED_IMPORTS) + len(_DS_EXTRA_IMPORTS)
    assert len(profile.allowed_imports) == expected_count, (
        f"expected {expected_count} imports, got {len(profile.allowed_imports)}: "
        f"{profile.allowed_imports}"
    )


def test_data_science_blocklist_nonempty():
    profile = load_profile("data-science")
    assert len(profile.blocklist) > 0, "data-science blocklist should not be empty"


def test_data_science_pre_stages():
    profile = load_profile("data-science")
    assert "ast_scan" in profile.scan_stages.pre, (
        f"expected 'ast_scan' in pre stages: {profile.scan_stages.pre}"
    )
    assert "blocklist_audit" in profile.scan_stages.pre, (
        f"expected 'blocklist_audit' in pre stages: {profile.scan_stages.pre}"
    )


@pytest.mark.parametrize(
    "module,attr",
    [
        pytest.param("numpy", "ctypeslib", id="numpy_ctypeslib"),
        pytest.param("pandas", "read_pickle", id="pandas_read_pickle"),
        pytest.param("pandas", "read_sql", id="pandas_read_sql"),
    ],
)
def test_data_science_blocklist_entries(module, attr):
    profile = load_profile("data-science")
    assert (module, attr) in profile.blocklist, (
        f"expected ({module!r}, {attr!r}) in blocklist: {profile.blocklist}"
    )


def test_data_science_resources_memory():
    profile = load_profile("data-science")
    assert profile.resources.memory == "512Mi", (
        f"unexpected memory: {profile.resources.memory!r}"
    )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_nonexistent_profile_raises():
    with pytest.raises(FileNotFoundError, match="nonexistent"):
        load_profile("nonexistent")


# ---------------------------------------------------------------------------
# get_active_profile
# ---------------------------------------------------------------------------


def test_get_active_profile_defaults_to_minimal(monkeypatch):
    monkeypatch.delenv("SANDBOX_PROFILE", raising=False)
    profile = get_active_profile()
    assert profile.name == "minimal", f"expected 'minimal', got: {profile.name!r}"


def test_get_active_profile_reads_env_var(monkeypatch):
    monkeypatch.setenv("SANDBOX_PROFILE", "data-science")
    profile = get_active_profile()
    assert profile.name == "data-science", f"expected 'data-science', got: {profile.name!r}"
