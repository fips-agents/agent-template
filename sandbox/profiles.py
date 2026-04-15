"""Profile loader for the code execution sandbox.

Reads YAML profile configs from ``sandbox/profiles/`` and resolves
inheritance (``extends`` field).  The active profile is selected via the
``SANDBOX_PROFILE`` environment variable (default: ``minimal``).

Usage::

    from sandbox.profiles import get_active_profile

    profile = get_active_profile()  # reads SANDBOX_PROFILE env var
    profile.allowed_imports         # frozenset of module names
    profile.blocklist               # list of (module, attr) tuples
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


_PROFILES_DIR = Path(__file__).parent / "profiles"


class ProfileResources(BaseModel):
    """Resource limits for the sandbox container."""

    memory: str = "256Mi"
    cpu: str = "500m"
    timeout_max: float = 30.0


class ScanStages(BaseModel):
    """Pre- and post-execution scan stages."""

    pre: list[str] = Field(default_factory=lambda: ["ast_scan"])
    post: list[str] = Field(default_factory=list)


class Profile(BaseModel):
    """Resolved sandbox profile — ready for use by guardrails and pipeline."""

    name: str
    description: str = ""
    allowed_imports: frozenset[str] = Field(default_factory=frozenset)
    blocklist: list[tuple[str, str]] = Field(default_factory=list)
    resources: ProfileResources = Field(default_factory=ProfileResources)
    scan_stages: ScanStages = Field(default_factory=ScanStages)


def load_profile(name: str) -> Profile:
    """Load a profile by name, resolving inheritance.

    Args:
        name: Profile name (matches filename without .yaml extension).

    Returns:
        Fully resolved ``Profile`` with inherited imports merged.

    Raises:
        FileNotFoundError: If the profile YAML does not exist.
        ValueError: If the profile YAML is malformed.
    """
    path = _PROFILES_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Profile '{name}' not found at {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Profile '{name}' must be a YAML mapping")

    # Resolve inheritance.
    parent_name = raw.get("extends")
    if parent_name:
        parent = load_profile(parent_name)
    else:
        parent = None

    # Build allowed imports.
    imports_section = raw.get("imports", {})
    if parent:
        allowed = set(parent.allowed_imports)
        allowed.update(imports_section.get("additional", []))
    else:
        allowed = set(imports_section.get("allowed", []))

    # Build blocklist — parent entries first, then profile-specific.
    blocklist: list[tuple[str, str]] = []
    if parent:
        blocklist.extend(parent.blocklist)
    for entry in raw.get("blocklist", []):
        if not isinstance(entry, list) or len(entry) != 2:
            raise ValueError(
                f"Profile '{name}': blocklist entry must be "
                f"[module, attribute], got {entry!r}"
            )
        blocklist.append((entry[0], entry[1]))

    # Resources — profile-specific overrides parent.
    resources_raw = raw.get("resources", {})
    if parent and not resources_raw:
        resources = parent.resources
    else:
        resources = ProfileResources(**resources_raw)

    # Scan stages — profile-specific overrides parent entirely.
    stages_raw = raw.get("scan_stages", {})
    if parent and not stages_raw:
        scan_stages = parent.scan_stages
    else:
        scan_stages = ScanStages(**stages_raw)

    return Profile(
        name=raw.get("name", name),
        description=raw.get("description", ""),
        allowed_imports=frozenset(allowed),
        blocklist=blocklist,
        resources=resources,
        scan_stages=scan_stages,
    )


def get_active_profile() -> Profile:
    """Load the profile selected by ``SANDBOX_PROFILE`` env var.

    Defaults to ``minimal`` if the variable is not set.
    """
    name = os.environ.get("SANDBOX_PROFILE", "minimal")
    return load_profile(name)
