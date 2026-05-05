"""Regression tests for the [files*] optional-dependency split.

The ``[files]`` extra used to bundle docling + python-magic, dragging in
~500 MB of torch + transformers even for image-only agents that just
need libmagic-based MIME sniffing.  We split it into:

- ``[files-image]`` — python-magic only
- ``[files-text]`` — python-magic + docling (heavy)
- ``[files]`` — meta-extra equal to the union, kept so callers pinning
  ``fipsagents[files]`` keep working unchanged

These tests assert the structure so a future refactor does not silently
collapse them back together.  They read ``pyproject.toml`` from the
package root rather than installed metadata so they exercise the source
of truth.
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover — package requires-python = ">=3.11"
    import tomli as tomllib  # type: ignore[import-not-found]

import pytest


@pytest.fixture(scope="module")
def extras() -> dict[str, list[str]]:
    pyproject = (
        Path(__file__).resolve().parents[1] / "pyproject.toml"
    )
    with pyproject.open("rb") as fh:
        data = tomllib.load(fh)
    return data["project"]["optional-dependencies"]


def _names(deps: list[str]) -> set[str]:
    """Strip version specifiers / extras markers and return the bare
    distribution names so assertions don't depend on minor version
    bumps."""
    out: set[str] = set()
    for spec in deps:
        # "python-magic>=0.4.27" → "python-magic"
        # "docling>=2.30" → "docling"
        for sep in (">=", "<=", "==", "!=", "~=", ">", "<", "[", ";"):
            spec = spec.split(sep, 1)[0]
        out.add(spec.strip())
    return out


def test_files_image_extra_exists_with_only_python_magic(extras):
    assert "files-image" in extras, "files-image extra was removed"
    names = _names(extras["files-image"])
    assert names == {"python-magic"}, (
        f"files-image should contain only python-magic, got {names}. "
        "Adding docling here defeats the purpose of the split."
    )


def test_files_text_extra_includes_docling(extras):
    assert "files-text" in extras, "files-text extra was removed"
    names = _names(extras["files-text"])
    assert "docling" in names, (
        f"files-text must include docling for binary parsing, got {names}"
    )
    assert "python-magic" in names, (
        f"files-text must include python-magic so MIME sniffing works "
        f"alongside docling parsing, got {names}"
    )


def test_files_meta_extra_is_union(extras):
    """``[files]`` is the backward-compatibility meta-extra.  It must
    pull in everything ``[files-image]`` and ``[files-text]`` do, so
    existing ``pip install fipsagents[files]`` keeps resolving identically.
    """
    assert "files" in extras, "files meta-extra was removed"
    files_names = _names(extras["files"])
    image_names = _names(extras["files-image"])
    text_names = _names(extras["files-text"])
    union = image_names | text_names
    assert union <= files_names, (
        f"files meta-extra is missing entries from the union: "
        f"missing={union - files_names}, files={files_names}"
    )


def test_files_image_does_not_drag_in_docling(extras):
    """Defensive duplicate of the first assertion — explicit so the
    intent ('image agents must not pull torch') is greppable."""
    names = _names(extras["files-image"])
    assert "docling" not in names
    # And no docling-adjacent heavyweights either.
    for heavy in ("torch", "transformers"):
        assert heavy not in names, (
            f"files-image leaked {heavy} — image-only agents should "
            "stay lightweight"
        )
