"""Tests for sandbox.landlock — Landlock LSM filesystem restriction wrapper."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from sandbox.landlock import (
    DEFAULT_READ_ONLY_PATHS,
    DEFAULT_READ_WRITE_PATHS,
    LandlockStatus,
    _LandlockPathBeneathAttr,
    _LandlockRulesetAttr,
    _READ_ONLY,
    _READ_WRITE,
    apply_sandbox_landlock,
)


# ---------------------------------------------------------------------------
# LandlockStatus
# ---------------------------------------------------------------------------


class TestLandlockStatus:
    def test_default_not_applied(self):
        status = LandlockStatus()
        assert status.applied is False
        assert status.abi_version == 0
        assert status.reason == ""
        assert status.rules_applied == []

    def test_applied_status(self):
        status = LandlockStatus(
            applied=True,
            abi_version=5,
            reason="ok",
            rules_applied=["ro:/usr", "rw:/tmp"],
        )
        assert status.applied is True
        assert status.abi_version == 5
        assert len(status.rules_applied) == 2


# ---------------------------------------------------------------------------
# ctypes structures
# ---------------------------------------------------------------------------


class TestStructures:
    def test_ruleset_attr_size(self):
        # struct landlock_ruleset_attr has one u64 field = 8 bytes
        import ctypes

        assert ctypes.sizeof(_LandlockRulesetAttr) == 8

    def test_path_beneath_attr_size(self):
        # struct landlock_path_beneath_attr: u64 + s32 = 12 bytes (packed)
        import ctypes

        assert ctypes.sizeof(_LandlockPathBeneathAttr) == 12

    def test_path_beneath_attr_fields(self):
        attr = _LandlockPathBeneathAttr(allowed_access=_READ_ONLY, parent_fd=3)
        assert attr.allowed_access == _READ_ONLY
        assert attr.parent_fd == 3


# ---------------------------------------------------------------------------
# Access right constants
# ---------------------------------------------------------------------------


class TestAccessConstants:
    def test_read_only_includes_execute(self):
        assert _READ_ONLY & (1 << 0)  # EXECUTE

    def test_read_only_includes_read_file(self):
        assert _READ_ONLY & (1 << 2)  # READ_FILE

    def test_read_only_includes_read_dir(self):
        assert _READ_ONLY & (1 << 3)  # READ_DIR

    def test_read_only_excludes_write(self):
        assert not (_READ_ONLY & (1 << 1))  # WRITE_FILE

    def test_read_write_includes_write(self):
        assert _READ_WRITE & (1 << 1)  # WRITE_FILE

    def test_read_write_includes_read(self):
        assert _READ_WRITE & (1 << 2)  # READ_FILE

    def test_read_write_includes_remove(self):
        assert _READ_WRITE & (1 << 4)  # REMOVE_DIR
        assert _READ_WRITE & (1 << 5)  # REMOVE_FILE

    def test_read_write_includes_create(self):
        assert _READ_WRITE & (1 << 7)  # MAKE_DIR
        assert _READ_WRITE & (1 << 8)  # MAKE_REG


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------


class TestDefaultPaths:
    def test_read_only_includes_usr(self):
        assert "/usr" in DEFAULT_READ_ONLY_PATHS

    def test_read_only_includes_etc(self):
        assert "/etc" in DEFAULT_READ_ONLY_PATHS

    def test_read_only_includes_app_root(self):
        assert "/opt/app-root" in DEFAULT_READ_ONLY_PATHS

    def test_read_write_is_tmp_only(self):
        assert DEFAULT_READ_WRITE_PATHS == ["/tmp"]


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    def test_non_linux_returns_not_applied(self):
        """On macOS/Windows, Landlock degrades gracefully."""
        with patch("sandbox.landlock.sys") as mock_sys:
            mock_sys.platform = "darwin"
            status = apply_sandbox_landlock()
            assert status.applied is False
            assert "Not Linux" in status.reason

    def test_custom_paths_accepted(self):
        """Custom path lists are accepted (even if Landlock isn't applied)."""
        with patch("sandbox.landlock.sys") as mock_sys:
            mock_sys.platform = "darwin"
            status = apply_sandbox_landlock(
                read_only_paths=["/custom/ro"],
                read_write_paths=["/custom/rw"],
            )
            assert status.applied is False

    @pytest.mark.skipif(
        sys.platform != "linux", reason="Landlock only available on Linux"
    )
    def test_returns_abi_version_on_linux(self):
        """On Linux, the ABI version is queried (may be 0 if not enabled)."""
        status = apply_sandbox_landlock()
        # Either it applied (abi >= 1) or it didn't (abi == 0 with reason)
        assert isinstance(status.abi_version, int)
        assert status.reason != ""
