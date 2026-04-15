"""Optional Landlock LSM filesystem restriction for the sandbox.

Applies kernel-level filesystem access rules when available (Linux 5.13+,
enabled by default on RHEL 9.6+ / OCP 4.18+).  Degrades gracefully on
older kernels, non-Linux platforms, or when the Landlock LSM is not in
the kernel's boot list.

Landlock rules are inherited by child processes, so applying them to the
FastAPI app automatically restricts the ``python3 -I`` subprocess that
executes LLM-generated code.

The ``no_new_privs`` bit is required for unprivileged Landlock.  On
OpenShift, ``restricted-v2`` SCC sets ``allowPrivilegeEscalation: false``
which causes CRI-O to set ``no_new_privs`` automatically.  We also call
``prctl(PR_SET_NO_NEW_PRIVS)`` ourselves for standalone (non-OpenShift)
usage.

Usage::

    from sandbox.landlock import apply_sandbox_landlock

    status = apply_sandbox_landlock()
    # status.applied is True if Landlock is active

See also: research/sandbox-hardening-v2.md Finding 1 and Finding 5.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import platform
import sys
from dataclasses import dataclass, field
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class LandlockStatus:
    """Result of applying Landlock restrictions."""

    applied: bool = False
    abi_version: int = 0
    reason: str = ""
    rules_applied: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Kernel constants (x86_64 and aarch64)
# ---------------------------------------------------------------------------

# Syscall numbers differ by architecture.
_SYSCALL_NUMBERS: dict[str, tuple[int, int, int]] = {
    # (create_ruleset, add_rule, restrict_self)
    "x86_64": (444, 445, 446),
    "aarch64": (444, 445, 446),  # Same on aarch64
}

# prctl constants
_PR_SET_NO_NEW_PRIVS = 38

# Landlock create_ruleset flags
_LANDLOCK_CREATE_RULESET_VERSION = 1 << 0

# Filesystem access rights (ABI v1)
_ACCESS_FS_EXECUTE = 1 << 0
_ACCESS_FS_WRITE_FILE = 1 << 1
_ACCESS_FS_READ_FILE = 1 << 2
_ACCESS_FS_READ_DIR = 1 << 3
_ACCESS_FS_REMOVE_DIR = 1 << 4
_ACCESS_FS_REMOVE_FILE = 1 << 5
_ACCESS_FS_MAKE_CHAR = 1 << 6
_ACCESS_FS_MAKE_DIR = 1 << 7
_ACCESS_FS_MAKE_REG = 1 << 8
_ACCESS_FS_MAKE_SOCK = 1 << 9
_ACCESS_FS_MAKE_FIFO = 1 << 10
_ACCESS_FS_MAKE_BLOCK = 1 << 11
_ACCESS_FS_MAKE_SYM = 1 << 12

# ABI v2+
_ACCESS_FS_REFER = 1 << 13

# ABI v3+
_ACCESS_FS_TRUNCATE = 1 << 14

# Rule type
_LANDLOCK_RULE_PATH_BENEATH = 1

# Composite access sets
_READ_ONLY = _ACCESS_FS_EXECUTE | _ACCESS_FS_READ_FILE | _ACCESS_FS_READ_DIR
_READ_WRITE = (
    _READ_ONLY
    | _ACCESS_FS_WRITE_FILE
    | _ACCESS_FS_REMOVE_DIR
    | _ACCESS_FS_REMOVE_FILE
    | _ACCESS_FS_MAKE_DIR
    | _ACCESS_FS_MAKE_REG
    | _ACCESS_FS_MAKE_SYM
    | _ACCESS_FS_TRUNCATE
)

# All filesystem access rights for ABI v1 (used in the ruleset attribute to
# declare which rights the ruleset will handle).
_ALL_ACCESS_FS_V1 = (
    _ACCESS_FS_EXECUTE
    | _ACCESS_FS_WRITE_FILE
    | _ACCESS_FS_READ_FILE
    | _ACCESS_FS_READ_DIR
    | _ACCESS_FS_REMOVE_DIR
    | _ACCESS_FS_REMOVE_FILE
    | _ACCESS_FS_MAKE_CHAR
    | _ACCESS_FS_MAKE_DIR
    | _ACCESS_FS_MAKE_REG
    | _ACCESS_FS_MAKE_SOCK
    | _ACCESS_FS_MAKE_FIFO
    | _ACCESS_FS_MAKE_BLOCK
    | _ACCESS_FS_MAKE_SYM
)


# ---------------------------------------------------------------------------
# ctypes structures matching kernel ABI
# ---------------------------------------------------------------------------


class _LandlockRulesetAttr(ctypes.Structure):
    """``struct landlock_ruleset_attr`` — declares handled access rights."""

    _fields_ = [
        ("handled_access_fs", ctypes.c_uint64),
    ]


class _LandlockPathBeneathAttr(ctypes.Structure):
    """``struct landlock_path_beneath_attr`` — a single path rule."""

    _pack_ = 1
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
    ]


# ---------------------------------------------------------------------------
# Default sandbox paths
# ---------------------------------------------------------------------------

# Paths that the sandbox needs to read (Python runtime, system libs, app code).
DEFAULT_READ_ONLY_PATHS: list[str] = [
    "/usr",           # Python binary, stdlib, system tools
    "/lib",           # Shared libraries (symlink on some distros)
    "/lib64",         # 64-bit shared libraries
    "/etc",           # Timezone, locale, nsswitch, ld.so.cache
    "/opt/app-root",  # UBI app directory (the FastAPI app lives here)
    "/proc/self",     # Python reads /proc/self/fd, /proc/self/status
]

# Paths that the sandbox needs to write (temp files for code execution).
DEFAULT_READ_WRITE_PATHS: list[str] = [
    "/tmp",
]


# ---------------------------------------------------------------------------
# Low-level syscall wrappers
# ---------------------------------------------------------------------------


def _get_libc() -> ctypes.CDLL | None:
    """Load libc for syscall access.  Returns None on non-Linux."""
    if sys.platform != "linux":
        return None
    path = ctypes.util.find_library("c")
    if path is None:
        return None
    return ctypes.CDLL(path, use_errno=True)


def _get_syscall_numbers() -> tuple[int, int, int] | None:
    """Return (create_ruleset, add_rule, restrict_self) for this arch."""
    machine = platform.machine()
    return _SYSCALL_NUMBERS.get(machine)


def _query_abi_version(libc: ctypes.CDLL, sys_create: int) -> int:
    """Query the supported Landlock ABI version.  Returns 0 if unavailable."""
    result = libc.syscall(
        sys_create,
        None,                   # attr = NULL
        ctypes.c_size_t(0),     # size = 0
        ctypes.c_uint32(_LANDLOCK_CREATE_RULESET_VERSION),
    )
    if result < 0:
        return 0
    return result


def _set_no_new_privs(libc: ctypes.CDLL) -> bool:
    """Set the no_new_privs bit via prctl.  Returns True on success."""
    result = libc.prctl(
        _PR_SET_NO_NEW_PRIVS,
        ctypes.c_ulong(1),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
    )
    return result == 0


# ---------------------------------------------------------------------------
# Rule application
# ---------------------------------------------------------------------------


def _add_path_rule(
    libc: ctypes.CDLL,
    sys_add_rule: int,
    ruleset_fd: int,
    path: str,
    access: int,
) -> bool:
    """Add a single path-beneath rule to the ruleset.

    Returns True on success.  Silently skips paths that don't exist
    (the container image may not have all expected paths).
    """
    if not os.path.exists(path):
        logger.debug("Landlock: skipping non-existent path %s", path)
        return True

    fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
    try:
        attr = _LandlockPathBeneathAttr(
            allowed_access=access,
            parent_fd=fd,
        )
        result = libc.syscall(
            sys_add_rule,
            ctypes.c_int(ruleset_fd),
            ctypes.c_int(_LANDLOCK_RULE_PATH_BENEATH),
            ctypes.byref(attr),
            ctypes.c_uint32(0),
        )
        return result == 0
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_sandbox_landlock(
    *,
    read_only_paths: list[str] | None = None,
    read_write_paths: list[str] | None = None,
) -> LandlockStatus:
    """Apply Landlock filesystem restrictions to the current process.

    Call this once at application startup, before spawning any
    subprocesses.  The restrictions are inherited by all child
    processes (including the ``python3 -I`` code execution subprocess).

    Parameters
    ----------
    read_only_paths:
        Filesystem paths to allow read + execute access.  Defaults to
        ``DEFAULT_READ_ONLY_PATHS`` (Python runtime, system libs, app).
    read_write_paths:
        Filesystem paths to allow full read/write access.  Defaults to
        ``DEFAULT_READ_WRITE_PATHS`` (``/tmp`` only).

    Returns
    -------
    LandlockStatus:
        Whether Landlock was applied, the ABI version, and details.
    """
    ro_paths = read_only_paths if read_only_paths is not None else DEFAULT_READ_ONLY_PATHS
    rw_paths = read_write_paths if read_write_paths is not None else DEFAULT_READ_WRITE_PATHS

    # -- Pre-checks --

    if sys.platform != "linux":
        return LandlockStatus(reason=f"Not Linux (platform={sys.platform})")

    libc = _get_libc()
    if libc is None:
        return LandlockStatus(reason="Cannot load libc")

    syscall_nums = _get_syscall_numbers()
    if syscall_nums is None:
        machine = platform.machine()
        return LandlockStatus(reason=f"Unsupported architecture: {machine}")

    sys_create, sys_add_rule, sys_restrict = syscall_nums

    # -- Query ABI version --

    abi = _query_abi_version(libc, sys_create)
    if abi < 1:
        return LandlockStatus(
            abi_version=0,
            reason="Landlock not available (kernel too old or LSM not enabled)",
        )

    logger.info("Landlock ABI version %d available", abi)

    # -- Determine handled access rights based on ABI --

    handled = _ALL_ACCESS_FS_V1
    if abi >= 2:
        handled |= _ACCESS_FS_REFER
    if abi >= 3:
        handled |= _ACCESS_FS_TRUNCATE

    # -- Create ruleset --

    attr = _LandlockRulesetAttr(handled_access_fs=handled)
    ruleset_fd = libc.syscall(
        sys_create,
        ctypes.byref(attr),
        ctypes.c_size_t(ctypes.sizeof(attr)),
        ctypes.c_uint32(0),
    )
    if ruleset_fd < 0:
        errno = ctypes.get_errno()
        return LandlockStatus(
            abi_version=abi,
            reason=f"landlock_create_ruleset failed (errno={errno})",
        )

    # -- Add path rules --

    rules_applied: list[str] = []

    try:
        for path in ro_paths:
            if _add_path_rule(libc, sys_add_rule, ruleset_fd, path, _READ_ONLY):
                if os.path.exists(path):
                    rules_applied.append(f"ro:{path}")

        for path in rw_paths:
            if _add_path_rule(libc, sys_add_rule, ruleset_fd, path, _READ_WRITE):
                if os.path.exists(path):
                    rules_applied.append(f"rw:{path}")

        # -- Set no_new_privs (required for unprivileged Landlock) --
        # On OpenShift, this is already set by CRI-O via the restricted-v2
        # SCC's allowPrivilegeEscalation: false.  Setting it again is a no-op.
        if not _set_no_new_privs(libc):
            logger.warning("prctl(PR_SET_NO_NEW_PRIVS) failed — Landlock "
                           "requires either this or CAP_SYS_ADMIN")

        # -- Apply the ruleset --

        result = libc.syscall(
            sys_restrict,
            ctypes.c_int(ruleset_fd),
            ctypes.c_uint32(0),
        )
        if result < 0:
            errno = ctypes.get_errno()
            return LandlockStatus(
                abi_version=abi,
                reason=f"landlock_restrict_self failed (errno={errno})",
                rules_applied=rules_applied,
            )
    finally:
        os.close(ruleset_fd)

    logger.info(
        "Landlock applied: %d rules (%s)",
        len(rules_applied),
        ", ".join(rules_applied),
    )
    return LandlockStatus(
        applied=True,
        abi_version=abi,
        reason="ok",
        rules_applied=rules_applied,
    )
