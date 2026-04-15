"""AST-based guardrails for the code execution sandbox.

Validates LLM-generated Python code before execution by walking the parse tree
and collecting all violations in a single pass.  Returns an empty list when the
code is safe to run; a non-empty list when it must be rejected.

The allowed import set is supplied by the active profile — the visitor no
longer owns a hardcoded module list.

Usage::

    from sandbox.guardrails import validate_code
    from sandbox.profiles import get_active_profile

    profile = get_active_profile()
    violations = validate_code(source, profile.allowed_imports)

    # or with default minimal profile (backward-compatible):
    violations = validate_code(source)
"""

from __future__ import annotations

import ast
import re
# Default allowed imports — used when no profile is passed (backward compat).
# Matches the ``minimal`` profile.
_DEFAULT_ALLOWED_IMPORTS: frozenset[str] = frozenset(
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

# Public alias for backward compatibility.
ALLOWED_IMPORTS = _DEFAULT_ALLOWED_IMPORTS

# Bare function names that are always blocked.
_BLOCKED_CALLS: frozenset[str] = frozenset(
    {
        "eval", "exec", "compile", "__import__", "open",
        "getattr", "setattr", "delattr",  # prevent dynamic attribute access bypasses
        "breakpoint", "input",  # would hang the subprocess until timeout
    }
)

# Top-level module names whose *any* attribute access is blocked.
_BLOCKED_MODULES: frozenset[str] = frozenset(
    {"subprocess", "socket", "importlib"}
)

# Specific (module, attr) pairs that are blocked.
_BLOCKED_MODULE_ATTRS: frozenset[tuple[str, str]] = frozenset(
    {
        ("os", "system"),
        ("os", "popen"),
    }
)

# Dunder attribute names that are blocked regardless of the object they appear on.
_BLOCKED_DUNDERS: frozenset[str] = frozenset(
    {"__subclasses__", "__globals__", "__builtins__"}
)

# Unsafe deserialization: (module, method) pairs.  These modules are already
# blocked by ALLOWED_IMPORTS, but explicit checks give clearer error messages
# (defense-in-depth).
_UNSAFE_DESER: frozenset[tuple[str, str]] = frozenset(
    {
        ("pickle", "loads"),
        ("pickle", "load"),
        ("pickle", "Unpickler"),
        ("yaml", "unsafe_load"),
        ("yaml", "load"),
        ("marshal", "loads"),
        ("marshal", "load"),
        ("shelve", "open"),
    }
)

# Weak cryptographic hash functions.
_WEAK_CRYPTO_CALLS: frozenset[tuple[str, str]] = frozenset(
    {
        ("hashlib", "md5"),
        ("hashlib", "sha1"),
    }
)

# Compiled regexes for credential / secret detection in string literals.
_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("AWS access key ID", re.compile(r"AKIA[0-9A-Z]{16}")),
    (
        "generic secret assignment",
        re.compile(
            r"(?:api[_-]?key|api[_-]?secret|token|secret[_-]?key"
            r"|password|passwd|auth[_-]?token)"
            r"""\s*[:=]\s*['"][A-Za-z0-9+/=_-]{16,}['"]"""
        ),
    ),
    (
        "PEM private key",
        re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
    ),
    (
        "high-entropy hex string",
        re.compile(r"\b[0-9a-fA-F]{32,}\b"),
    ),
]

# SQL keyword pattern for injection detection.
_SQL_KEYWORD_RE: re.Pattern[str] = re.compile(
    r"\b(?:SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|EXEC|UNION)\b",
    re.IGNORECASE,
)

# Path traversal pattern — any string containing "../".
_PATH_TRAVERSAL_RE: re.Pattern[str] = re.compile(r"\.\./")


class _GuardrailVisitor(ast.NodeVisitor):
    """Single-pass AST visitor that collects all policy violations."""

    def __init__(
        self,
        allowed_imports: frozenset[str] | None = None,
    ) -> None:
        self.violations: list[str] = []
        self._allowed_imports = (
            allowed_imports if allowed_imports is not None
            else _DEFAULT_ALLOWED_IMPORTS
        )

    # ------------------------------------------------------------------
    # Import checking
    # ------------------------------------------------------------------

    def _check_module_name(self, name: str, lineno: int) -> None:
        """Reject any module not on the allowlist (checks the top-level name)."""
        top = name.split(".")[0]
        if top not in self._allowed_imports:
            self.violations.append(
                f"Line {lineno}: import of '{name}' is not allowed "
                f"(not in allowed imports)"
            )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._check_module_name(alias.name, node.lineno)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        # `from . import foo` has module=None; treat as forbidden.
        module: str = node.module or ""
        self._check_module_name(module, node.lineno)
        self.generic_visit(node)

    # ------------------------------------------------------------------
    # Call checking
    # ------------------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func

        if isinstance(func, ast.Name):
            # Simple call: eval(...), exec(...), open(...)
            if func.id in _BLOCKED_CALLS:
                self.violations.append(
                    f"Line {node.lineno}: call to '{func.id}()' is not allowed"
                )

        elif isinstance(func, ast.Attribute):
            obj = func.value
            attr = func.attr

            # os.system(...) / os.popen(...)
            if isinstance(obj, ast.Name):
                if (obj.id, attr) in _BLOCKED_MODULE_ATTRS:
                    self.violations.append(
                        f"Line {node.lineno}: call to '{obj.id}.{attr}()' is not allowed"
                    )
                # subprocess.run(...), socket.connect(...), importlib.import_module(...)
                elif obj.id in _BLOCKED_MODULES:
                    self.violations.append(
                        f"Line {node.lineno}: call to '{obj.id}.{attr}()' is not allowed "
                        f"(module '{obj.id}' is blocked)"
                    )

                # Unsafe deserialization: pickle.loads(), yaml.load(), etc.
                if (obj.id, attr) in _UNSAFE_DESER:
                    self.violations.append(
                        f"Line {node.lineno}: call to '{obj.id}.{attr}()' is unsafe "
                        f"deserialization"
                    )

                # Weak crypto: hashlib.md5(), hashlib.sha1()
                if (obj.id, attr) in _WEAK_CRYPTO_CALLS:
                    self.violations.append(
                        f"Line {node.lineno}: call to '{obj.id}.{attr}()' uses weak "
                        f"cryptography"
                    )

            # str.format() SQL injection: "SELECT ...".format(...)
            if (
                attr == "format"
                and isinstance(obj, ast.Constant)
                and isinstance(obj.value, str)
                and _SQL_KEYWORD_RE.search(obj.value)
            ):
                self.violations.append(
                    f"Line {node.lineno}: potential SQL injection via "
                    f".format() on a string containing SQL keywords"
                )

        self.generic_visit(node)

    # ------------------------------------------------------------------
    # Attribute access checking (dunders + blocked module attrs)
    # ------------------------------------------------------------------

    def visit_Attribute(self, node: ast.Attribute) -> None:
        attr = node.attr

        # Block dangerous dunder attributes on any object.
        if attr in _BLOCKED_DUNDERS:
            self.violations.append(
                f"Line {node.lineno}: access to '{attr}' attribute is not allowed"
            )

        self.generic_visit(node)

    # ------------------------------------------------------------------
    # String literal checking (credentials, path traversal)
    # ------------------------------------------------------------------

    def visit_Constant(self, node: ast.Constant) -> None:
        if not isinstance(node.value, str) or len(node.value) < 16:
            self.generic_visit(node)
            return

        value = node.value

        # Credential / secret patterns
        for name, pattern in _SECRET_PATTERNS:
            if pattern.search(value):
                self.violations.append(
                    f"Line {node.lineno}: string literal matches {name} pattern"
                )
                break  # one credential violation per string is enough

        # Path traversal
        if _PATH_TRAVERSAL_RE.search(value):
            self.violations.append(
                f"Line {node.lineno}: string literal contains path traversal "
                f"sequence ('../')"
            )

        self.generic_visit(node)

    # ------------------------------------------------------------------
    # SQL injection — f-strings
    # ------------------------------------------------------------------

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        # Reconstruct the static text portions of the f-string and check
        # for SQL keywords.
        static_parts: list[str] = []
        has_interpolation = False
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                static_parts.append(part.value)
            else:
                has_interpolation = True

        # Only flag if there is actual interpolation (otherwise it's just a
        # regular string that happens to use f"" syntax).
        if has_interpolation:
            combined = " ".join(static_parts)
            if _SQL_KEYWORD_RE.search(combined):
                self.violations.append(
                    f"Line {node.lineno}: potential SQL injection via f-string "
                    f"containing SQL keywords"
                )

        self.generic_visit(node)

    # ------------------------------------------------------------------
    # SQL injection — %-formatting
    # ------------------------------------------------------------------

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if (
            isinstance(node.op, ast.Mod)
            and isinstance(node.left, ast.Constant)
            and isinstance(node.left.value, str)
            and _SQL_KEYWORD_RE.search(node.left.value)
        ):
            self.violations.append(
                f"Line {node.lineno}: potential SQL injection via "
                f"%-formatting on a string containing SQL keywords"
            )

        self.generic_visit(node)


def validate_code(
    source: str,
    allowed_imports: frozenset[str] | None = None,
) -> list[str]:
    """Validate *source* against sandbox policy (AST scan stage).

    Parameters
    ----------
    source:
        Python source code to validate.
    allowed_imports:
        Set of allowed top-level module names.  Defaults to the minimal
        profile's allowlist when ``None``.

    Returns
    -------
    list[str]
        A list of human-readable violation strings.  An empty list means the
        code passed all checks and is safe to execute in the sandbox.  If the
        source cannot be parsed, the list contains a single entry describing
        the ``SyntaxError``.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"SyntaxError: {exc.msg} (line {exc.lineno})"]

    visitor = _GuardrailVisitor(allowed_imports=allowed_imports)
    visitor.visit(tree)
    return visitor.violations


# ------------------------------------------------------------------
# Blocklist audit stage (Tier 2+)
# ------------------------------------------------------------------


class _BlocklistVisitor(ast.NodeVisitor):
    """Lightweight AST visitor that checks attribute access against a blocklist.

    The blocklist is a list of ``(module_or_type, attribute)`` tuples.  The
    visitor flags any ``name.attr`` access where ``(name, attr)`` matches a
    blocklist entry.  This catches both module-level access (``pandas.read_pickle``)
    and method calls on known types (``DataFrame.to_sql``).
    """

    def __init__(self, blocklist: list[tuple[str, str]]) -> None:
        self.violations: list[str] = []
        self._blocklist = {(m, a) for m, a in blocklist}

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name):
            if (node.value.id, node.attr) in self._blocklist:
                self.violations.append(
                    f"Line {node.lineno}: access to "
                    f"'{node.value.id}.{node.attr}' is blocked by profile"
                )
        self.generic_visit(node)


def blocklist_audit(
    source: str,
    blocklist: list[tuple[str, str]],
) -> list[str]:
    """Check *source* for attribute access blocked by the active profile.

    This is a separate, lightweight AST pass that only checks attribute
    access against the profile's blocklist.  It runs after ``validate_code``
    in Tier 2+ profiles.

    Parameters
    ----------
    source:
        Python source code (must already pass ``validate_code``).
    blocklist:
        List of ``(module_or_type, attribute)`` pairs to reject.

    Returns
    -------
    list[str]
        Violation strings.  Empty means the code passed.
    """
    if not blocklist:
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []  # validate_code already caught this

    visitor = _BlocklistVisitor(blocklist)
    visitor.visit(tree)
    return visitor.violations
