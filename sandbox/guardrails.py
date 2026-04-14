"""AST-based guardrails for the code execution sandbox.

Validates LLM-generated Python code before execution by walking the parse tree
and collecting all violations in a single pass.  Returns an empty list when the
code is safe to run; a non-empty list when it must be rejected.

Usage::

    from sandbox.guardrails import validate_code

    violations = validate_code(source)
    if violations:
        # surface all violations to the LLM so it can fix them in one retry
        ...
"""

import ast

# Modules the sandbox is allowed to import.
ALLOWED_IMPORTS: frozenset[str] = frozenset(
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


class _GuardrailVisitor(ast.NodeVisitor):
    """Single-pass AST visitor that collects all policy violations."""

    def __init__(self) -> None:
        self.violations: list[str] = []

    # ------------------------------------------------------------------
    # Import checking
    # ------------------------------------------------------------------

    def _check_module_name(self, name: str, lineno: int) -> None:
        """Reject any module not on the allowlist (checks the top-level name)."""
        top = name.split(".")[0]
        if top not in ALLOWED_IMPORTS:
            self.violations.append(
                f"Line {lineno}: import of '{name}' is not allowed "
                f"(not in ALLOWED_IMPORTS)"
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


def validate_code(source: str) -> list[str]:
    """Validate *source* against sandbox policy.

    Parameters
    ----------
    source:
        Python source code to validate.

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

    visitor = _GuardrailVisitor()
    visitor.visit(tree)
    return visitor.violations
