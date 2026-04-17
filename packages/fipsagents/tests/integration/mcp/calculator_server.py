"""Trivial MCP server for stdio transport testing.

Exposes ``add`` and ``multiply`` tools, an ``explain_result`` prompt, and
``calculator://help`` / ``calculator://history/{operation}`` resources.
Run directly to start on stdio::

    python calculator_server.py
"""

from fastmcp import FastMCP

mcp = FastMCP("calculator")


@mcp.tool()
def add(a: float, b: float) -> str:
    """Add two numbers.

    Args:
        a: First number.
        b: Second number.
    """
    return str(a + b)


@mcp.tool()
def multiply(a: float, b: float) -> str:
    """Multiply two numbers.

    Args:
        a: First number.
        b: Second number.
    """
    return str(a * b)


@mcp.prompt()
def explain_result(result: str, operation: str) -> str:
    """Explain a calculation result in plain language.

    Args:
        result: The numeric result to explain.
        operation: The operation that produced it (e.g. "addition").
    """
    return f"Explain the {operation} result {result} in simple terms."


@mcp.resource("calculator://help")
def calculator_help() -> str:
    """Help text for the calculator server."""
    return "This calculator supports addition and multiplication of two numbers."


@mcp.resource("calculator://history/{operation}")
def operation_history(operation: str) -> str:
    """History of calculations for a given operation.

    Args:
        operation: The operation type (add or multiply).
    """
    if operation == "add":
        return f"Recent {operation} operations: 2+3=5, 10+20=30"
    return f"Recent {operation} operations: 4*7=28, 6*9=54"


if __name__ == "__main__":
    mcp.run(transport="stdio")
