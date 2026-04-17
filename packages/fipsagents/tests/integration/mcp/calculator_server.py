"""Trivial MCP server for stdio transport testing.

Exposes ``add`` and ``multiply`` tools.  Run directly to start on stdio::

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


if __name__ == "__main__":
    mcp.run(transport="stdio")
