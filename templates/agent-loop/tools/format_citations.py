"""Citation formatting tool — agent-code only (plane 1).

Called by the agent's Python code after collecting search results, not by
the LLM.  Demonstrates the ``agent_only`` tool plane: the agent decides
when to format citations as a post-processing step.
"""

from fipsagents.baseagent.tools import tool


# Intentionally synchronous: this is pure CPU string formatting with no I/O.
# The tool system runs sync tools in a thread executor, so they won't block
# the event loop.  Prefer sync for functions that don't await anything.
@tool(
    description="Format raw URLs and titles into clean citation strings",
    visibility="agent_only",
)
def format_citations(urls: list, titles: list) -> str:
    """Format URLs and titles into numbered citation lines.

    Args:
        urls: List of source URLs.
        titles: List of source titles (same length as urls).

    Returns:
        Newline-separated citation strings, one per source.
    """
    lines = []
    for i, (url, title) in enumerate(zip(urls, titles), 1):
        # Strip whitespace, skip empty entries
        url = str(url).strip()
        title = str(title).strip()
        if not url:
            continue
        if title:
            lines.append(f"[{i}] {title} — {url}")
        else:
            lines.append(f"[{i}] {url}")
    return "\n".join(lines)
