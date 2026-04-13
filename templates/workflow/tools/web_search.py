"""Web search tool — LLM-callable (plane 2).

The LLM decides when and what to search.  This stub returns realistic mock
results for template demonstration.  In production, replace the body with
a call to Tavily, Brave Search, or wire up an MCP search server via
``agent.yaml``::

    mcp_servers:
      - url: ${SEARCH_MCP_URL:-http://tavily-mcp:8080/mcp}

MCP-discovered tools are automatically registered with ``llm_only``
visibility, so you can delete this file entirely once a search MCP server
is configured.
"""

from fipsagents.baseagent.tools import tool


@tool(
    description="Search the web for information on a topic",
    visibility="llm_only",
)
async def web_search(query: str) -> str:
    """Search the web and return relevant results.

    Args:
        query: The search query string.

    Returns:
        Formatted search results with titles, snippets, and URLs.
    """
    # --- Stub implementation for the template ---
    # Replace this with a real search provider. For example, with Tavily:
    #
    #   import httpx
    #   async with httpx.AsyncClient() as client:
    #       resp = await client.post(
    #           "https://api.tavily.com/search",
    #           json={"query": query, "api_key": os.environ["TAVILY_API_KEY"]},
    #       )
    #       data = resp.json()
    #       return "\n\n".join(
    #           f"**{r['title']}**\n{r['content']}\nURL: {r['url']}"
    #           for r in data["results"]
    #       )

    results = [
        {
            "title": f"Research findings on: {query}",
            "snippet": (
                f"Recent studies indicate significant developments in {query}. "
                f"Multiple peer-reviewed sources confirm these findings."
            ),
            "url": f"https://example.com/research/{query.replace(' ', '-')}",
        },
        {
            "title": f"Expert analysis: {query}",
            "snippet": (
                f"Leading experts have published comprehensive analyses of "
                f"{query}, highlighting both opportunities and challenges."
            ),
            "url": f"https://example.com/analysis/{query.replace(' ', '-')}",
        },
        {
            "title": f"Technical overview of {query}",
            "snippet": (
                f"A detailed technical overview covering the fundamentals of "
                f"{query} and practical applications."
            ),
            "url": f"https://example.com/overview/{query.replace(' ', '-')}",
        },
    ]

    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] **{r['title']}**")
        lines.append(f"    {r['snippet']}")
        lines.append(f"    URL: {r['url']}")
        lines.append("")

    return "\n".join(lines)
