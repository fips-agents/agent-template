# Example Agent: Research Assistant

The scaffolded template ships with a working research assistant agent that demonstrates the core BaseAgent patterns. This agent is the reference implementation -- it exercises enough of the API to validate the design while remaining simple enough that the pattern is immediately obvious.

## What It Does

The research assistant takes a question, searches for information using a web search tool, validates that the response actually addresses the question, and returns a structured answer with citations.

## Why This Example

A research assistant is generic (not domain-specific), immediately understandable, and naturally exercises the features that matter most in the template:

**LLM interaction.** The agent uses `call_model()` for reasoning about search results, `call_model_json()` for producing structured output, and `call_model_validated()` for ensuring the response addresses the original question. All three primary model-calling patterns appear in a single agent.

**Two tool planes.** Web search is an LLM-callable tool (plane 2, `llm_only`) -- the LLM decides when to search and what to search for. Citation formatting is an agent-code tool (plane 1, `agent_only`) -- the agent's Python code calls it after getting results, and the LLM never sees it. This makes the two-plane distinction concrete without being contrived.

**MCP integration.** The web search tool can come from an MCP server (Tavily or similar), demonstrating how MCP tools are discovered and made available to the LLM.

**Prompt loading.** The system prompt comes from `prompts/system.md`, demonstrating the Markdown-with-YAML-frontmatter format.

**Structured output.** The final answer is a structured object (answer text, confidence, citations list) produced via `call_model_json()`.

**Validation loop.** `call_model_validated()` checks that the response actually addresses the question before returning it, demonstrating the retry-with-backoff pattern.

## Agent Subclass (~25 lines)

The subclass implements `step()` which:
1. Loads the system prompt
2. Calls the model with the user's question and available LLM tools (web search)
3. Handles any tool calls the LLM makes (search, follow-up searches)
4. Calls `call_model_validated()` to produce a final answer, validating relevance
5. Calls the `format_citations` agent-code tool to clean up citations
6. Returns the structured result

## Tools

**`web_search`** (visibility: `llm_only`) -- Searches the web via Tavily API. The LLM decides when and what to search. In production, this would come from an MCP server. The example includes a local tool that calls Tavily directly.

**`format_citations`** (visibility: `agent_only`) -- Takes raw search results and formats them into clean citation objects. Called by agent code, invisible to the LLM.

## Configuration

The Tavily API key comes from an environment variable (`TAVILY_API_KEY`), which in OpenShift would be injected from a Secret. A `.env` file (gitignored) provides it for local development.

## What a Developer Learns

After reading the example agent, a developer understands:
- How to subclass BaseAgent and implement `step()`
- How to define tools with the `@tool` decorator and different visibility levels
- How to write prompts as Markdown with YAML frontmatter
- How to use `call_model()`, `call_model_json()`, and `call_model_validated()`
- How LLM tool calling works through BaseAgent
- How agent-code tools differ from LLM-callable tools
- How configuration flows from agent.yaml through env vars
