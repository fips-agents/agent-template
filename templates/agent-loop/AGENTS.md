# Research Assistant

An example agent built on the `agent-loop` template. Takes a research query,
searches the web, validates relevance, and returns a structured report with
citations. It demonstrates all three BaseAgent model-calling patterns and both
tool planes.

## Version

0.1.0

## Capabilities

- Search the web for information on a given topic
- Evaluate and filter results for relevance to the original query
- Produce a structured research report with a confidence score
- Format source URLs into numbered citation strings

## Tools

### web_search

**Plane:** LLM-callable (plane 2)  
**Visibility:** `llm_only`

The LLM decides when and what to search. In the template this is a stub; in
production replace the body with a Tavily, Brave, or similar API call, or
remove the file entirely and configure a search MCP server in `agent.yaml`.

| Parameter | Type   | Description          |
|-----------|--------|----------------------|
| `query`   | string | The search query     |

Returns formatted results: title, snippet, and URL for each result.

### format_citations

**Plane:** Agent-code only (plane 1)  
**Visibility:** `agent_only`

Called by the agent's Python code as a post-processing step after the report
is generated. The LLM never sees this tool. Formats raw URLs and titles into
numbered citation lines.

| Parameter | Type         | Description                            |
|-----------|--------------|----------------------------------------|
| `urls`    | list[string] | Source URLs                            |
| `titles`  | list[string] | Source titles (same length as `urls`)  |

Returns newline-separated citation strings.

## Input / Output

**Input:** A user message containing a research query (plain text).

**Output:** A `ResearchReport` object with three fields:

| Field        | Type          | Description                          |
|--------------|---------------|--------------------------------------|
| `answer`     | string        | The research answer in Markdown      |
| `confidence` | float (0–1)   | Model confidence in the answer       |
| `citations`  | list[string]  | Formatted citation strings           |

## Configuration

Agent behavior is controlled by `agent.yaml`. All values support
`${VAR:-default}` environment variable substitution so that the configuration
structure stays baked into the container image while environment-specific
values come from OpenShift ConfigMaps and Secrets at deploy time.

Key configuration sections:

```yaml
model:
  endpoint: ${MODEL_ENDPOINT:-http://llamastack:8321/v1}
  name: ${MODEL_NAME:-meta-llama/Llama-3.3-70B-Instruct}
  temperature: 0.7
  max_tokens: 4096

loop:
  max_iterations: ${MAX_ITERATIONS:-100}
```

See `agent.yaml` for the full schema.

## Dependencies

The agent requires an LLM endpoint that speaks the OpenAI-compatible chat
completions API. The template ships with `litellm` as the LLM client, which
supports 100+ providers (vLLM, LlamaStack, Anthropic, OpenAI, Azure, Bedrock,
and others) via model string prefix.

**Required at runtime:**

- An LLM inference endpoint (vLLM, LlamaStack, or any OpenAI-compatible API)

**Optional:**

- A search MCP server (if replacing the `web_search` stub with an MCP-based
  implementation — configure via `mcp_servers` in `agent.yaml`)
- A MemoryHub instance (for persistent cross-session memory — wire via
  `memoryhub config init`)

The agent has no dependency on LangChain, LangGraph, or any agent framework.
Core Python dependencies are: `litellm`, `fastmcp` (v3), `pydantic`, `httpx`,
and `python-frontmatter`.

## Deployment

The agent deploys to OpenShift as an immutable container image built from a
Red Hat UBI base. Everything that defines agent behavior — code, tools,
prompts, skills, rules — is baked into the image. Only endpoint URLs and
credentials are injected at runtime.

```sh
make build   # Build the container image
make deploy  # Deploy to OpenShift via Helm
```

See `Makefile` and `chart/` for details.

## Development

This agent was scaffolded using the `agent-loop` template via the `fips-agents`
CLI. The slash command workflow in `.claude/commands/` guides development:

```
/plan-agent   → design the agent
/create-agent → scaffold from the plan
/add-tool     → add a new tool
/add-skill    → add a new skill
/exercise-agent → test agent behavior
/deploy-agent → build and deploy to OpenShift
```
