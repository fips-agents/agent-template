# fipsagents

Production-ready AI agent framework for FIPS/OpenShift environments. Provides `BaseAgent` — a pure Python, async-first base class that handles LLM communication, tool dispatch, MCP connections, prompt loading, skill management, configuration, and lifecycle so your agent subclass stays small.

## Install

```bash
pip install fipsagents[server]
```

With optional MemoryHub support:

```bash
pip install fipsagents[server,memory]
```

Or vendor the source directly into your project for full control:

```bash
fips-agents vendor
```

See `VENDORED` marker in `src/fipsagents/` for provenance tracking.

## Quick start

```python
from fipsagents.baseagent import BaseAgent, StepResult

class MyAgent(BaseAgent):
    async def step(self) -> StepResult:
        response = await self.call_model()
        response = await self.run_tool_calls(response)
        return StepResult.done(response.content)

if __name__ == "__main__":
    from fipsagents.server import OpenAIChatServer
    server = OpenAIChatServer(agent_class=MyAgent, config_path="agent.yaml")
    server.run()
```

## What's included

- **LLM client** via the openai async SDK — connects to any OpenAI-compatible endpoint (vLLM, LlamaStack, llm-d)
- **Two-plane tool system** — `@tool` decorator with `agent_only`, `llm_only`, or `both` visibility
- **MCP client** via FastMCP v3 — connect to remote servers (tools, prompts, and resources)
- **Prompt loading** — Markdown with YAML frontmatter
- **Skills** — agentskills.io progressive disclosure
- **Configuration** — YAML with `${VAR:-default}` env var substitution
- **MemoryHub** — optional persistent memory (dual-path: MCP for LLM, SDK for agent code)
- **Protective patterns** — max iterations, exponential backoff, rate limiting
- **HTTP server** — OpenAI-compatible `/v1/chat/completions` endpoint with streaming
- **`run_tool_calls()`** — one-line tool dispatch loop for non-streaming agents
- **Agent identity** — name, description, version exposed via `/v1/agent-info`

## Key methods

| Method | Purpose |
|--------|---------|
| `call_model()` | LLM completion with auto-included tool schemas |
| `run_tool_calls(response)` | Execute tool calls and loop until the model stops |
| `call_model_json(schema)` | Structured output with Pydantic validation |
| `call_model_validated(fn)` | Call, validate, retry with backoff |
| `use_tool(name, **kw)` | Agent-code tool call (plane 1) |
| `connect_mcp(target)` | Connect to an MCP server |
| `get_mcp_prompt(name)` | Render an MCP-provided prompt |
| `read_resource(uri)` | Read an MCP-provided resource |

## Used by

This package is the shared framework for templates scaffolded by the [fips-agents CLI](https://github.com/fips-agents/agent-template):

- **agent-loop** — single-agent loop (`step()` in a loop)
- **workflow** — directed graph of nodes with typed state

## License

Apache 2.0
