# fipsagents

Production-ready AI agent framework for FIPS/OpenShift environments. Provides `BaseAgent` — a pure Python, async-first base class that handles LLM communication, tool dispatch, MCP connections, prompt loading, skill management, configuration, and lifecycle so your agent subclass stays small.

## Install

```bash
pip install fipsagents
```

With optional MemoryHub support:

```bash
pip install fipsagents[memory]
```

## Quick start

```python
from fipsagents.baseagent import BaseAgent, StepResult

class MyAgent(BaseAgent):
    async def step(self) -> StepResult:
        response = await self.call_model()
        return StepResult.done(response.content)

import asyncio
asyncio.run(MyAgent().start())
```

## What's included

- **LLM client** via litellm — one interface for 100+ providers (vLLM, LlamaStack, OpenAI, Anthropic, Azure, Bedrock)
- **Two-plane tool system** — `@tool` decorator with `agent_only`, `llm_only`, or `both` visibility
- **MCP client** via FastMCP v3 — connect to remote tool servers
- **Prompt loading** — Markdown with YAML frontmatter
- **Skills** — agentskills.io progressive disclosure
- **Configuration** — YAML with `${VAR:-default}` env var substitution
- **MemoryHub** — optional persistent memory (dual-path: MCP for LLM, SDK for agent code)
- **Protective patterns** — max iterations, exponential backoff, rate limiting

## Used by

This package is the shared framework for templates scaffolded by the [fips-agents CLI](https://github.com/redhat-ai-americas/agent-template):

- **agent-loop** — single-agent loop (`step()` in a loop)
- **workflow** — directed graph of nodes with typed state

## License

Apache 2.0
