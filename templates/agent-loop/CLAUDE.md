# CLAUDE.md

This is an AI agent project built on the BaseAgent framework. The agent runs as an async loop: `setup()` initializes all subsystems, `run()` calls your `step()` method repeatedly, `shutdown()` cleans up.

## Development Workflow

```bash
make install       # Create .venv, install all dependencies
make run-local     # Run the agent locally
make test          # Run pytest
make test-cov      # Run pytest with coverage report
make eval          # Run eval cases from evals/evals.yaml
make lint          # Lint with ruff
make build         # Build container (podman, linux/amd64)
make deploy PROJECT=<ns>   # Deploy to OpenShift
```

## Slash Command Workflow

The commands form two tracks: a **scaffolding pipeline** that takes you from idea to deployment, and **extension commands** for adding capabilities after the agent exists.

### Scaffolding Pipeline

Each step produces an artifact that the next step consumes. Run them in order.

**`/plan-agent`** -- The entry point. Runs a structured design conversation (purpose, tools, prompts, eval cases) and produces `AGENT_PLAN.md`. No code is written. The developer must approve the plan before proceeding.

**`/create-agent`** -- Reads `AGENT_PLAN.md` and generates everything: `src/agent.py`, all tools in `tools/`, prompts, skills, rules, updated `agent.yaml`. Replaces the example Research Assistant with your agent. Runs `make test` and `make lint` to verify before handing back. Will refuse to start if `AGENT_PLAN.md` is missing.

**`/exercise-agent`** -- Reads the full implementation (agent subclass, tools, prompts, rules, skills) and designs 7+ test scenarios across happy paths, edge cases, and failure modes. Supports two modes: **live** (calls the LLM) and **dry-run** (traces step() logic structurally). Writes eval cases to `evals/evals.yaml` so `make eval` can re-run them later.

**`/deploy-agent`** -- Pre-flight checks (tests pass, no uncommitted changes, no hardcoded URLs), builds the container, pushes to a registry, deploys via Helm, and verifies pod startup. On macOS, recommends a remote x86_64 build since podman defaults to ARM64.

### Extension Commands

Run these any time after `/create-agent` to add capabilities incrementally.

**`/add-tool`** -- Asks what the tool does, who calls it (LLM or agent code), its parameters, and sync vs async. Generates the tool file, verifies registry discovery, and updates `src/agent.py` or `prompts/system.md` depending on visibility.

**`/add-skill`** -- Gatekept: if the capability is just a function, it redirects to `/add-tool`; if it's just a template, it suggests adding a prompt file. Skills are for capabilities with their own instructions, scripts, or references that are too large to keep in context permanently. Creates the skill directory with `SKILL.md` following agentskills.io progressive disclosure (frontmatter loads at startup, body loads on activation).

**`/add-memory`** -- Wires MemoryHub via two paths: SDK (`self.memory`) for agent-code memory operations, and MCP tools for LLM-initiated memory calls. Generates `.memoryhub.yaml`, memory hygiene rules, and shows the code patterns for reading/writing memories in `step()`. Gracefully degrades to `NullMemoryClient` if MemoryHub is unavailable.

### The Iterative Loop

After the initial scaffold, development follows a cycle:

```
/add-tool or /add-skill or /add-memory
  -> /exercise-agent (re-validate with new capabilities)
  -> /deploy-agent (ship the update)
```

Each command enforces prerequisites from the previous step and tells you what to run next.

## Project Structure

```
src/agent.py           # YOUR agent subclass — most work happens here
src/fipsagents/baseagent/        # Framework — do not edit
tools/                 # One @tool-decorated .py file per tool
prompts/system.md      # System prompt (required). Add more prompts as needed.
skills/<name>/SKILL.md # One directory per skill, agentskills.io spec
rules/                 # Plain Markdown, one constraint per file
agent.yaml             # Config with ${VAR:-default} env var substitution
chart/                 # Helm chart for OpenShift deployment
evals/                 # Eval cases and runner
```

## Writing Your Agent Subclass

Your agent is a subclass of `BaseAgent` that implements `step()`. Everything else is inherited.

```python
from fipsagents.baseagent import BaseAgent, StepResult

class MyAgent(BaseAgent):
    async def step(self) -> StepResult:
        response = await self.call_model()

        # Process LLM-initiated tool calls (uses tools.execute directly —
        # this is the LLM's dispatch path; for agent-code-initiated calls,
        # use self.use_tool() instead)
        while response.tool_calls:
            # Append assistant message first -- tool_results must follow a tool_use.
            self.messages.append({
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in response.tool_calls
                ],
            })
            for tc in response.tool_calls:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                result = await self.tools.execute(tc.function.name, **args)
                self.messages.append({
                    "role": "tool",
                    "content": result.result,
                    "tool_call_id": tc.id,  # REQUIRED by OpenAI-compatible APIs
                })
            response = await self.call_model()

        return StepResult.done(response.content)
```

### Key BaseAgent Methods

| Method | Purpose |
|--------|---------|
| `start()` | Full lifecycle: setup + run + shutdown with guaranteed cleanup. Recommended entry point. |
| `call_model(messages=None, *, tools=None, include_tools=True, **kw)` | Basic LLM completion. Defaults to `self.messages`. Pass `include_tools=False` to suppress tool schemas. |
| `call_model_json(schema, messages=None, **kw)` | Structured output. `schema` is a Pydantic model or dict. |
| `call_model_stream(messages=None, **kw)` | Async iterator of content chunks. |
| `call_model_validated(validator_fn, messages=None, *, max_retries=3, **kw)` | Call model, validate, retry with backoff. `validator_fn(ModelResponse) -> T` must raise to trigger retry. |
| `use_tool(name, **kwargs)` | Dispatch a tool call from agent code (plane 1). |
| `get_tool_schemas()` | OpenAI-compatible schemas for LLM-visible tools. |
| `build_system_prompt()` | Assembles system prompt + rules + skill manifest. |
| `connect_mcp(server_url)` | Connect to an MCP server via FastMCP v3. |
| `add_message(role, content)` | Append to conversation history. |
| `get_messages()` | Return current conversation history. |
| `clear_messages()` | Reset conversation history. |

## Tool System (Two Planes)

Every tool declares its visibility:

| Visibility | Who calls it | Use for |
|-----------|-------------|---------|
| `llm_only` | LLM decides via tool-calling | Search, retrieval, information gathering |
| `agent_only` | Agent code via `self.use_tool()` | Validation, formatting, internal logic |
| `both` | Either | Rare — only when genuinely needed by both |

```python
from fipsagents.baseagent import tool

@tool(description="Search the web for information", visibility="llm_only")
async def web_search(query: str) -> str:
    """Search for relevant information.

    Args:
        query: The search query string.
    """
    ...
```

Conventions:
- One file per tool in `tools/`. Files starting with `_` are skipped.
- Type hints are mandatory -- the registry builds JSON schemas from them.
- Google-style docstring `Args:` sections become per-parameter descriptions.
- `async def` for anything with I/O. Sync functions run in a thread executor.
- MCP-discovered tools default to `llm_only` regardless of `tools.visibility_default`.

## Prompt Format

Markdown with YAML frontmatter in `prompts/`:

```markdown
---
name: system
description: Main system prompt
variables:
  - name: context
    required: true
  - name: max_length
    default: "500 words"
---

You are an assistant. {context}

Limit responses to {max_length}.
```

`build_system_prompt()` loads `prompts/system.md`, appends all rules, and appends the skill manifest.

## Skills (agentskills.io)

```
skills/summarize/
  SKILL.md      # Required. YAML frontmatter + Markdown body.
```

Frontmatter: `name`, `description`, `version`, `triggers`, `dependencies`, `parameters`.

Only frontmatter is loaded at startup (~100 tokens per skill). Full content loads on activation. Do not create a skill for something that should be a tool or a prompt.

## Rules

Plain Markdown in `rules/`. No frontmatter. Filename is the identifier. One constraint per file. All rules load at startup and are injected into the system prompt.

## Configuration (`agent.yaml`)

Uses `${VAR:-default}` for env var substitution. All deployment-variable values should use this pattern. Key env vars:

- `MODEL_ENDPOINT` -- LLM API endpoint
- `MODEL_NAME` -- Model identifier
- `MAX_ITERATIONS` -- Agent loop cap
- `LOG_LEVEL` -- Python logging level

The agent runs locally with zero external config using the defaults.

## Common Mistakes

- **Do not import `openai` directly.** Use litellm through BaseAgent's `call_model*` methods.
- **Do not import LlamaStack libraries.** LlamaStack is an external endpoint, not a library dependency.
- **Do not hardcode model names or endpoints.** Use `agent.yaml` with `${VAR:-default}`.
- **Do not skip `visibility` on tools.** Every tool must declare its plane.
- **Do not omit `tool_call_id` when appending tool results.** The API requires it.
- **Do not create ConfigMaps for prompts.** Prompts are baked into the image for traceability.
- **Do not build on macOS without `--platform linux/amd64`.** Use `make build` (sets it automatically).
- **Do not use `self.use_tool()` for LLM-originated tool calls.** Those go through `self.tools.execute()` in the tool-call loop. `self.use_tool()` is for agent-code-initiated calls (plane 1).
- **Do not edit `src/fipsagents/baseagent/`.** It is the framework. Your code goes in `src/agent.py`, `tools/`, `prompts/`, `skills/`, and `rules/`.

## Deployment

1. `make test` -- tests must pass
2. `git status` -- no uncommitted changes
3. `make build IMAGE_NAME=<name> IMAGE_TAG=<tag>`
4. Push image to registry
5. Configure `chart/values.yaml` (image reference, env overrides, secrets)
6. `make deploy PROJECT=<namespace>`
7. Verify: `oc get pods -n <ns>`, `oc logs <pod> -n <ns>`

The image is immutable: code, tools, prompts, skills, rules, and `agent.yaml` defaults are all baked in. Only env var overrides (via ConfigMap) and secrets are injected at runtime.

## Dependencies

- **litellm** -- LLM client (provider-portable)
- **fastmcp** (v3) -- MCP client
- **pydantic** -- Config validation and structured output schemas
- **pyyaml** -- Config parsing
- **httpx** -- Async HTTP
- **python-frontmatter** -- Prompt/skill file parsing
- **memoryhub** (optional) -- MemoryHub SDK
