# Workflow Development Rules

This project uses the workflow framework. Follow these conventions:

- Your workflow definition lives in `src/agent.py`. The framework lives in `src/workflow/` — do not edit it.
- Use `BaseNode` for routing, transformation, and gating logic (no LLM needed).
- Use `AgentNode` for nodes that need LLM, tools, prompts, memory, or MCP.
- The core method is `process(state) -> state`. Do NOT implement `step()` on AgentNodes.
- State is a Pydantic model with `extra="forbid"`. Keep state minimal — data only, no metadata.
- Tools go in `tools/`, one file per tool, using the `@tool` decorator with a `visibility` parameter.
- Prompts go in `prompts/`, one file per prompt, as Markdown with YAML frontmatter.
- Run `make test` and `make lint` before committing.
