# Constraints

- **Python only** — templates are Python-based, async throughout
- **No framework dependencies** — no LangChain, no LangGraph. Pure Python + litellm + FastMCP v3.
- **litellm for LLM calls** — provider-portable, OpenAI-compatible interface
- **Red Hat UBI base images** for all containers
- **FIPS compliance may be required** — need to ask per deployment, but templates must not preclude it
- **OpenShift deployment target** — Helm charts must work on OpenShift
- **fips-agents CLI integration** — template is a public repo that gets cloned; .claude/ directory drives the developer experience
- **Prompts as Markdown with YAML frontmatter** — one file per prompt
- **Tools use @tool decorator** — same convention as FastMCP, auto-discovered
- **Skills follow agentskills.io spec** — directory per skill, SKILL.md with frontmatter
- **Rules as plain markdown** — no frontmatter, filename is identifier
- **FastMCP v3** for MCP client — not v2
- **Immutable container images** — code, tools, prompts, skills, rules all baked in
- **Agent loop first** — workflow template is deferred
- **memoryhub SDK** for programmatic memory access (optional)
