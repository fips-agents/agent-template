# agent-template

A BaseAgent built on the `agent-loop` template.

## Version

0.1.0

## Capabilities

<!-- Describe what your agent does -->

## Tools

<!-- Document your agent's tools here. Example:

### tool_name

**Visibility:** `llm_only` | `agent_only` | `both`

| Parameter | Type   | Description      |
|-----------|--------|------------------|
| `param`   | string | What it controls |

-->

## Input / Output

<!-- Describe expected inputs and outputs -->

## Configuration

Agent behavior is controlled by `agent.yaml`. All values support
`${VAR:-default}` environment variable substitution so that the configuration
structure stays baked into the container image while environment-specific
values come from OpenShift ConfigMaps and Secrets at deploy time.

See `agent.yaml` for the full schema.

## Dependencies

The agent requires an LLM endpoint that speaks the OpenAI-compatible chat
completions API. The template ships with `litellm` as the LLM client, which
supports 100+ providers (vLLM, LlamaStack, Anthropic, OpenAI, Azure, Bedrock,
and others) via model string prefix.

The agent has no dependency on LangChain, LangGraph, or any agent framework.

## Deployment

```sh
make build   # Build the container image
make deploy  # Deploy to OpenShift via Helm
```

See `Makefile` and `chart/` for details.

## Development

This agent was scaffolded using the `agent-loop` template via the `fips-agents`
CLI. The slash command workflow in `.claude/commands/` guides development:

```
/plan-agent   -> design the agent
/create-agent -> scaffold from the plan
/add-tool     -> add a new tool
/add-skill    -> add a new skill
/exercise-agent -> test agent behavior
/deploy-agent -> build and deploy to OpenShift
```
