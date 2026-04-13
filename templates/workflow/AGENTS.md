# Workflow Template Agent

## Overview

This is a workflow agent that composes multiple nodes into a directed graph.
Each node processes typed state and passes it to the next node in the graph.

## Capabilities

- Classifies input queries by complexity
- Routes to appropriate processing pipeline
- Performs research using LLM and tools
- Summarizes results

## Tools

Tools are available to AgentNode instances within the workflow. See `tools/` for available tool implementations.

## Input / Output

<!-- Populated by /create-agent from the Interaction Model section of AGENT_PLAN.md -->

## Configuration

Agent behavior is controlled by `agent.yaml`. All values support
`${VAR:-default}` environment variable substitution so that the configuration
structure stays baked into the container image while environment-specific
values come from OpenShift ConfigMaps and Secrets at deploy time.

See `agent.yaml` for the full schema.

## Dependencies

The workflow requires an LLM endpoint that speaks the OpenAI-compatible chat
completions API. The template ships with `litellm` as the LLM client, which
supports 100+ providers (vLLM, LlamaStack, Anthropic, OpenAI, Azure, Bedrock,
and others) via model string prefix.

The workflow framework has no dependency on LangChain, LangGraph, or any
external agent framework.

## Deployment

```sh
make build   # Build the container image
make deploy  # Deploy to OpenShift via Helm
```

See `Makefile` and `chart/` for details.

## Development

This workflow was scaffolded using the `workflow` template via the `fips-agents`
CLI. The slash command workflow in `.claude/commands/` guides development:

```
/plan-agent   -> design the workflow
/create-agent -> scaffold from the plan
/add-tool     -> add a new tool
/add-skill    -> add a new skill
/exercise-agent -> test workflow behavior
/deploy-agent -> build and deploy to OpenShift
```
