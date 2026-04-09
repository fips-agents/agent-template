# Vision

A developer runs `fips-agents create agent my-agent`, picks "agent loop" or "agentic workflow," and gets a project that:

1. **Compiles and runs immediately** — no setup beyond configuring an LLM endpoint
2. **Deploys to OpenShift via Helm** — `helm install` and it's running
3. **Connects to vLLM/LlamaStack transparently** — OpenAI-compatible API, swap providers by changing a URL
4. **Has all the scaffolding in place** — prompts/, tools/, evals/, AGENTS.md, skills, rules, commands
5. **Lets the developer focus on what matters** — write 20-30 lines in their agent subclass, craft prompts, define tools, run evals

## What Changes

- Creating a new agent goes from "days of boilerplate" to "minutes to first working agent"
- Agent code is tiny because BaseAgent handles all the common concerns
- Switching between LlamaStack, raw vLLM, or a cloud model is a config change, not a code change
- Teams standardize on a common agent structure, making it easier to review, share, and maintain agents
- The ecosystem layers (memory-hub, LlamaStack guardrails, tracing) are opt-in infrastructure decisions, not agent code decisions

## What Success Looks Like

- A developer with no agent experience can scaffold and deploy a working agent in under an hour
- Agent subclasses stay small (20-30 lines) even as agents grow in capability
- The template is the default way agents get built in the fips-agents ecosystem
