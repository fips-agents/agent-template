# Problem

Building AI agents that run on OpenShift requires too much boilerplate. A developer who wants to build an agent that checks inventory levels, processes documents, or coordinates emergency response spends most of their time on infrastructure wiring -- model client setup, tool registration, MCP connections, Helm charts, container configuration -- before writing a single line of agent logic. The work that actually differentiates an agent (prompt engineering, model selection, RAG connections, evals) gets squeezed into whatever time remains.

This is not a tooling gap at the infrastructure level. The infrastructure exists and works well. The gap is in the layer between infrastructure and agent logic -- the developer experience layer.

## What Exists Today

**rh-ai-quickstart/ai-architecture-charts** deploys the services agents need: vLLM for inference, LlamaStack for orchestration and guardrails, PGVector for vector storage, MinIO for object storage. These composable Helm charts are actively maintained and solve the infrastructure problem. What they do not provide is any developer-facing scaffolding, agent abstractions, or guidance on how to build an agent that consumes those services.

**red-hat-data-services/agentic-starter-kits** offers eight sample agent implementations spanning LangGraph, LlamaIndex, CrewAI, AutoGen, and vanilla Python. These are useful as references but they are a collection, not a scaffold. Each sample follows a different pattern. There is no shared base class, no common deployment model, and no way to generate a new agent from a template.

**LangGraph templates** provide scaffolding through `langgraph new`, with reference apps for ReAct, memory, and retrieval patterns. The tooling and community are strong. However, there is no OpenShift deployment story, no FIPS awareness, no LlamaStack integration, and the observability path runs through LangSmith rather than enterprise-standard OpenTelemetry.

**OpenAI Agents SDK** demonstrates clean abstractions -- an Agent class with a Runner, multi-agent handoffs, and tool calling -- but it is optimized for OpenAI's ecosystem and has no Kubernetes deployment story.

## The Missing Layer

Each of these projects solves part of the problem. None of them provides the full path from "I want to build an agent" to "my agent is running on OpenShift, talking to vLLM through LlamaStack, with tools, prompts, skills, and evals in place." That path requires: CLI scaffolding that produces a working project, a base class that handles common agent concerns, an LLM client that is portable across providers, a deployment model that works on OpenShift, and integration points for enterprise services like MemoryHub.

No existing project combines these. The infrastructure exists. The framework abstractions exist. The glue layer for enterprise developers is what is missing, and that is what this project provides.

## Who This Affects

**Agent developers** building on OpenShift AI spend days on boilerplate before their agent does anything useful. They need a scaffold that gives them a running agent in minutes and lets them focus on the logic that matters.

**Teams with multi-provider requirements** need agents that are portable across LlamaStack, raw vLLM, and cloud providers without code changes. Today, switching providers means rewriting the client layer.

**Organizations standardizing on the fips-agents ecosystem** need a consistent, repeatable way to create new agents that follow established patterns for deployment, configuration, and observability.
