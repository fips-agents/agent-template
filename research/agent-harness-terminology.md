# Agent Harness: Terminology Research

**Date:** 2026-04-14
**Purpose:** Inform how we describe BaseAgent in project documentation.

## The Core Formula

By early 2026, a consensus definition has emerged across industry, academia,
and open-source communities:

> **Agent = Model + Harness**

The model provides intelligence. The harness is everything else — the
orchestration loop, tool dispatch, context management, memory, permissions,
error handling, and guardrails. As Anthropic's engineering team put it when
describing their managed agents architecture: "We virtualized the components
of an agent: a session (the append-only log of everything that happened), a
harness (the loop that calls Claude and routes Claude's tool calls to the
relevant infrastructure), and a sandbox." [1]

Sebastian Raschka offers a practitioner's summary: the harness is "the
software layer around the model that assembles prompts, exposes tools, tracks
file state, applies edits, runs commands, manages permissions, caches stable
prefixes, stores memory, and many more." [2]

## Harness vs. Framework vs. Runtime

LangChain proposed a three-level taxonomy in October 2025 that has gained
traction [3]:

- **Agent Runtime** — infrastructure-level execution: durable execution,
  persistence, resilience. The lowest layer. "Agent runtimes are generally
  lower level than agent frameworks and can power agent frameworks."
- **Agent Framework** — abstractions and mental models for building agents.
  Most packages that help build with LLMs fall here.
- **Agent Harness** — opinionated, batteries-included solutions. "It's more
  than a framework — it comes with batteries included."

A 2026 arXiv preprint by Bui adds a useful distinction between *scaffolding*
and *harness* as sequential phases rather than synonyms: scaffolding "assembles
the agent (system prompt, tool schemas, subagent registry) before the first
prompt," while the harness "orchestrates tool dispatch, context management,
and safety enforcement at runtime." [4]

## Harness Quality and Agent Performance

Early benchmarking suggests that harness design has outsized impact on agent
performance. Stanford and UW-Madison researchers found that "changing the
harness around a fixed large language model can produce a 6x performance gap
on the same benchmark" — though this was measured on specific coding benchmarks,
not general agent tasks [5]. LangChain similarly reported moving their coding
agent from outside the top 30 to the top 5 on TerminalBench 2.0 "by only
changing the harness" [6].

Anthropic's Prithvi Rajasekaran frames this as a design principle: "Every
component in a harness encodes an assumption about what the model can't do on
its own, and those assumptions are worth stress testing." [7] Philipp Schmid
takes this further, arguing that as models improve, harnesses must get thinner:
"Developers must build harnesses that allow them to rip out the smart logic
they wrote yesterday." [8] These two observations are complementary — the
harness matters enormously today, but should be designed to shed complexity as
model capabilities grow.

## The OS Analogy

Multiple authors map the harness to an operating system, building on Andrej
Karpathy's 2023 "LLM OS" concept. The LLM is the CPU, the context window is
RAM, retrieval/storage is the filesystem, tools are device drivers, and the
harness is the OS kernel mediating between them. Schmid writes: "The Agent
Harness is the Operating System: It curates the context, handles the 'boot'
sequence (prompts, hooks), and provides standard drivers (tool handling)." [8]

Birgitta Bockeler of Thoughtworks extends this with the concept of "ambient
affordances" — structural properties of the working environment that make it
legible and navigable to agents. A good harness is not just about wrapping the
model; it's also about shaping the environment the model operates in. [9]

## Implications for BaseAgent

BaseAgent sits squarely in harness territory. It provides the agentic loop,
two-plane tool dispatch, prompt loading, skill discovery, configuration
management, and optional MemoryHub integration — everything around the model.
It does not provide the model itself (that's litellm's job) or the execution
infrastructure (that's OpenShift).

Using the LangChain taxonomy: BaseAgent is a lightweight *harness* that
intentionally avoids becoming a *framework*. It is opinionated about structure
(tool planes, prompt format, config shape) but for single-agent use cases,
provides no abstractions that lock you into a particular orchestration pattern.
You subclass BaseAgent with 20-30 lines and get a working agent. The opinions
are about deployment and operations, not about how you think. (The workflow
template adds a directed-graph orchestration layer for multi-agent use cases,
but that is opt-in and built on top of the same harness.)

Whether to call it a "harness" or "proto-harness" depends on scope. A full
harness in the Anthropic or LangChain sense includes session management,
sandboxing, and durable execution. BaseAgent handles the orchestration loop
and tool dispatch but delegates session persistence to MemoryHub and execution
sandboxing to the container platform. "Harness" is accurate; "lightweight
harness" is precise.

## References

[1] L. Martin, G. Cemaj, M. Cohen. "Scaling Managed Agents: Decoupling the Brain from the Body." Anthropic Engineering, Feb 4 2026 (updated Apr 10 2026). https://www.anthropic.com/engineering/managed-agents

[2] S. Raschka. "Components of a Coding Agent." Apr 4 2026. https://magazine.sebastianraschka.com/p/components-of-a-coding-agent

[3] LangChain. "Agent Frameworks, Runtimes, and Harnesses — Oh My!" Oct 25 2025. https://blog.langchain.com/agent-frameworks-runtimes-and-harnesses-oh-my/

[4] N. D. Q. Bui. "Building AI Coding Agents for the Terminal." arXiv preprint, Mar 5 2026. https://arxiv.org/html/2603.05344v1

[5] Y. Lee, R. Nair, Q. Zhang, K. Lee, O. Khattab, C. Finn. "Meta-Harness: End-to-End Optimization of Model Harnesses." Stanford / UW-Madison, Mar 30 2026. https://arxiv.org/html/2603.28052v1

[6] V. Trivedi. "The Anatomy of an Agent Harness." LangChain, Mar 10-11 2026. https://blog.langchain.com/the-anatomy-of-an-agent-harness/

[7] P. Rajasekaran. "Harness Design for Long-Running Application Development." Anthropic Labs, Mar 24 2026. https://www.anthropic.com/engineering/harness-design-long-running-apps

[8] P. Schmid. "The Importance of Agent Harness in 2026." Jan 5 2026. https://www.philschmid.de/agent-harness-2026

[9] B. Bockeler. "Harness Engineering for Coding Agent Users." Thoughtworks / martinfowler.com, Apr 2 2026. https://martinfowler.com/articles/harness-engineering.html
