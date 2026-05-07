# Vision

A developer runs `fips-agents create agent my-agent`, selects a template variant, and gets a project that compiles, runs locally, and deploys to Red Hat AI. The scaffolded project includes an agent subclass of roughly 20-30 lines, a set of prompts, a tools directory, an evals directory, a Helm chart, and AI-assisted slash commands that guide development from design through deployment. The developer's job from that point forward is to write agent logic, craft prompts, define tools, and run evals. Everything else is handled.

## What Changes

**Time to first working agent drops from days to minutes.** Today, a developer starting a new agent project spends significant time on client setup, tool plumbing, deployment configuration, and container builds before the agent does anything useful. With this template, the first `fips-agents create agent` command produces a project that runs immediately against any configured LLM endpoint.

**Agent code stays small even as capability grows.** BaseAgent handles LLM communication, tool dispatch, MCP connections, prompt loading, skill management, memory access, and lifecycle. The subclass implements only the agent's unique behavior. Adding a new tool means writing a decorated function in `tools/`. Adding a new prompt means dropping a Markdown file in `prompts/`. Adding memory means running `memoryhub config init`. None of these changes require modifying the agent subclass.

**Switching LLM endpoints is a configuration change.** The same agent code works against vLLM, LlamaStack, llm-d, or any OpenAI-compatible endpoint. Moving from a local vLLM instance to a cloud-hosted model for comparison testing requires changing a model name and endpoint URL in `agent.yaml`, not rewriting code.

**Teams converge on a shared structure.** When every agent follows the same directory layout, deployment pattern, and base class, code review becomes faster, onboarding becomes simpler, and agents become interchangeable in multi-agent architectures. A developer who has worked on one agent can immediately navigate any other.

**Enterprise capabilities are infrastructure decisions, not code decisions.** MemoryHub, LlamaStack guardrails, tracing, and other enterprise layers are opt-in services that the agent consumes through configuration. Adding shared memory to an agent does not mean importing a library and writing integration code; it means running a config wizard and pointing the agent at an endpoint.

## Success Criteria

The template is successful when a developer with no prior agent-building experience can scaffold and deploy a working agent to Red Hat AI in under an hour. It is successful when agent subclasses remain 20-30 lines even for production agents with multiple tools, skills, and memory integration. It is successful when it becomes the default way agents are built in the fips-agents ecosystem -- not because it is mandated, but because it is easier than the alternatives.
