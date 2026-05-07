# Documentation

agent-template scaffolds production-ready AI agents for Red Hat AI via the `fips-agents` CLI. For the project overview and getting started, see the [root README](../README.md).

## Design

- [Architecture](architecture.md) -- System design, BaseAgent class, tool planes, deployment model, and the reasoning behind each decision.
- [Responsibilities & non-goals](responsibilities.md) -- What each adjacent platform layer owns (OGX, kagenti, MemoryHub, OpenShift, sibling repos) and what the main template deliberately does not build. Read before proposing features that may belong in an extension or a sibling project.
- [ADRs](adr/) -- Architectural decision records for individual subsystems. Each ADR records context, decision, alternatives considered, and consequences for one design choice.

## Context

- [Problem](problem.md) -- The ecosystem gap this project fills and who benefits.
- [Vision](vision.md) -- What success looks like and what changes for developers.

## Related Directories

- [planning/](../planning/) -- In-flight design work: requirements, scope, constraints, and next steps.
- [fips-agents/research](https://github.com/fips-agents/research) (private) -- Ecosystem research, competitive analysis, and side-quest investigations.
