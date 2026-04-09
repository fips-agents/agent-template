# Stakeholders

## Primary Users

- **Agent developers** — people building AI agents on OpenShift AI using the fips-agents ecosystem. They want to write agent logic, not boilerplate. Typical agent subclass should be 20-30 lines.

## Secondary Users

- **Platform teams** — teams managing OpenShift AI infrastructure (vLLM, LlamaStack via rh-ai-quickstart). They benefit from agents having a standard deployment pattern (Helm chart, ConfigMap-driven config).

## Affected Parties

- **fips-agents CLI maintainers** — this template becomes part of the CLI's scaffolding. Template structure must align with CLI expectations.
- **rh-ai-quickstart maintainers** — we document their charts as the expected infra layer. Our Helm chart should complement, not conflict with, their deployments.
