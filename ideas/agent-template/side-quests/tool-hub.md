# Side Quest: Tool Hub

Captured 2026-04-09 during agent-template ideation. Not in scope for the agent template project, but worth pursuing as a separate project once agents are deploying reliably.

## Vision

An enterprise tool registry living in OpenShift AI. Agents register to use tools. Tools have RBAC, logging, audit trails. Defective tools can be quarantined. Bad agents can be quarantined.

## The Forensic Analysis Story

In an enterprise, you're doing forensic analysis on a problem:
1. You see that agent_id 5 acted on behalf of driver_id 2 and caused a certain issue
2. In the trace, you see that tool 8 was used
3. You review similar traces where the same problem occurred
4. You conclude that tool 8 is defective
5. You go to Tool Hub and **revoke all access to the tool and quarantine it**
6. You tell the developers to take the logs, fix their tool, and submit it for republishing
7. Meanwhile, agents either get by without the tool or wait — but at least the tool isn't creating new problems

Same pattern applies to agents — quarantine a bad agent.

## Key Capabilities

- Central tool registry with versioning
- Per-tool RBAC with agent permissions
- Full logging and audit trail on every tool invocation
- Tool quarantine (revoke access across all agents)
- Agent quarantine (disable a bad agent)
- Forensic tracing: agent_id → driver_id → tool_id
- Tool republishing workflow (fix → test → republish)

## Prerequisites

- Solid agent patterns (this is what agent-template provides)
- Reliable agent deployment on OpenShift
- Established logging/tracing via LlamaStack OTel

## Relationship to Agent Template

The agent template's design enables Tool Hub later:
- Every tool call goes through BaseAgent (logging already captured)
- Tool visibility attributes (agent_only, llm_only, both) map to registry metadata
- RBAC hookpoints exist in BaseAgent's tool dispatch
- When Tool Hub exists, BaseAgent can be updated to check permissions against the registry instead of local config
