# Maturation Lifecycle Example

Demonstrates stage-gated routing in a workflow: proto-agents can only
propose skills for review, while apprentice+ agents can learn them
directly.

## What it shows

- **Conditional routing by maturation stage** — `CheckStageNode` reads
  the agent's stage and `route_by_stage` sends proto-agents to
  `SuggestSkillNode` and apprentice+ agents to `LearnSkillNode`.
- **Self-healing + work items composition** — `suggest_skill` creates a
  `review_pending` work item so proposals enter the review queue.
- **Prompt assembly** — `identity.md` describes stage-aware behavior,
  loaded as layer 0 via `prompt_assembly:` config.
- **Configuration composition** — `agent.yaml` enables self-healing,
  maturation, work items, sessions, and traces together.

## Graph

```
check_stage ──┬── (proto_agent) ──→ suggest ──→ END
              └── (apprentice+) ──→ learn   ──→ END
```

## Running

```bash
cd templates/workflow
python -m examples.maturation.agent
```

## What to observe

1. **Proto-agent run**: The agent routes to `SuggestSkillNode` and
   reports a proposal.
2. **Apprentice run**: Same input routes to `LearnSkillNode` and reports
   a learned skill.

In a real deployment, the maturation stage comes from `TrustManager` via
`MaturationManager.current_stage()`. The example sets it on the input
state for demonstration.

## Stage permissions

| Stage | Trust Level | can_create_skills | review_gate | can_edit_own | can_delete_own |
|-------|------------|-------------------|-------------|-------------|---------------|
| Proto-agent | 0 | No | — | No | No |
| Apprentice | 1 | Yes | human_review | No | No |
| Journeyman | 2-3 | Yes | peer_review | Yes | No |
| Specialist | 4+ | Yes | audit_only | Yes | Yes |

## Files

| File | Purpose |
|------|---------|
| `state.py` | `MaturationState` — typed workflow state |
| `agent.py` | Nodes, routing, graph wiring |
| `agent.yaml` | Config with self-healing + maturation + work items |
| `identity.md` | Stage-aware agent identity |
| `tests/test_maturation_workflow.py` | Stage-gated routing tests |
