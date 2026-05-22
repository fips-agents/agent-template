---
name: session-continuity
description: Resume and hand off work across agent sessions using the work-item pool
version: "1.0"
triggers:
  - resume work
  - hand off
  - pick up where I left off
  - check for work
  - session continuity
dependencies: []
parameters: {}
---

# Session Continuity

Manages seamless work transitions across agent sessions by following a
structured resume and handoff protocol with the work-item pool.

## When to activate

- At the start of a session when work items may be available.
- When the agent's budget or time is running low and it needs to hand
  off cleanly.
- When the agent encounters a blocker and needs to release its current
  work item.

## Resume steps

1. Call `check_available_work` to see pending items.
2. Review any handoff notes from previous attempts.
3. Verify the environment (services, data, outputs from prior work).
4. Call `checkout_work_item` to claim the most appropriate item.
5. Plan your approach based on the handoff note and current state.

## Handoff steps

1. Call `update_work_progress` with your current status.
2. Verify your work (run tests, check outputs).
3. If done: `complete_work_item` with results and accomplishments.
4. If blocked or out of budget: `release_work_item` with a structured
   handoff note.
