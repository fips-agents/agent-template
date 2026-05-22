# Session Continuity Patterns for Long-Running Agents

## Problem

Long-running agents face a fundamental challenge: context windows are
finite, but tasks span hours or days. Each new context window starts
blank. The agent must:

- Understand what it was doing
- Know what has been accomplished
- Know what remains
- Avoid re-doing completed work
- Avoid breaking things that already work
- Make incremental progress rather than attempting everything at once

Anthropic's "Effective Harnesses for Long-Running Agents" (May 2026)
identifies four failure modes:

1. **One-shotting.** The agent tries to do too much in one pass.
2. **Premature victory.** The agent declares work complete without
   verification.
3. **Broken state.** The agent leaves inconsistent state between
   sessions.
4. **False completion.** The agent marks work as done without proper
   testing.

These failure modes apply beyond coding. A research agent that tries to
analyze 50 papers in one context window produces shallow results. A
compliance agent that declares "all policies reviewed" after checking 3
of 20 is premature. A document processing agent that crashes mid-batch
leaves orphaned state. An ops runbook agent that marks "remediation
complete" without verifying the fix has not remediated anything.

Anthropic's paper solves the coding case with a specific harness: an
initializer agent writes a feature list and progress file, a coding
agent reads them each session, works on one feature, tests it with
Puppeteer, and commits. This document generalizes those patterns for
enterprise agents doing non-coding work and maps them to fips-agents
primitives that already exist.

## Anthropic's Solution (Summary)

Two specialized prompts, same agent harness:

**Initializer agent** (first session only):
- Writes a comprehensive feature list (JSON, not Markdown) with
  pass/fail status per feature
- Creates a progress file for session-to-session context transfer
- Sets up `init.sh` for environment initialization
- Makes an initial git commit as a baseline

**Coding agent** (every subsequent session):
- Reads progress file and git log to orient
- Runs a smoke test to verify nothing is broken
- Picks one feature to work on (incremental progress)
- Tests the feature end-to-end (Puppeteer browser automation)
- Updates progress file and commits to git

Key insight: the progress file + git log gives each new context window
just enough context to continue effectively without needing the full
prior conversation.

## Mapping to fips-agents Primitives

fips-agents already has the building blocks for session continuity.
The Anthropic patterns map directly:

### Progress file --> AgentState (#190)

The progress file is untyped free text. AgentState is a typed Pydantic
model with reducer-based recovery. AgentState is strictly better:

- Typed fields prevent the model from silently corrupting state
- Reducers replay events to recover state after crashes
- Schema versioning detects incompatible state and starts fresh
- State is checkpointed per-session via SessionStore

For a document processing agent, this looks like:

```python
class DocProcessingState(AgentState):
    total_documents: int = 0
    processed: list[str] = []       # document IDs
    failed: list[str] = []
    current_document: str | None = None
    last_checkpoint: datetime | None = None
```

For a compliance monitoring agent:

```python
class ComplianceState(AgentState):
    policy_sections: dict[str, str] = {}   # section_id -> status
    findings: list[str] = []               # finding IDs
    last_reviewed_section: str | None = None
    risk_level: str = "unknown"
```

The progress file works for coding because git provides the recovery
mechanism. AgentState works for everything because the framework
provides the recovery mechanism (trace replay through reducers).

### Feature list --> WorkItemStore (#214)

The JSON feature list with pass/fail status is a simplified work item
registry. WorkItemStore generalizes this with priority, capability
matching, lease-based checkout, budget headroom, and structured handoff
notes. See `planning/work-item-coordination-design.md` for the full
design.

For single-agent scenarios where WorkItemStore is overkill, a simpler
pattern works: an AgentState field listing tasks with status.

```python
class ResearchState(AgentState):
    questions: list[ResearchQuestion] = []

class ResearchQuestion(BaseModel):
    id: str
    text: str
    status: str = "pending"       # pending, in_progress, answered, blocked
    answer_summary: str = ""
    sources_checked: list[str] = []
    blockers: list[str] = []
```

The agent reads `questions` at session start, picks one with
`status="pending"`, works on it, and updates the status before session
end. This is the Anthropic feature list pattern expressed as typed
state. Rules enforce the discipline.

### Git as ground truth --> Session persistence + traces

For coding agents, git provides state recovery (revert bad changes)
and audit trail (what happened when). For non-coding agents, these
roles are filled by:

- **SessionStore**: message history across context windows
- **TraceStore**: what the agent did, which tools it called, what
  results it got
- **AgentState checkpoints**: typed state snapshots for fast recovery
- **MemoryHub**: cross-session semantic memory (broader than session
  state)

### Init script --> setup() protocol

The `init.sh` pattern (run the dev server, verify it works) maps to
`setup()` in BaseAgent's lifecycle. But `setup()` handles framework
initialization (MCP connections, memory loading, tool registration),
not domain-specific orientation.

What is missing is a **resume protocol** -- a domain-specific
orientation step that runs at the start of each session. This should
be a rule or skill, not framework machinery. See section 4.

### Smoke test --> Verification step

The "start by testing basic functionality" pattern is domain-specific.
A document processing agent might verify its output store is reachable.
A compliance agent might check that the policy database is current. An
ops agent might verify cluster connectivity.

This maps to a **verification tool** or a section in the resume
protocol rule. The framework provides `setup()` for infrastructure
checks; the agent developer defines domain-specific verification in
rules, skills, or tools.

## The Resume Protocol

This is the key missing pattern. A resume protocol is a structured
startup ritual for agents that work across many sessions. It bridges
the gap between framework-level `setup()` (infrastructure) and
domain-specific work (the agent's actual job).

### Structure

```
Session Start
    |
    +-- setup()                    [framework: MCP, memory, tools]
    |
    +-- Resume Protocol            [agent-level: orientation]
    |   +-- Load state             [read AgentState checkpoint]
    |   +-- Review history         [read recent traces or session messages]
    |   +-- Verify environment     [domain-specific smoke test]
    |   +-- Assess current state   [what is done, what is broken, what is next]
    |   +-- Plan next increment    [pick ONE thing to work on]
    |
    +-- Execute                    [work on the chosen increment]
    |
    +-- Handoff Protocol           [agent-level: cleanup]
    |   +-- Verify work            [test what was just done]
    |   +-- Update state           [checkpoint AgentState]
    |   +-- Write handoff note     [what was done, what is next, any blockers]
    |   +-- Commit/persist         [ensure all state is durable]
    |
    +-- Session End
```

### Why this is not a framework lifecycle hook

The resume protocol is domain-specific. A document processing agent
orients by checking its output store and reading its queue depth. A
compliance agent orients by checking policy update timestamps. An ops
agent orients by reviewing the current alert state.

Making this a framework hook (e.g., `BaseAgent.resume()`) would either
be too generic to be useful or too prescriptive to fit all domains.
Rules and skills are the right mechanism: they are injected into the
system prompt, the model follows them because they are in context, and
agent developers customize them freely.

This is consistent with fips-agents' design principle: the framework
provides primitives, not policy.

### Implementation as a rule

The resume protocol should be a rule template that agent developers
customize:

```markdown
# Resume Protocol

At the start of each session:

1. Read your state checkpoint to understand where you left off.
2. Review the last 5 trace entries to see what happened recently.
3. Run your verification checklist to confirm your environment is
   healthy.
4. If anything is broken from a previous session, fix it before
   starting new work.
5. Choose ONE task to work on this session. Do not attempt multiple
   tasks.
6. State your plan before executing. Name the specific task, what
   success looks like, and how you will verify it.

At the end of each session (or when approaching budget limits):

1. Verify that your current work is in a clean state (no partial
   writes, no orphaned resources, no unclosed connections).
2. Update your state checkpoint with what you accomplished.
3. Write a handoff note: what you did, what you tried that did not
   work, what is next, any blockers you discovered.
4. Ensure all state is persisted (AgentState checkpointed, work items
   updated, artifacts committed).
```

This rule lives in `rules/` and is injected into the system prompt.
The framework does not enforce it -- the model follows it because it
is prompt-level instruction. This is the same mechanism by which
Anthropic's coding agent follows its progress-file discipline: the
prompt says to do it, so the model does it.

### Implementation as a skill

For more complex resume protocols -- agents that need to query external
systems, run diagnostic tools, or make conditional decisions about what
to work on -- a skill is more appropriate than a rule. Skills support
progressive disclosure and can include structured guidance with
reference material.

```
skills/
  session-continuity/
    SKILL.md           # frontmatter + instructions
    resources/
      state-schema.md  # how to define AgentState for this domain
      verification.md  # how to write domain-specific verification
      handoff.md       # how to write effective handoff notes
```

The SKILL.md frontmatter loads at startup (~100 tokens). The full
instructions load when the skill activates. Reference material loads on
demand. This keeps the cost low for agents that do not need the skill
on every session.

## Domain-Specific Patterns

The resume protocol is generic. These patterns show how it applies to
specific non-coding use cases.

### Document processing pipeline

**State shape:**

```python
class DocProcessingState(AgentState):
    batch_id: str = ""
    total: int = 0
    processed: list[str] = []
    failed: dict[str, str] = {}         # doc_id -> error
    current_doc: str | None = None
    output_location: str = ""
```

**Verification:** Output store is reachable. Source documents are
accessible. Previously processed documents still exist in the output
store (detect silent data loss).

**Increment:** Process ONE document (or a small batch of 3-5 if each
takes under a minute). Never attempt the full queue in one session.

**Handoff note:**
```
accomplished: ["Processed invoice-2024-0847.pdf (12 pages, 3 tables extracted)"]
attempted: ["invoice-2024-0848.pdf failed: Docling timeout on page 9 (oversized table)"]
remaining: ["14 documents in queue"]
blockers: ["Docling times out on documents with tables >500 rows"]
artifacts: {"output": "s3://batch-47/invoice-2024-0847/", "error_log": "traces/abc123"}
```

### Compliance monitoring

**State:** `framework` (SOC2, FedRAMP), `sections: dict[str, SectionStatus]` tracking per-section review status and timestamps, `findings: list[Finding]` with severity and remediation status.

**Verification:** Policy database is current. Previous findings are still open. Audit target systems are accessible.

**Increment:** Review ONE policy section or follow up on ONE finding. Depth over breadth.

**Handoff:** Which section was reviewed, how many new findings, what access is missing for deferred sections.

### Research agent

**State:** `questions: list[ResearchQuestion]` with status and evidence, `hypotheses: list[Hypothesis]` with confidence scores and supporting/contradicting evidence, `sources_reviewed: list[str]`.

**Verification:** Search APIs are accessible. Previously retrieved sources are still available (detect link rot).

**Increment:** Investigate ONE question or review ONE source.

**Handoff:** What was learned, which hypotheses were updated, what sources remain, any access blockers (paywalls, rate limits).

### Ops runbook execution

**State:** `incident_id`, `steps: list[RunbookStep]` with per-step status and results, `diagnostics: dict[str, str]`, `escalated: bool`.

**Verification:** Cluster is reachable. Alerting systems are accessible. The incident is still open (do not remediate a resolved incident).

**Increment:** Run ONE diagnostic or execute ONE remediation step. Never batch remediation actions -- each one changes system state and the next step's preconditions may have changed.

**Handoff:** Which step was executed, what was found, what the actual root cause appears to be (may differ from the alert), remaining steps with any ordering constraints.

## Compaction and Memory Interaction

fips-agents has four mechanisms that participate in session continuity.
They serve different purposes, operate at different timescales, and
compose rather than compete.

### Compaction (#166)

Summarizes older messages to free context space within a single
session. The compactor preserves recent turns and system messages.
Compaction summaries are lossy -- they capture what happened but not
the agent's plan, priorities, or assessment of remaining work.

**Compaction alone is not sufficient for session continuity.** This is
Anthropic's core finding. Compaction helps within a single long
session. It does not help when a session ends and a new one begins
with a fresh context window. AgentState and handoff notes bridge the
inter-session gap.

### Memory (MemoryHub, file-based)

Long-term semantic memory. Useful for facts, preferences, and
decisions that persist across projects, not just sessions. Memory is
too slow-changing for session continuity -- it captures "this user
prefers YAML over JSON" not "I am 60% through processing batch #47."

Memory is the right place for lessons learned during multi-session
work: "Docling times out on tables larger than 500 rows -- use the
table-splitting preprocessor." That lesson persists beyond the current
task and benefits future sessions on different tasks.

### AgentState (#190)

Per-session typed state. This is the primary mechanism for session
continuity. State is checkpointed after each turn, recoverable via
trace replay, and schema-versioned. It captures "I am 60% through
processing batch #47" with typed precision.

AgentState has a different lifecycle than memory: it is born with the
session, lives as long as the task runs, and may be archived or
discarded when the task completes. Memory outlives tasks.

### HandoffNote (WorkItemStore, #214)

Per-work-item structured context for the next actor. Handoff notes
bridge not just sessions but agents -- a diagnostic agent's handoff
note is read by a remediation agent that has a completely different
AgentState schema.

The `HandoffNote` model (`accomplished`, `attempted`, `remaining`,
`blockers`, `artifacts`, `context`) provides enough structure for
programmatic reading while allowing free-form context.

### How they compose

An agent starting a new session goes through these layers in order:

```
1. Load memory           [MemoryHub: cross-project, long-lived facts]
                         [eager or lazy per loading_pattern config]

2. Load AgentState       [SessionStore: per-session typed state]
   checkpoint            [what was I doing, how far did I get]

3. Read handoff note     [WorkItemStore: per-work-item context]
   from work item        [what did the last agent accomplish/discover]

4. Benefit from          [Compactor: intra-session, automatic]
   compaction during     [prevents context overflow on long turns]
   execution
```

These are independent. An agent can use AgentState without
WorkItemStore (single-agent, no shared work pool). An agent can use
WorkItemStore without AgentState (stateless workers that rely entirely
on handoff notes). An agent can use memory without either (simple Q&A
with long-term preferences). All four compose additively.

## The Incremental Progress Discipline

The single most impactful pattern from the Anthropic paper: **do one
thing per session, do it well, verify it, document it.**

This is counter-intuitive for capable models. A model that can reason
about 200 features will try to implement all 200. The discipline of
"pick one, finish it, hand off clean" must be enforced through
prompting.

### Why it works

- Smaller scope = higher quality per increment
- Clean handoff = less wasted time re-discovering context
- Verification per increment = problems caught early
- Incremental checkpoints = easy rollback when something goes wrong
- Budget consumed on depth rather than breadth = better outcomes

### Why models resist it

Large models have strong completion bias. When shown a list of 20
items, the instinct is to process all 20. Saying "work through these
items" invites one-shotting. Saying "pick ONE item and complete it
fully before considering any other item" constrains the model. The
resume protocol rule must include this explicit single-item constraint.

### How to enforce it

Four mechanisms, in order of effectiveness:

1. **Rules.** "Choose ONE task from your state. Do not start a second
   task in the same session." This is the primary enforcement and
   works reliably with frontier models. Put it in `rules/` so it is
   always in context.

2. **WorkItemStore checkout semantics.** Lease-based checkout with
   budget headroom naturally constrains scope. The agent checks out
   one item, works on it, and the budget headroom reservation
   guarantees enough capacity for a clean handoff. Checking out a
   second item while holding one is possible but discouraged by the
   budget split.

3. **Per-turn resource limits (#195).** `max_tokens_per_turn` and
   `max_iterations_per_turn` cap individual model calls. They do not
   directly enforce single-item focus, but they prevent runaway
   execution that often accompanies one-shotting.

4. **Doom-loop detection (#167).** Catches agents spinning on
   too-large tasks by detecting repeated tool calls with identical
   arguments. This is a safety net, not a primary enforcement.

### When to relax it

The single-item discipline is a default, not an absolute rule.

- Trivial tasks that take seconds (update a config value, rename a
  field) can be batched. If the total batch takes less time than the
  resume protocol, batching is more efficient.
- Tightly coupled tasks that cannot be separated (create a model and
  its migration, write a function and its test) should be one work
  item with a clear scope boundary.
- Pipeline stages where the output of one task is the input to the
  next and both are fast (parse a document, then extract entities)
  can be a single work item.

Agent developers decide the granularity. The framework provides the
discipline primitives; the rule template provides the default
constraint; the developer overrides as needed for their domain.

## Event-Driven vs. Loop-Based Session Continuity

Two models for how sessions restart:

### Loop-based (Anthropic's model)

An external loop runs the agent repeatedly:

```
while not done:
    run_agent(prompt)
```

Each iteration is a new context window. The progress file bridges
iterations. Simple, works for single-agent coding scenarios where
"done" is well-defined (all features pass).

fips-agents does not have a built-in loop runner. The equivalent is a
shell script or orchestrator that calls the agent's HTTP endpoint
repeatedly. This is intentional -- loop-based continuity is a
deployment pattern, not a framework feature.

### Event-driven (fips-agents model)

CronSource or WebhookSource triggers the agent:

```yaml
server:
  event_sources:
    - type: cron
      schedule: "*/10 * * * *"
      event_type: resume-work
```

Each trigger is a new session (or resumes an existing one via the
`event:cron:resume-work` session key). The agent reads its AgentState,
does work, updates state, and waits for the next trigger.

This is more natural for enterprise deployments:

- Agent runs as a persistent service (container in OpenShift)
- Cron triggers periodic work checks
- Webhooks trigger reactive work (new documents posted, reviews
  completed, alerts fired)
- Sessions persist across triggers via SessionStore
- AgentState survives across triggers via checkpoints
- Safety features (compaction, limits, doom-loop) apply automatically

### Event-driven with WorkItemStore

The event-driven model composes with WorkItemStore naturally:

```
  CronSource fires
       |
       v
  Load AgentState checkpoint
       |
       v
  list_available(capabilities=my_capabilities)
       |
       +-- no items --> emit status log, return, wait for next cron tick
       |
       +-- items available
       |       |
       |       v
       |   checkout(item_id, actor_id=my_session_id)
       |       |
       |       v
       |   Read handoff_note from previous actor (if present)
       |       |
       |       v
       |   Execute work (tool calls, model calls)
       |       |
       |       +-- success --> complete(item_id, result, handoff_note)
       |       +-- partial --> release(item_id, handoff_note)
       |       +-- failure --> fail(item_id, error, handoff_note)
       |
       v
  Checkpoint AgentState
       |
       v
  Wait for next cron tick
```

Multiple replicas of the same agent can run concurrently. The lease
mechanism prevents double-processing. Budget headroom guarantees clean
handoff even when a session is interrupted.

This is the "ambient agent" pattern: the agent is always deployed,
wakes periodically, processes available work, and sleeps. No external
orchestrator is needed beyond the deployment itself.

## Handoff Note Quality

Handoff notes are the bridge between sessions and between agents.
Their quality determines whether the next session starts productively
or wastes time re-discovering context.

### What makes a good handoff note

A good handoff note answers five questions for the next actor:

1. **What was accomplished?** Concrete, verifiable statements. "Processed
   invoice-2024-0847.pdf and extracted 3 tables" not "made progress on
   document processing."
2. **What was attempted but failed?** Including the failure reason. The
   next actor should not retry the same approach unless circumstances
   changed.
3. **What remains?** Scoped to the work item, not the entire project.
4. **What blocks progress?** External dependencies, missing access,
   broken infrastructure, ambiguous requirements.
5. **Where are the artifacts?** References (file paths, URLs, trace IDs,
   commit SHAs) that the next actor can follow without searching.

The `HandoffNote` model in WorkItemStore encodes these five questions as
typed fields. The `context` free-form field handles everything that does
not fit neatly into the structured fields.

### What makes a bad handoff note

- "Made progress." (No specifics, no verification path)
- "Everything is done." (No artifact references, no verification)
- "See the logs." (Which logs? Where? What to look for?)
- Omitting failed attempts. (Next actor wastes time retrying)
- Mixing work-item scope with project scope. ("Also, I noticed the
  database schema needs updating" -- that belongs in a new work item
  or a memory, not this handoff note)

### Enforcing quality

Handoff note quality is a prompting concern. The resume protocol rule
should include examples of good handoff notes and explicitly prohibit
empty or vague notes. For agents using WorkItemStore, the
`complete_work_item` and `release_work_item` tools require structured
fields (`accomplished`, `remaining`) -- the schema itself enforces
minimum structure.

For higher assurance, a review pattern can validate handoff notes: the
agent writes the note, a subagent or a cheaper model checks it against
the acceptance criteria, and the note is revised if it fails. This is
the same adversarial review pattern described in
`planning/work-item-coordination-design.md` applied to handoff quality
rather than work product quality.

## Failure Recovery Patterns

Long-running agents will fail. Pods restart, MCP servers go down,
model endpoints return errors, context windows overflow. The
continuity patterns must handle failure gracefully.

### Crash during execution

The agent dies mid-work. On restart:

1. SessionStore loads the last saved messages.
2. AgentState loads the last checkpoint (which may be stale -- it was
   checkpointed at the end of the previous turn, not at the crash
   point).
3. If state recovery is enabled (#190), traces since the last
   checkpoint are replayed through the reducer to reconstruct the
   most recent state.
4. If WorkItemStore is in use, the lease on the current work item
   will expire (the agent cannot renew it from the grave). The item
   returns to `available` with whatever handoff note was last written
   via `update_progress()`.

The key design property: **no mechanism requires the agent to cleanly
shut down.** Leases expire. Checkpoints are periodic. Trace replay
recovers state from the last checkpoint forward. The worst case is
re-doing the work between the last checkpoint and the crash.

### Model or tool failure

The framework's built-in backoff handles transient model errors. For
persistent failures, the agent should write a handoff note (from
AgentState, no model call needed), checkpoint state, and exit. If an
MCP server goes down mid-session, the agent should record it in
`blockers` and release the work item if the tool is essential. This
is domain-specific policy encoded in rules, not framework behavior.

### State corruption

AgentState deserialization fails (schema changed, data corrupted). The
framework's schema versioning (`state_schema_key()` hash) detects
incompatible schemas and starts fresh. This is safe: schema changes
mean agent code changed, WorkItemStore is independent of AgentState,
and MemoryHub memories survive schema resets.

## Scope Guard (Open Design Question)

The doom-loop detector catches repetitive behavior. It does not catch
scope overreach: an agent that checks out one work item but processes
three others "while it is at it." Possible signals for a scope guard
include in-flight tool call chain count, token consumption rate vs.
single-item baseline, and distinct entity references in recent
messages. This is not proposed for implementation -- the failure mode
is real but the most practical mitigation today is prompt engineering
via the resume protocol rule. Revisit if deployments show prompting
is insufficient.

## Configuration Summary

No new configuration is introduced by these patterns. Everything maps
to existing fips-agents config:

| Pattern | Config location | Relevant settings |
|---------|----------------|-------------------|
| AgentState | `server.state_recovery.enabled` | traces at `standard`+ fidelity |
| Session persistence | `server.sessions.enabled` | `max_age_hours` |
| WorkItemStore | `server.work_items.enabled` | backend, lease_duration, budget_headroom_pct |
| Compaction | `server.compaction.enabled` | threshold_messages, keep_recent_turns |
| Memory | `memory.backend` | loading_pattern, budget, injection_mode |
| Event triggers | `server.event_sources` | type, schedule |
| Per-turn limits | `model.limits` | max_tokens_per_turn, max_iterations_per_turn |
| Doom-loop | `loop.guard` | repeat_threshold, pattern_window |

The resume protocol, handoff note quality, and incremental progress
discipline are all prompt-level patterns (rules or skills), not
configuration.

## Template Deliverables

These patterns should be captured as reusable templates in the
agent-loop scaffold:

### Rule template: `rules/session-continuity.md`

A ready-to-customize rule file that implements the resume protocol and
handoff protocol. Included in the scaffold but commented out (or in a
`rules/examples/` directory). Agent developers enable and customize it
for their domain.

### Skill template: `skills/session-continuity/`

A more detailed version for agents with complex resume requirements.
Includes reference material on writing good AgentState schemas, domain
verification patterns, and handoff note examples. Progressive
disclosure keeps the cost low for agents that do not activate the
skill.

### AgentState examples

Type-complete AgentState examples for common domains (document
processing, compliance, research, ops) in the skill's reference
material. Developers copy and adapt rather than designing from scratch.

These templates are not framework code. They are prompt-level
artifacts that ship with the scaffold and guide agent developers
toward effective session continuity patterns.

## Open Questions

1. **Should the resume protocol be a framework lifecycle hook or a
   prompt-level pattern?** This document argues for prompt-level
   (rule or skill). Framework hooks are rigid; prompt-level patterns
   let developers customize orientation for their domain. The
   counter-argument is that a lifecycle hook could guarantee execution
   (the model might skip a rule). In practice, frontier models follow
   clear rules reliably, and the flexibility of prompt-level patterns
   outweighs the guarantee of a hook.

2. **How does compaction interact with AgentState?** If compaction
   summarizes away the messages that produced a state transition, can
   the reducer still replay? Answer: no, and that is why checkpoints
   exist. State is checkpointed explicitly, not derived from message
   replay. Replay is only for recovery after checkpoint loss, and it
   uses the trace store (which is independent of message history),
   not the compacted messages.

3. **What is the right granularity for work items?** Too coarse (one
   item = "process all 500 contracts") invites one-shotting. Too fine
   (one item = "extract clause 3.1 from document 47") makes
   coordination overhead dominate. Rules of thumb:
   - One work item should be completable in one session (one context
     window worth of work).
   - If an item routinely needs to be released mid-work, it is too
     coarse.
   - If an item takes less time than the resume protocol, it is too
     fine and should be batched.

4. **Should the framework detect one-shotting?** The scope guard
   concept (section 10) is appealing but hard to implement without
   false positives. An agent processing a batch of small items
   quickly looks like one-shotting but is actually correct. Prompting
   remains the most practical enforcement. Revisit if real-world
   deployments show that prompting is insufficient.

5. **How does this interact with subagent-as-tool (#165)?** A parent
   agent that delegates to subagents for individual work items does
   not need session continuity itself -- each subagent has its own
   session and state. The parent's role is decomposition and
   coordination, which is lightweight enough to fit in one context
   window. Session continuity patterns apply to the leaf agents that
   do the actual work.

6. **Should handoff notes be stored in memory?** When a work item
   completes, should the handoff note's `context` field be written to
   MemoryHub as a project-scoped memory? This would make lessons
   learned from one work item discoverable by agents working on
   future items. The risk is memory bloat -- hundreds of completed
   work items producing hundreds of memories. A selective approach
   (write to memory only when `blockers` is non-empty or the item
   required multiple attempts) may be more practical.

## References

- Anthropic, "Effective Harnesses for Long-Running Agents" (May 2026)
- Claude 4 Prompting Guide, "Multi-Context Window Workflows" section
- fips-agents issue #214 (WorkItemStore ABC)
- fips-agents issue #188 (Event-triggered mode)
- fips-agents issue #190 (Reducer-based state recovery)
- fips-agents issue #166 (Compaction)
- fips-agents issue #167 (Doom-loop detection)
- fips-agents issue #195 (Per-turn resource limits)
- fips-agents issue #165 (Subagent-as-tool)
- `planning/work-item-coordination-design.md`
- `planning/event-triggered-design.md`
- `docs/architecture.md` (Session persistence, AgentState, Compaction,
  Memory Integration sections)
