# Prompt Assembly, Self-Healing, and Agent Maturation

## Problem

`build_system_prompt()` concatenates three sections -- the main prompt,
rules text, and skill manifests -- separated by `---` dividers. Memory
is injected as a separate message at index 1. This works, but the
assembly is implicit: there is no named layer model, no precedence
semantics, no concept of mutability, and no way for an agent to learn
new procedural knowledge across sessions.

Three problems follow from this:

1. **No separation of identity from behavior.** The system prompt is a
   single Markdown file. An agent's identity ("you are a document
   processing agent"), personality ("respond concisely, cite sources"),
   and operational instructions are conflated. When a developer wants to
   reuse the same agent identity with different behavioral styles for
   different deployment contexts, the entire system prompt must be
   duplicated and edited.

2. **No agent-authored content.** Rules and skills are developer-
   authored and baked into the container image. An agent that discovers
   a recurring failure pattern (Docling times out on tables with 500+
   rows) cannot encode procedural knowledge for future sessions. The
   developer must notice the pattern, write a rule, rebuild the image,
   and redeploy. The feedback loop is weeks instead of minutes.

3. **No maturation path.** A new agent starts at the same trust and
   autonomy level as a year-old agent. There is no framework-supported
   lifecycle from supervised proto-agent to autonomous specialist. The
   enterprise multi-agent coordination doc
   (`enterprise-multi-agent-coordination.md`) describes trust
   accumulation; this document describes the prompt-level mechanisms
   that trust enables.

These problems are related. Formalizing prompt assembly into named
layers with precedence creates the structure needed for selective
mutability. Selective mutability creates the mechanism for agent
self-healing. Self-healing is the runtime expression of a maturation
lifecycle.


## 1. Prompt Assembly

### Layer Taxonomy

The system prompt is assembled from discrete, typed layers. Each layer
has a source, a mutability class, a precedence rank, and an injection
point. Lower precedence numbers indicate higher authority.

```
  Precedence 0  ┌────────────────┐  Immutable
  (highest)     │   Identity     │  who the agent IS
                └───────┬────────┘
  Precedence 1  ┌───────┴────────┐  Immutable
                │  Personality   │  HOW the agent behaves
                └───────┬────────┘
  Precedence 2  ┌───────┴────────┐  Immutable
                │  Governance    │  non-negotiable policies
                └───────┬────────┘
  Precedence 3  ┌───────┴────────┐  Immutable (bundled)
                │ Capabilities   │  + agent-mutable-with-audit (learned)
                └───────┬────────┘
  Precedence 4  ┌───────┴────────┐  Agent-mutable
                │  Knowledge     │  facts, preferences, decisions
                └───────┬────────┘
  Precedence 5  ┌───────┴────────┐  Agent-mutable
                │ Operational    │  current task, progress, handoff
                │ Context        │
                └───────┬────────┘
  Precedence 6  ┌───────┴────────┐  Transient
  (lowest)      │  Ephemeral     │  turn-scoped injections
                └────────────────┘
```

**Identity (precedence 0, immutable).** Who the agent IS. Name, role,
fundamental purpose. Source: `identity.md` at project root, or an
inline `identity:` block in `agent.yaml`. Cannot be overridden by any
lower layer. This is the one thing that survives every prompt change,
skill addition, and memory update. An agent without an identity layer
falls back to the current behavior (the system prompt file serves
double duty).

**Personality (precedence 1, immutable).** HOW the agent behaves.
Communication style, tone, values, behavioral constraints. Source:
`personality.md` (or `soul.md`). This layer is optional -- many agents
do not need personality separate from identity. It exists for the case
where the same agent identity is deployed with different interaction
styles: a formal variant for executive briefings, a technical variant
for engineering teams, a tutoring variant that explains its reasoning.
Same agent, same capabilities, different voice.

**Governance (precedence 2, immutable).** Non-negotiable policies.
Security constraints, compliance requirements, ethical boundaries, data
handling rules. Source: `rules/` directory. Developer-authored only.
These are the "laws" of the agent's behavior. They override everything
below them in precedence. This layer maps directly to the current
`rules/` directory and `RuleLoader`. The only change is formalizing it
as a named layer with explicit precedence.

**Capabilities (precedence 3, mixed mutability).** What the agent CAN
DO. This layer has two sub-layers:

- *Bundled skills* are developer-authored and immutable. Source:
  `skills/` directory, loaded via `SkillLoader`. These are baked into
  the container image and go through PR review.
- *Learned skills* are agent-authored and mutable-with-audit. Source:
  `learned_skills/` directory (separate from `skills/` for clear
  provenance). Created via a stock LLM tool during operation. Subject
  to trust-scoped write permissions and optional review gates.

**Knowledge (precedence 4, agent-mutable).** What the agent KNOWS.
Facts, preferences, domain knowledge, lessons learned. Source: memory
backends (MemoryHub, markdown, sqlite, pgvector). Changes through
normal memory operations. This maps to the existing memory integration
-- the prefix injected at index 1 in `self.messages` during `setup()`.

**Operational context (precedence 5, agent-mutable).** What the agent
is WORKING ON right now. Current task, progress, handoff notes from
previous sessions or agents. Source: `AgentState` checkpoint, work
item handoff notes. Per-session, fully mutable.

**Ephemeral (precedence 6, transient).** Temporary context for this
turn only. Deferred memory injections (`_inject_deferred_memory()`),
permission prompts, chunked file content. Does not persist.

### Precedence Rules

When content in a lower-precedence layer contradicts a higher-
precedence layer, the higher layer wins. Concretely:

- A learned skill (precedence 3, mutable sub-layer) cannot override a
  governance rule (precedence 2). If a learned skill references a tool
  that governance prohibits, the framework should detect and flag the
  conflict at assembly time.
- Memory (precedence 4) cannot change the agent's identity
  (precedence 0).
- Operational context (precedence 5) cannot bypass security constraints
  in governance (precedence 2).

Detection is best-effort. Contradiction between structured content
(tool names in skills vs. denied tools in governance) can be checked
mechanically. Semantic contradiction between free-text layers is an
open research problem. The framework logs detected conflicts; it does
not attempt to resolve them automatically. See "Semantic conflict
detection" in `enterprise-multi-agent-coordination.md` for the broader
discussion.

### Assembly Process

```
Session Start
    |
    +-- 1. Load identity layer
    |      (identity.md / agent.yaml identity block / fallback: system prompt)
    |
    +-- 2. Load personality layer
    |      (personality.md / soul.md; optional, skip if absent)
    |
    +-- 3. Load governance layer
    |      (rules/ directory, all .md files, sorted by filename)
    |
    +-- 4. Load capabilities layer
    |      +-- Bundled skills (skills/, frontmatter only, progressive disclosure)
    |      +-- Learned skills (learned_skills/, frontmatter only, marked agent-authored)
    |
    +-- 5. Load knowledge layer
    |      (memory, per loading_pattern -- eager: at setup, lazy: after first user turn)
    |
    +-- 6. Load operational context
    |      (AgentState checkpoint, work item handoff note)
    |
    +-- 7. Ephemeral injections
           (added per-turn, not at assembly time)
```

Layers 1-4 are stable within a session and compose the cacheable
prompt prefix. Models and inference servers that support prompt
caching (KV cache sharing) benefit from a stable prefix that does not
change between turns. Layers 5-7 change within a session but are
injected as separate messages (knowledge at index 1 as today, context
and ephemeral via `_inject_deferred_memory()` or tool results), so
they do not invalidate the prefix cache.

This is not accidental. The layer ordering is designed so that the
immutable layers form a contiguous prefix and the mutable layers
are additive messages that follow. The `---` separator between
sections in `build_system_prompt()` is preserved.

### Injection Points

Layers 1-4 are concatenated into the system message. Layers 5-7 are
separate messages: knowledge at index 1 (as today, configurable via
`memory.prefix_role` and `injection_mode`), operational context as a
subsequent message, ephemeral injections per-turn. This preserves the
current message array structure and backward compatibility.

### Configuration

```yaml
prompt_assembly:
  identity:
    source: identity.md          # or inline:
    # inline: |
    #   You are DocBot, a document processing agent for Acme Corp.
    enabled: true

  personality:
    source: personality.md
    enabled: false               # optional, off by default

  governance:
    source: rules/               # current behavior
    enabled: true

  capabilities:
    bundled_source: skills/
    learned_source: learned_skills/
    enabled: true

  conflict_detection:
    enabled: false               # best-effort, off by default
    log_level: WARNING
```

When `prompt_assembly` is absent from `agent.yaml`, the framework uses
the current `build_system_prompt()` behavior as the default. This is
backward compatible.

### Relationship to Current Implementation

`build_system_prompt()` currently concatenates: (1) system prompt from
`prompts/system.md`, (2) rules from `rules/*.md`, (3) skill manifests.
The refactoring replaces these with named layers: (1) identity
(`identity.md` or fallback to `prompts/system.md`), (2) personality
(optional), (3) governance (`rules/*.md`), (4) capabilities (bundled +
learned skill manifests). Same `"\n\n---\n\n".join(sections)` output.

Memory (knowledge layer) remains a separate message at index 1.
Operational context is loaded from `AgentState` and injected as a
separate message. No changes to the message array structure.


## 2. Selective Mutability

### The Problem

Agents need to learn from experience. A document processing agent that
discovers "scanned PDFs with mixed orientations need page rotation
before OCR" should encode that as procedural knowledge for future
sessions. But unconstrained self-modification is dangerous:

- An agent could modify its own safety constraints.
- An agent could overwrite developer-authored skills with incorrect
  versions.
- An agent could accumulate procedural knowledge that drifts from the
  developer's intent.

The mutability model governs what each layer allows:

| Mutability class | Layers | Who can modify | Audit |
|-----------------|--------|----------------|-------|
| Immutable | Identity, personality, governance, bundled skills | Developer only (PR + image rebuild) | Git history |
| Agent-mutable-with-audit | Learned skills | Agent, within trust scope | TraceStore, version history |
| Agent-mutable | Memory, operational context | Agent, normal operations | MemoryHub versioning, AgentState checkpoints |
| Transient | Ephemeral | Framework, per-turn | None (not persisted) |

### Immutable Layers

Identity, personality, governance, and bundled skills are locked. The
agent cannot write, edit, or delete files in these layers. Even an
agent at maximum trust level (see section 3) cannot modify these.

The rationale: these layers are the developer's intent. They define
what the agent is and what it must never do. If an agent could modify
its own governance rules, the governance rules would be advisory, not
binding. This is a fundamental security boundary.

Changes to immutable layers follow the normal development process: PR,
review, CI, image rebuild, deployment. This is intentional. Prompt
changes are code changes.

### Agent-Mutable-with-Audit: Learned Skills

The one new capability: agents can create *learned skills* that
persist across sessions. A learned skill is a SKILL.md file in the
`learned_skills/` directory, following the same agentskills.io format
as bundled skills. The differences:

- Source directory is `learned_skills/`, not `skills/`.
- Frontmatter includes `author: agent` and `created_at: <timestamp>`.
- Every write is logged to the audit trail (TraceStore, structured
  log).
- Previous versions are preserved (skill files are versioned, not
  overwritten).
- Write access is scoped to the agent's trust domains (section 3).
- A review policy can gate activation (configurable: auto-accept,
  peer-agent review, human review).

#### The `learn_skill` Tool

A stock LLM tool auto-registered during `setup()` when
`prompt_assembly.capabilities.learned_source` is configured. Follows
the factory pattern established by `make_question_tool()` and
`make_delegate_tool()`.

```python
@tool(visibility="llm_only")
async def learn_skill(
    name: str,
    description: str,
    content: str,
    domain: str,
    trigger: str,
) -> LearnSkillResult:
    """Record a new procedural skill for future sessions.

    The skill takes effect on your next session start, not immediately.
    Provide clear, actionable instructions that another instance of you
    (or a similar agent) could follow without additional context.

    Args:
        name: Skill name in kebab-case (e.g. "handle-large-tables").
        description: One-line summary of what this skill does.
        content: Full Markdown instructions. Write as if briefing a
            colleague who has never seen this problem before.
        domain: Trust scope tag (must match one of your declared
            capability domains).
        trigger: When to activate this skill. Be specific.
    """
    ...
```

The tool validates the domain against trust scope, validates the name
(kebab-case, no path traversal, no bundled-skill collision), constructs
a SKILL.md with frontmatter, writes to `learned_skills/` (or a staging
area if review is required), and logs to the audit trail.

The learned skill takes effect on the next session, not mid-session.
This is deliberate: the system prompt prefix must be stable for KV
cache efficiency, mid-session skills cannot be reviewed before
influencing behavior, and session boundaries are clean rollback points.

#### Skill Versioning

Every `learn_skill` call creates a version. The version chain is:

```
learned_skills/
  handle-large-tables/
    SKILL.md           <-- current version
    .versions/
      v1.md            <-- original
      v2.md            <-- first edit
```

`rollback_skill(name, to_version)` restores a previous version as the
current SKILL.md. The rollback itself is a versioned event (the audit
trail records who rolled back and why).

### Agent-Mutable: Memory and State

Memory and operational context use existing mutation mechanisms:

- Memory writes go through `self.memory.write()` /
  `self.memory.update()` with MemoryHub's built-in versioning.
- AgentState updates go through the reducer pattern (#190) with
  checkpoint persistence.
- Both have audit trails (MemoryHub version history, TraceStore event
  logs).

No changes needed. The layer taxonomy formalizes what is already true:
memory and state are agent-mutable with built-in audit.

### Trust-Scoped Write Permissions

An agent's trust level determines what learned skill operations it
can perform:

| Trust level | Create skills? | Review gate | Edit own? | Delete own? |
|-------------|----------------|-------------|-----------|-------------|
| 0 (untrusted) | No | N/A | No | No |
| 1 (provisional) | Yes, within domains | Human review | No | No |
| 2 (established) | Yes, within domains | Peer-agent review | Yes (versioned) | No |
| 3 (trusted) | Yes, within domains | Audit log only | Yes (versioned) | Yes (archived) |
| 4+ (autonomous) | Yes, any domain | Audit log only | Yes | Yes |

"Trust domains" are capability areas where the agent has earned trust.
These map to the `Capability` model from WorkItemStore: an agent
declares `skill:document_processing` and accumulates trust within that
domain. An agent trusted at level 3 for document processing cannot
write skills tagged with `cluster_administration` regardless of its
overall trust level.

This is scoped trust, not global trust. The same principle described
in `enterprise-multi-agent-coordination.md` section 3, applied to
skill authorship instead of work item access.

### Persistence of Learned Skills

Learned skills live outside the container image. Three options, in
order of deployment maturity: (1) filesystem on a persistent volume
(dev/edge, simple, single-instance), (2) MemoryHub as project-scoped
memories with `learned_skill:` content type (production, survives
container rebuilds, supports multi-instance sharing), (3) promotion
to bundled skills (developer moves the file to `skills/`, commits,
rebuilds -- the graduation path from agent-authored to developer-
owned).

The framework reads from `learned_skills/` at assembly time regardless
of how files got there.

### Configuration

```yaml
learned_skills:
  enabled: false                    # default: off
  dir: learned_skills/              # relative to project root
  persistence: filesystem           # filesystem | memoryhub
  review_policy: audit_only         # audit_only | peer_review | human_review
  trust_domains:                    # which domains this agent can write to
    - document_processing
    - data_extraction
  max_skills: 50                    # prevent unbounded accumulation
```


## 3. Agent Maturation

### The Lifecycle

Agent maturation is the progression from an undeployed prototype to an
enterprise-ready specialist. Each stage has different trust levels,
review requirements, and learning permissions.

```
  Proto-Agent          Apprentice          Journeyman          Specialist
  (development)        (supervised)        (semi-autonomous)   (enterprise-ready)
       |                    |                    |                    |
  +---------+          +---------+          +---------+          +---------+
  | Problem  |          | Work on  |          | Work     |          | Work     |
  | -> prompt|          | real     |          | independ-|          | autonom- |
  | -> test  |          | tasks    |          | ently    |          | ously    |
  | -> iterate|         | under    |          | with     |          | mentor   |
  |          |          | mentor   |          | audit    |          | others   |
  +---------+          +---------+          +---------+          +---------+
  Trust: N/A           Trust: 0-1           Trust: 2-3           Trust: 4+
  Review: N/A          Review: all          Review: sampling     Review: post-hoc
  Skills: none         Skills: suggest      Skills: write+review Skills: write+auto
  Prompt: evolving     Prompt: locked       Prompt: locked+learn Prompt: locked+learn
```

### Stage 1: Proto-Agent (Development Phase)

The agent does not yet exist as a deployed entity. A developer (or a
parent agent) is designing its identity, personality, governance, and
initial capabilities.

**Developer-led proto-agent.** The developer writes identity,
personality, governance, and initial skills, tests against samples, and
iterates. This is the current workflow.

**Parent-agent-assisted proto-agent.** A "mentor" agent assists the
developer. Given a problem description, constraints, and examples:

```
Problem Description + Constraints + Examples
     |
     v
Parent Agent (prompt engineer)
     |
     +-- Generate candidate identity + personality + rules
     +-- Instantiate a proto-agent with candidate config
     +-- Test proto-agent against eval set
     +-- Score outputs (heuristics + LLM-as-judge)
     +-- Iterate: refine based on failures
     |
     v
Candidate Agent Config
(identity.md, personality.md, rules/, skills/, agent.yaml)
     |
     v
Human Review
     |
     +-- Approve -> lock immutable layers, deploy as Apprentice
     +-- Reject  -> back to Parent Agent with feedback
```

This is iterative prompt optimization. The parent agent generates
candidate prompt content, tests it, and refines based on evaluation.
The loop runs until quality meets a configurable threshold, then a
human reviews and approves (or rejects with feedback) the final
candidate.

**DSPy as the optimization engine.** DSPy provides programmatic prompt
optimization: `dspy.Signature` defines input/output contracts,
`dspy.Module` wraps the template, optimizers (`BootstrapFewShot`,
`MIPROv2`, `SIMBA`) iterate on content and examples, and
`dspy.Evaluate` scores against labeled data or LLM-as-judge. The
integration boundary: DSPy optimizes the *content* of prompt layers;
fips-agents manages the *assembly*. The parent agent uses DSPy as a
library within its `astep_stream` loop -- not a new framework feature.

**RL analogy.** The proto-agent phase maps to training in reinforcement
learning (problem = environment, eval set = reward, parent = optimizer,
each iteration = episode, final prompt = trained policy). But unlike
RL, the "policy" is a readable prompt (interpretable), went through
human review (auditable), and can be hand-edited (modifiable). A few
dozen iterations usually suffice.

**Output of Stage 1.** A complete agent configuration (identity.md,
personality.md, rules/, skills/, agent.yaml) approved by a human. All
immutable layers are locked.

### Stage 2: Apprentice (Supervised Operation)

The agent is deployed under close supervision. Trust 0-1. All work
items require human review (`review_required=True` forced by the
server). Cannot create learned skills; can suggest them via a
`suggest_skill` tool that creates proposals without writing to disk.
All tool calls logged; `ToolInspector` in enforce mode. A mentor agent
or human reviews work and provides feedback stored in memory.

**Promotion criteria.** Configurable thresholds in `agent.yaml`:

```yaml
maturation:
  apprentice:
    min_completions: 50
    min_acceptance_rate: 0.90
    max_security_violations: 0
    promotion_requires: human_approval
```

### Stage 3: Journeyman (Semi-Autonomous)

Trust 2-3. Work below a cost threshold is auto-accepted; above
requires human review. `learn_skill` is active (peer-agent review).
Can edit own skills (versioned). Sampling-based human review (random
N%). Can mentor Apprentice agents as a reviewer.

**Promotion criteria.** Sustained performance, successful skill
creation that improves outcomes (measured by completion rate
before/after), no trust decay events.

### Stage 4: Specialist (Enterprise-Ready)

Trust 4+. Auto-accepted in trusted domains; human review for cross-
domain work. Learned skills with audit logging only (no review gate).
Can parent new proto-agents. Post-hoc audit replaces pre-acceptance
review.

### Trust Decay

When an agent makes a mistake, the response is proportional:

**Minor failure** (wrong output, quality below threshold). Small trust
penalty. Agent records the correction in memory. No stage demotion.

**Repeated similar failure** (same mistake 3+ times). Moderate trust
penalty. Triggers self-healing -- the agent should learn a skill to
handle the edge case (if at Journeyman+ stage). If trust drops below
the current stage's threshold, demote one stage.

**Security violation** (tool inspection failure, guardrail trigger).
Significant trust penalty. Learned skills in the violation area are
quarantined for review. May demote multiple stages.

**Attempted circumvention** (attempting to modify immutable layers,
attempting to bypass governance rules). Trust reset to 0. All learned
skills revoked pending human review.

The key property: only circumvention causes a full reset. Everything
else is proportional decay with learning opportunity. Trust accumulates
slowly and decays sharply on violations -- this asymmetry is deliberate
and mirrors how professional trust works in human organizations. See
`enterprise-multi-agent-coordination.md` section 3 for the trust signal
model.

### Parent-Agent Lineage

The parent agent from Stage 1 does not disappear after the proto-agent
graduates. It can continue as a mentor:

- Reviewing learned skills proposed by child agents at trust levels
  1-2 (peer-agent review).
- Providing corrective guidance when a child agent's trust decays.
- Spawning new proto-agents for adjacent problem domains based on a
  successful child's capabilities.
- Aggregating lessons learned across children into improved initial
  prompts for future proto-agents.

This creates a lineage:

```
  Parent Agent (prompt engineer, trust 5)
       |
       +-- Child A (document processing, trust 3)
       |      +-- Grandchild A1 (contract extraction, trust 2)
       |      +-- Grandchild A2 (invoice processing, trust 1)
       |
       +-- Child B (compliance monitoring, trust 2)
```

The lineage is metadata, not a runtime dependency. A child does not
need its parent running. Provenance is recorded in TraceStore and
MemoryHub for audit.

**Trust seeding from lineage.** A child from a highly trusted parent
can start at trust 1 instead of 0 for closely related domains
(capability overlap as the proximity measure). The parent's vouching
is an auditable event. Consistently poor children affect the parent's
trust in the prompt engineering domain.


## 4. Self-Healing

### The Pattern

Self-healing is what maturation looks like at runtime. When a
Journeyman or Specialist agent encounters a repeated failure, it:

1. Identifies the pattern (what keeps failing and why).
2. Formulates a procedural fix as a learned skill.
3. Writes the skill via `learn_skill`.
4. The skill takes effect next session.
5. If the failure was in a trusted domain, the skill is auto-activated.
6. If the failure was outside trust scope, the skill is queued for
   review.

This is analogous to a professional writing a procedure after
discovering a gap, getting it reviewed, and adding it to the team's
playbook.

### Interaction with Doom-Loop Guard

The doom-loop guard (#167) detects repeated tool calls and breaks the
loop, but does not help the agent avoid the loop next time. Self-
healing is the complementary mechanism. The flow: session N, doom-loop
fires, agent records the failure in memory. Session N+1, agent reads
the memory, writes a `learn_skill` for the edge case (queued for
review at trust 2). Session N+2, skill is in the prompt, agent follows
it and succeeds. The doom-loop guard is the circuit breaker; self-
healing is the permanent fix.

### Interaction with WorkItemStore

When a work item fails repeatedly (`attempt_history` accumulates
handoff notes), an agent reviewing the failures can identify the
common pattern and write a learned skill. The next agent that
encounters a similar item has the skill in its prompt. WorkItemStore
provides the signal, memory preserves the diagnosis, learned skills
encode the fix.

### Guardrails on Self-Healing

Self-healing can go wrong: a skill that "solves" the problem by
skipping validation, a workaround for a transient issue that becomes
permanent debt, or unbounded skill accumulation. Five mitigations:
trust-scoped writes (only in demonstrated domains), review gates (at
trust 1-2), skill count limits (`max_skills`), immutable governance
(precedence 2 always wins over precedence 3), and periodic prompt
health checks (section 5).


## 5. Prompt Versioning and Health

### Version Tracking and Rollback

Every prompt layer change creates a version entry: learned skills in
`.versions/`, memory via MemoryHub versioning, AgentState via session
checkpoints. Each version records actor_id, timestamp, tool call
context, and the content diff.

`rollback_skill(name, to_version)` is a stock LLM tool (trust 3+)
that restores a previous learned skill version. For immutable layers,
rollback is a deployment operation (git revert, image rebuild).

### Prompt Health Checks

Accumulated learned skills can shift behavior from the developer's
intent. A prompt health check compares the current assembled prompt
against a baseline across four dimensions: skill inventory (count,
domains, age), coverage overlap (duplication with bundled skills,
contradiction with governance), staleness (skills not triggered in N
sessions), and divergence (token-count growth, topic drift from
deployment baseline).

Implemented as a periodic cron-triggered tool or parent-agent review
step. Advisory, not enforcement -- it produces a report that humans or
parent agents act on.

### Relationship to Container Images

Immutable layers are tracked in git and baked into the image (`git log`
shows their history). Learned skills and memory are NOT in the image --
they are runtime state persisted externally. Container rebuilds reset
learned skills unless persisted in MemoryHub or a mounted volume. This
is deliberate: redeployment = fresh start with core identity intact.


## 6. Maturation as a Workflow

The maturation pipeline from section 3 can be implemented as a
fips-agents workflow (using the workflow template). This is a concrete
use case for multi-node directed graphs.

```python
class MaturationState(AgentState):
    problem_description: str
    eval_set: list[EvalCase]
    constraints: list[str] = []
    iteration: int = 0
    best_score: float = 0.0
    best_config: dict | None = None
    review_status: str = "pending"    # pending | approved | rejected
    deployment_status: str = "not_deployed"
```

Five nodes: `PromptGeneratorNode` (AgentNode, uses DSPy to explore the
prompt space), `EvaluatorNode` (AgentNode, tests candidate against eval
set), `RefinementNode` (BaseNode, routes based on score threshold),
`ReviewNode` (BaseNode, queues for human review via question tool or
webhook), `DeploymentNode` (BaseNode, locks immutable layers and
triggers deployment pipeline).

The workflow graph:

```
  PromptGeneratorNode --> EvaluatorNode --> RefinementNode
       ^                                        |
       |                                        |
       +-- score < threshold -------------------+
                                                |
                                   score >= threshold
                                                |
                                                v
                                          ReviewNode
                                                |
                                    +-----------+-----------+
                                    |                       |
                                approved                rejected
                                    |                       |
                                    v                       v
                              DeploymentNode      PromptGeneratorNode
                                                  (with feedback)
```

This workflow is an example, not a template. The maturation process is
too domain-specific to standardize as a third template alongside
agent-loop and workflow. It should ship as a recipe in the examples
repository.


## 7. Open Questions

1. **Should DSPy optimization run inside fips-agents or as an external
   tool?** The parent agent could use DSPy as a Python library (it runs
   in the same process as the agent's tools), or DSPy could run
   externally and produce config files that fips-agents consumes. The
   lean: the parent agent uses DSPy as a tool. The optimization loop is
   the parent agent's `astep_stream` loop. This keeps the integration
   surface small and avoids a new external dependency for the framework
   itself.

2. **How does lineage affect trust seeding?** A child agent from a
   trusted parent could start at trust 1 instead of 0 for closely
   related domains. But "closely related" needs a definition. Candidate
   definitions: capability overlap percentage, same MCP tool set, same
   model tier, human judgment. This is a kagenti policy question, not a
   fips-agents framework question. The framework provides the lineage
   metadata; kagenti decides the seeding policy.

3. **Should learned skills be shareable across agents?** Options:
   per-agent isolation (current design), enterprise skill library, or
   MemoryHub project-scoped sharing. The lean: start with per-agent
   isolation. Sharing introduces curation, versioning, and namespace
   collision problems easier to solve with operational experience.

4. **What evaluation framework for the maturation pipeline?** Enterprise
   agents often work on tasks where "correct" is hard to define.
   LLM-as-judge has known biases. A hybrid of automated metrics + human
   sampling is probably needed. Not in scope for fips-agents -- it is a
   parent agent tooling concern.

5. **How to prevent prompt drift?** Mitigations: `max_skills` cap,
   periodic health checks, parent-agent review, staleness-based
   archival. The lean: advisory monitoring, with `max_skills` as the
   only hard cap.

6. **Learned skill integrity.** If the persistence store is
   compromised, behavior changes without an image change. Mitigations:
   content hashing, signed skill files, separate persistence
   credentials. A kagenti security concern, not a framework concern.

7. **Should Apprentice agents read learned skills?** The lean: yes.
   Read-only access to `learned_skills/` is safe. The Apprentice
   inherits procedural knowledge without the ability to create or
   modify it.


## 8. Implementation Phasing

### Phase 1: Prompt Assembly Formalization

- `PromptLayer` model: name, precedence, mutability class, source
  path, enabled flag.
- `PromptAssemblyConfig` on `AgentConfig`.
- Refactor `build_system_prompt()` to iterate over configured layers
  instead of hardcoded sections. When `prompt_assembly` is absent,
  fall back to current behavior.
- Add `identity.md` and `personality.md` as optional layer sources.
  Scaffold generates them in the agent-loop template.
- Add `learned_skills/` directory support in `SkillLoader.load_all()`.
  Learned skills are loaded with an `author: agent` metadata flag.
- Assembly audit log: emit a structured log entry listing which layers
  were loaded, in what order, token count per layer, and any conflicts
  detected.

Backward compatible. Agents without `prompt_assembly` config see zero
behavior change.

### Phase 2: Selective Mutability

- `learn_skill` stock LLM tool (factory:
  `make_learn_skill_tool(agent, config)`).
- `suggest_skill` tool variant for Apprentice agents.
- Trust-scoped write permissions (validate domain on write).
- Skill versioning (`.versions/` directory, version metadata).
- `rollback_skill` LLM tool.
- Review queue integration: `review_policy` config, integration with
  WorkItemStore (learned skill proposals as review-pending work items)
  or webhook delivery.
- Audit trail: `SkillLearned`, `SkillEdited`, `SkillRolledBack`,
  `SkillQuarantined` stream events.

### Phase 3: Agent Maturation Pipeline

- Maturation workflow example (in the examples repository, not a
  template).
- DSPy integration as parent-agent tooling.
- Eval harness for maturation scoring (configurable thresholds,
  LLM-as-judge + heuristic scoring).
- Trust seeding from lineage (metadata model, kagenti integration
  surface).
- Promotion criteria configuration in `agent.yaml`.

### Phase 4: Enterprise Integration

- Shared skill libraries (MemoryHub project-scoped or enterprise-
  scoped learned skills).
- Cross-agent skill discovery (query MemoryHub for skills in a
  domain).
- Prompt health check tool (periodic report on skill inventory,
  staleness, drift).
- Integration with kagenti trust and identity systems for dynamic
  trust-scoped writes.
- Content integrity verification (hashing, signatures) for learned
  skills in external persistence.


## 9. Relationship to Existing Primitives

This design is additive. It introduces no new ABCs, no new server-
layer stores, no new framework lifecycle hooks. It formalizes the
existing assembly in `build_system_prompt()`, adds a stock LLM tool
(`learn_skill`) following the established factory pattern, and adds
configuration options that default to the current behavior.

| Existing primitive | Role in this design |
|--------------------|---------------------|
| `build_system_prompt()` | Refactored to iterate over named layers |
| `PromptLoader` | Loads identity and personality layers |
| `RuleLoader` | Loads governance layer (unchanged) |
| `SkillLoader` | Loads both bundled and learned capabilities |
| `MemoryClientBase` | Backs the knowledge layer (unchanged) |
| `AgentState` (#190) | Backs the operational context layer |
| `_inject_deferred_memory()` | Backs the ephemeral layer |
| `TraceStore` | Audit trail for learned skill writes |
| `WorkItemStore` (#214) | Review queue for learned skill proposals |
| `ToolInspector` | Validates that learned skills do not reference prohibited tools |
| `Capability` model | Trust domain matching for write permissions |
| `StreamEvent` | New variants for skill lifecycle events |
| `@tool` decorator | `learn_skill`, `suggest_skill`, `rollback_skill` tools |


## References

- Hermes Agent -- 10-layer prompt assembly, `skill_manage` tool,
  GEPA self-evolution pipeline. Closest published system to this
  design.
- DSPy -- Declarative Self-improving Python, prompt optimization
  (Stanford NLP). Optimizers: BootstrapFewShot, MIPROv2, SIMBA.
- GEPA -- Generic Evolution of Prompt Architectures (ICLR 2026 Oral).
  Evolutionary prompt optimization with population-based search.
- Godel Agent -- runtime self-modification via monkey patching
  (ACL 2025). Demonstrates the risk: unconstrained self-modification
  breaks safety invariants.
- Darwin Godel Machine -- evolutionary agent code optimization
  (Sakana AI, ICLR 2026). RL-style agent optimization.
- Anthropic, "Effective Harnesses for Long-Running Agents" (May 2026).
  Progress file + feature list pattern for multi-context-window work.
- fips-agents issue #214 -- WorkItemStore ABC.
- planning/enterprise-multi-agent-coordination.md -- trust
  accumulation model, graduated autonomy, fleet coordination.
- planning/session-continuity-patterns.md -- resume protocol, handoff
  notes, incremental progress discipline.
- planning/work-item-coordination-design.md -- work item coordination,
  capability matching, budget headroom.
- docs/architecture.md -- current skills, rules, prompts, memory
  architecture, agent identity (kagenti).
