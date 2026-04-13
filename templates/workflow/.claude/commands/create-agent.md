# Create Workflow

Scaffold the workflow implementation from an approved AGENT_PLAN.md. Generates the state model, node classes, graph definition, tools, prompts, skills, rules, and configuration.

**Prerequisites: AGENT_PLAN.md must exist at the project root and be developer-approved.**

## Process

### Step 1: Read and Validate the Plan

Read `AGENT_PLAN.md`. Confirm it has: Purpose, State Model, Nodes, Graph, Tools, Prompts, Skills, Rules, Configuration, Eval Cases. If anything is missing, ask whether to proceed with defaults or go back to `/plan-agent`.

### Step 2: Generate the Workflow

Create `src/agent.py` with:

- A `WorkflowState` subclass matching the state model from the plan
- A `BaseNode` subclass for each routing/transformation node
- An `AgentNode` subclass for each LLM-enabled node
- A `build_graph()` function wiring nodes with edges from the plan
- A `main()` entry point with `WorkflowRunner`

Use `@node()` decorator on all node classes. Keep each node's `process()` method focused and under 50 lines.

### Step 3: Generate Tools

Same as agent-loop: one file per tool in `tools/`, `@tool` decorator with visibility, proper docstrings and type hints.

### Step 4: Generate Prompts

Same as agent-loop: Markdown with YAML frontmatter in `prompts/`. System prompt is required.

### Step 5: Generate Skills, Rules

Same as agent-loop.

### Step 6: Update Configuration

Update `agent.yaml` and `pyproject.toml` as needed.

### Step 7: Generate AGENTS.md

Populate `AGENTS.md` with workflow-specific content: node list, graph topology, state model fields, tools.

### Step 8: Verify

1. Run `python -c "import src.agent"` to verify imports.
2. Run `make test` to execute tests.
3. Run `make lint` if ruff is available.

Fix any failures before reporting completion. Tell the developer to run `/exercise-agent` next.
