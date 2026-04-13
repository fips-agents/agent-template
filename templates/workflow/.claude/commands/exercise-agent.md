# Exercise Workflow

Test workflow behavior through structured scenarios. Validates that nodes process state correctly, conditional edges route as expected, tool calls work within AgentNodes, and error edges handle failures gracefully.

**Prerequisites: The workflow must be implemented (run `/create-agent` first).**

## Process

### Step 1: Load the Workflow Design

Read `src/agent.py`, `prompts/`, `tools/`, `rules/`, `skills/`, and `agent.yaml`. Build a mental model of: the state model, node graph topology, which nodes are BaseNode vs AgentNode, and what tools are available.

### Step 2: Define Test Scenarios

Create scenarios covering:

**Happy path** (at least 3): Inputs that exercise each branch of the graph. Ensure every conditional edge path is tested at least once.

**Edge cases** (at least 2): Boundary inputs that test state validation, minimal/maximal state, or unusual routing.

**Failure scenarios** (at least 2): Inputs that should trigger error edges, node failures, or graceful degradation.

For each scenario, document: input state, expected node traversal order, expected final state fields.

### Step 3: Run Scenarios

For each scenario, trace execution through the graph:

1. Create initial state
2. Step through nodes in graph order
3. Verify state transformations at each node
4. Confirm correct edge routing (especially conditional edges)
5. Validate final state

Use live mode (real LLM) or dry-run mode (structural analysis) as appropriate.

### Step 4: Write Eval Cases

Write scenarios to `evals/evals.yaml`. Include assertions on: state field values, node traversal order, tool calls made by AgentNodes.

### Step 5: Report

Present passed/failed scenarios, observations about routing logic, and suggestions for improvement. Ask the developer whether to fix issues or proceed to `/deploy-agent`.
