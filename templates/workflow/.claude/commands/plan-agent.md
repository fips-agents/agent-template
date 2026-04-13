# Plan Workflow

Design a workflow before writing any code. This command produces an AGENT_PLAN.md file that captures the workflow's nodes, edges, state model, tools, and prompts. The plan is reviewed and approved before any implementation begins.

**This is planning only. Do not generate code, create files outside of AGENT_PLAN.md, or modify any source files.**

## Process

### Step 1: Understand the Workflow's Purpose

Start by asking the developer what this workflow should do. Understand:

- What problem does this workflow solve?
- What is the input and what is the desired output?
- What processing steps are needed between input and output?
- Which steps need LLM capabilities and which are pure logic?

### Step 2: Design the State Model

Identify the typed state that flows through the workflow:

- What fields does the state need?
- What types are they? (str, int, list, etc.)
- Which fields are populated by which nodes?
- Keep it minimal -- data only, no execution metadata.

### Step 3: Design Nodes

For each processing step, determine:

- **Name** and what it does
- **Type**: `BaseNode` (routing, transformation, gating -- no LLM) or `AgentNode` (needs LLM, tools, prompts)
- **Input**: which state fields it reads
- **Output**: which state fields it writes

### Step 4: Design Edges

Define the graph topology:

- **Entry point**: which node runs first
- **Fixed edges**: A always leads to B
- **Conditional edges**: A leads to B or C depending on state
- **Error edges**: where to route on failure (optional)
- **Terminal nodes**: which nodes lead to END

### Step 5: Identify Tools

For each AgentNode that needs tools, follow the same tool identification process as agent-loop: name, visibility (`agent_only`, `llm_only`, `both`), source (local, MCP, existing), parameters.

### Step 6: Design Prompts, Skills, Rules

Same as agent-loop. System prompt is shared across AgentNodes by default. Additional per-node prompts as needed.

### Step 7: Define Eval Cases

How do we know the workflow works? Define:

- 3-5 cases covering the happy path through different branches
- 2-3 cases covering edge cases and failure modes
- For each: input state, expected node traversal path, expected output state

### Step 8: Write AGENT_PLAN.md

Compile into `AGENT_PLAN.md` with sections: Purpose, State Model, Nodes, Graph (edges), Tools, Prompts, Skills, Rules, Memory, Configuration, Eval Cases.

Present for developer approval. When approved, tell the developer to run `/create-agent`.
