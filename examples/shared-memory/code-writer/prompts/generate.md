---
name: generate
description: Code generation prompt for the AgentNode
variables:
  - name: request
    required: true
  - name: context
    required: false
    default: "No prior context available."
---

Write Python code to solve this request:

{request}

Context from previous sessions:
{context}

Requirements:
- Use only standard library modules from the sandbox allowlist
- Include docstrings on all functions
- Add a test/demo section at the bottom that prints results
- Output ONLY the Python code, no explanations

If the context mentions specific conventions (e.g., use Decimal for currency), follow them.
