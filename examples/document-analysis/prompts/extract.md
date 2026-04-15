---
name: extract
description: Extract structured specifications from a technical document
variables:
  - name: document
    required: true
    description: The technical document to analyze
---

Analyze the following technical document and extract the key specifications
in a structured format. Include:

- API endpoints (method, path, parameters)
- Data schemas and types
- Requirements and constraints
- Version information

Format your response as clear, organized Markdown.

## Document

{document}
