# Stakeholders

## Primary: Agent Developers

The primary users are developers building AI agents on OpenShift AI using the fips-agents ecosystem. These range from experienced Python developers who have built agents before (and are tired of the boilerplate) to developers new to agent architectures who need structured guidance.

What they need is a scaffold that produces a working agent quickly and stays out of their way as they iterate on the parts that matter: prompts, tools, model selection, and evals. A typical agent subclass should be 20-30 lines. The slash commands in `.claude/` should guide them through the design-build-test-deploy cycle without requiring deep knowledge of BaseAgent internals.

What they do not want is another framework to learn. BaseAgent is an implementation detail that handles common concerns; the developer's mental model should be "I write my step logic, define my tools, and craft my prompts."

## Secondary: Platform Teams

Teams managing OpenShift AI infrastructure -- deploying and operating vLLM, LlamaStack, PGVector, and other services via rh-ai-quickstart/ai-architecture-charts -- benefit from agents having a standardized deployment pattern. When every agent ships as a Helm chart with the same structure (Deployment, Service, ConfigMap, optional Route), platform teams can manage, monitor, and scale agents uniformly without understanding each agent's internal logic.

Standard configuration via `agent.yaml` with environment variable substitution means platform teams control environment-specific settings (endpoints, resource limits, replicas) through familiar OpenShift mechanisms (ConfigMaps, Secrets, values overrides) without modifying agent source code.

## Affected: fips-agents CLI Maintainers

This template becomes part of the fips-agents CLI's scaffolding. The CLI clones this repository when a developer runs `fips-agents create agent`. This means the template's directory structure, naming conventions, and metadata must align with what the CLI expects. Changes to the template structure may require corresponding CLI updates.

## Affected: rh-ai-quickstart Maintainers

We document rh-ai-quickstart/ai-architecture-charts as the expected infrastructure layer and link to it from our documentation. Our Helm chart must complement their deployments -- consuming the same service endpoints, following the same labeling conventions, and not conflicting with their resource names or namespace assumptions. Changes to their chart structure may require documentation updates on our side.

## Future: Multi-Agent System Operators

Once agents are deploying reliably and MemoryHub enables shared memory across agents, operators managing multi-agent systems become a stakeholder. They need visibility into agent behavior (which agent called which tool, what memories were shared, what decisions were made) and control mechanisms (disabling a misbehaving agent, revoking tool access). This stakeholder group informs the Tool Hub side quest and the forensic traceability requirements, but they are not a primary audience for the initial template.
