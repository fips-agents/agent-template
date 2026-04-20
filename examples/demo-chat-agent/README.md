# Demo Chat Agent — fips-agents three-component reference

This example shows the **default pattern** for deploying a chat agent
with a web UI on OpenShift using fips-agents. Three components, each
scaffolded from its own template, each with its own Helm chart:

```
Browser ──► demo-chat-ui (Go) ──► demo-chat-gateway (Go) ──► demo-chat-agent (Python)
            :3000                  :8080                       :8080
            embedded static        OpenAI /v1/chat/completions  BaseAgent + MemoryHub
            no build step          SSE relay + sync             stateless
```

The contract between each layer is **OpenAI `/v1/chat/completions`**. This
means any OpenAI-compatible client (LibreChat, eval harnesses, other
agents) can talk to the agent without bespoke integration.

## Why this pattern

- **Don't reinvent the UI.** `ui-template` is a single Go binary with
  embedded static files. It runs against any OpenAI-compatible API.
- **Don't reinvent the gateway.** `gateway-template` handles SSE
  streaming lifecycle (heartbeats, flush, `[DONE]` termination) and
  publishes an `/.well-known/agent.json` card. Go stdlib only — no
  middleware framework.
- **Agent stays stateless.** The client sends the full `messages` array
  each turn. No server-side session storage. The UI tracks history.
- **One URL for the audience.** Only the UI gets a public Route. The
  gateway and agent stay ClusterIP.

## This deployment

Live on the workshop cluster (`n7pd5`, namespace `demo-chat-agent`):

| Component            | Route                                                                                       |
|----------------------|---------------------------------------------------------------------------------------------|
| UI (public entry)    | `https://demo-chat-ui-demo-chat-agent.apps.cluster-n7pd5.n7pd5.sandbox5167.opentlc.com`      |
| Gateway (debug)      | `https://demo-chat-gateway-demo-chat-agent.apps.cluster-n7pd5.n7pd5.sandbox5167.opentlc.com` |
| Agent (debug)        | `https://demo-chat-agent-demo-chat-agent.apps.cluster-n7pd5.n7pd5.sandbox5167.opentlc.com`   |

Service DNS inside the namespace:
- Agent: `http://demo-chat-agent:8080`
- Gateway: `http://demo-chat-gateway:8080`
- UI: `http://demo-chat-ui:3000`

## Reproducing from scratch

### 1. Scaffold all three projects

```bash
cd ~/Developer/AGENTS

fips-agents create agent   demo-chat-agent    --local --yes --no-git
fips-agents create gateway demo-chat-gateway  --local --yes --no-git
fips-agents create ui      demo-chat-ui       --local --yes --no-git
```

### 2. Customize the agent

Edit the agent to expose OpenAI `/v1/chat/completions`. See
`src/server.py` and `src/agent.py` in this directory for a reference
implementation. Key points:

- FastAPI `POST /v1/chat/completions` handling both sync JSON and SSE
- No session storage — take `messages` from the request, run the agent,
  return one assistant message
- Wrap the agent's final response in the OpenAI schema (see
  `_sync_response` and `_stream_chunk` helpers)

MemoryHub integration via the SDK path: `.memoryhub.yaml` plus
`self.memory.search()` / `self.memory.write()` calls in `step()`. The
`MEMORYHUB_API_KEY` Secret is projected into the pod env.

### 3. Build each component in-cluster

```bash
NS=demo-chat-agent
oc new-project $NS

# Agent — BuildConfig uses the Containerfile in the repo
oc new-build --binary --strategy=docker --name=demo-chat-agent -n $NS
cd demo-chat-agent && oc start-build demo-chat-agent --from-dir=. --follow -n $NS

# Gateway — template ships a BuildConfig at build/buildconfig.yaml
cd ../demo-chat-gateway
sed 's/PLACEHOLDER/demo-chat-gateway/g' build/buildconfig.yaml | oc apply -n $NS -f -
oc start-build demo-chat-gateway --from-dir=. --follow -n $NS

# UI — no buildconfig in template, use oc new-build
cd ../demo-chat-ui
oc new-build --binary --strategy=docker --name=demo-chat-ui -n $NS
oc patch bc/demo-chat-ui -n $NS -p '{"spec":{"strategy":{"dockerStrategy":{"dockerfilePath":"Containerfile"}}}}'
oc start-build demo-chat-ui --from-dir=. --follow -n $NS
```

### 4. Create the MemoryHub Secret

```bash
oc create secret generic memoryhub-api-key \
  --from-literal=api-key="$MEMORYHUB_API_KEY" -n $NS
```

### 5. Helm install the three components

```bash
REG=image-registry.openshift-image-registry.svc:5000/$NS

cd demo-chat-agent
helm upgrade --install demo-chat-agent chart/ -n $NS \
  --set image.repository=$REG/demo-chat-agent --wait

cd ../demo-chat-gateway
helm upgrade --install demo-chat-gateway chart/ -n $NS \
  --set image.repository=$REG/demo-chat-gateway \
  --set config.BACKEND_URL=http://demo-chat-agent:8080 --wait

cd ../demo-chat-ui
helm upgrade --install demo-chat-ui chart/ -n $NS \
  --set image.repository=$REG/demo-chat-ui \
  --set config.API_URL=http://demo-chat-gateway:8080 --wait
```

### 6. Bump route timeouts past HAProxy's 30s default

LLM first-token latency can exceed 30s on a cold model. Set per-route:

```bash
for r in demo-chat-agent demo-chat-gateway demo-chat-ui; do
  oc annotate route $r -n $NS haproxy.router.openshift.io/timeout=120s --overwrite
done
```

## Known scaffolding gaps (as of fips-agents-cli at time of writing)

The `create gateway` and `create ui` scaffolders miss some sentinel
replacements. After scaffolding, run:

```bash
# Go imports
cd demo-chat-gateway && find . -name "*.go" -not -path "./vendor/*" \
  -exec sed -i '' 's|github.com/fips-agents/gateway-template|github.com/redhat-ai-americas/demo-chat-gateway|g' {} +

cd demo-chat-ui && find . -name "*.go" -not -path "./vendor/*" \
  -exec sed -i '' 's|github.com/fips-agents/ui-template|github.com/redhat-ai-americas/demo-chat-ui|g' {} +

# Helm chart helper references
cd demo-chat-gateway/chart/templates && sed -i '' 's|gateway-template\.|demo-chat-gateway.|g' *.yaml
cd demo-chat-ui/chart/templates      && sed -i '' 's|ui-template\.|demo-chat-ui.|g' *.yaml
```

These should be upstreamed into the CLI's sentinel list.

## What this reference directory contains

Only the **agent** code lives here (`src/`, `tools/`, `prompts/`,
`agent.yaml`, `chart/`, `Dockerfile`). The gateway and UI projects live
as siblings at `~/Developer/AGENTS/demo-chat-gateway/` and
`~/Developer/AGENTS/demo-chat-ui/` — just like a real customer would
have them. This directory documents the pattern; the working code for
the other two layers lives in its own project, as intended.
