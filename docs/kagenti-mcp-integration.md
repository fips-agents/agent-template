# Kagenti MCP Integration

How BaseAgent interacts with the Kagenti MCP Gateway, what works, and
what developers need to know.

## Architecture

Kagenti's MCP Gateway is a three-layer stack:

```
Client (BaseAgent)
  │
  ▼
OpenShift Route (TLS termination)
  │
  ▼
Istio Ingress Gateway (port 8080)
  │
  ▼ (ext_proc)
MCP Gateway Broker/Router
  │
  ├── handles tools/list (aggregates from all registered upstreams)
  ├── handles initialize (JWT session management)
  └── does NOT forward tools/call (as of v0.1.2)
```

The broker is an **Envoy external processor** (ext_proc), not a simple
reverse proxy. Every request is intercepted by the broker, which:

1. Rewrites the `:authority` header to the internal hostname
   (`mcp.127-0-0-1.sslip.io` by default)
2. Sets routing headers (`x-mcp-method`, `x-mcp-servername`)
3. Clears the Envoy route cache so re-routing takes effect

The upstream MCP servers (e.g., `weather-tool-mcp`) have HTTPRoute
resources that Istio CAN route to, but only when the broker's ext_proc
response tells Envoy to do so.

## What Works Today

**Tool discovery** works end-to-end from outside the cluster:

```python
from fastmcp import Client

url = "https://mcp-gateway-gateway-system.apps.<cluster>/mcp"
async with Client(url) as client:
    tools = await client.list_tools()
    # Returns tools aggregated from all registered MCP servers
```

BaseAgent's `connect_mcp()` successfully discovers and registers tools
from the Kagenti gateway. The tools appear in the registry with
`llm_only` visibility, exactly like any other MCP server.

## What Does NOT Work

**Tool execution** (`tools/call`) fails:

```
ToolError: Kagenti MCP Broker doesn't forward tool calls
```

The broker aggregates tool listings but does not proxy the actual calls
to upstream servers. This is a current limitation of the Kagenti MCP
broker (v0.1.2, tested 2026-04-17).

## Why Not Just Fix the Routing?

The broker's ext_proc architecture means you cannot bypass it by
changing hostnames or adding path-based routing. We tested this
(Option C from the investigation):

1. Updated Gateway listener hostname to match the OpenShift route
2. Updated HTTPRoute hostnames to match
3. Result: **broke everything** — the broker hardcodes the authority
   rewrite to `mcp.127-0-0-1.sslip.io`, and the virtual host must
   match for re-routing to work

The `mcp.127-0-0-1.sslip.io` hostname is not a misconfiguration — it's
what the broker expects internally.

## How to Use Kagenti MCP Today

### Option A: Agent Runs Inside the Cluster (Recommended)

Deploy your agent as a pod. Connect to upstream MCP servers directly
via their ClusterIP services:

```yaml
# agent.yaml
mcp_servers:
  - url: http://weather-tool-mcp.team1.svc.cluster.local:8000/mcp
```

Use the broker for discovery only (e.g., a startup probe that lists
available tools), then connect to each upstream directly.

### Option B: Expose Upstream MCP Servers via Routes

Create OpenShift Routes for each MCP server you need:

```bash
oc expose svc/weather-tool-mcp -n team1 --context=mcp-rhoai
```

Then connect from outside the cluster directly to each server,
bypassing the Kagenti gateway entirely:

```yaml
mcp_servers:
  - url: https://weather-tool-mcp-team1.apps.<cluster>/mcp
```

This works but loses the gateway's aggregation and (future)
authorization features.

### Option C: Wait for Broker Forwarding

The Kagenti roadmap includes `tools/call` forwarding through the
broker. When this lands, agents can use the single gateway URL for
both discovery and execution:

```yaml
mcp_servers:
  - url: https://mcp-gateway-gateway-system.apps.<cluster>/mcp
```

The test harness in `tests/integration/mcp/test_mcp_kagenti.py` has
`xfail(strict=True)` tests for tool dispatch. When broker forwarding
ships, these tests will start failing (indicating the xfail should be
removed), which serves as an automatic alert.

## Registering MCP Servers with Kagenti

Deploy the MCP server, create a Service, then register via CRDs:

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: my-tool-mcp
  namespace: my-namespace
  labels:
    kagenti.io/type: tool
spec:
  hostnames:
    - mcp.127-0-0-1.sslip.io   # Required — must match broker's expected hostname
  parentRefs:
    - kind: Gateway
      name: mcp-gateway
      namespace: gateway-system
  rules:
    - matches:
        - path:
            type: PathPrefix
            value: /my-tool
      backendRefs:
        - name: my-tool-mcp
          port: 8000
---
apiVersion: mcp.kagenti.com/v1alpha1
kind: MCPServerRegistration
metadata:
  name: my-tool
  namespace: my-namespace
spec:
  path: /mcp
  targetRef:
    group: gateway.networking.k8s.io
    kind: HTTPRoute
    name: my-tool-mcp
    namespace: my-namespace
```

The Kagenti controller detects the registration, connects to the
upstream server, discovers tools, and adds them to the broker's
aggregated listing.

## Cluster State Reference

Tested against Kagenti v0.1.2 on cluster n7pd5:

- Gateway: `mcp-gateway` in `gateway-system`
- Broker: `mcp-gateway-broker-router` in `mcp-system`
- External route: `mcp-gateway-gateway-system.apps.cluster-n7pd5.n7pd5.sandbox5167.opentlc.com/mcp`
- CRDs: `mcpserverregistrations`, `mcpgatewayextensions`, `mcpvirtualservers`
- Auth: disabled in this deployment (Keycloak available but not enforced)
