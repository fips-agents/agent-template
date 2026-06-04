# MCP Gateway Authentication

Agents can authenticate to MCP servers behind an auth gateway (such as Kuadrant MCP Gateway with Keycloak) using JWT bearer tokens in HTTP headers. The framework handles the full lifecycle: initial token acquisition at startup, header injection on every MCP request, and automatic token refresh when the server returns 401/403.

## How it works

Three mechanisms work together to provide transparent, continuous authentication.

**`headers` on `McpServerConfig`** -- `agent.yaml` declares HTTP headers with `${VAR}` env var templates. At config load time, these are resolved against the current environment. The raw templates (pre-substitution) are preserved internally so they can be re-resolved later with updated env vars.

**`pre_mcp_connect` hook** -- fires before each MCP server connection. The hook script acquires a token and prints `KEY=VALUE` to stdout. The framework parses these lines into `os.environ`, then resolves the header templates against the updated environment, passing the resulting headers to the MCP transport.

**`mcp_auth_refresh` hook** -- fires automatically when an MCP tool call receives a 401 or 403 response. The hook acquires a fresh token, the framework updates env vars and re-resolves headers from the stored templates, closes the old MCP client, opens a new connection with fresh headers, and retries the failed tool call once.

## Auth lifecycle

```
Startup
-------
  pre_mcp_connect hook runs
    -> acquire-token.sh prints MCP_AUTH_TOKEN=eyJ...
    -> framework sets os.environ["MCP_AUTH_TOKEN"]
    -> header templates resolved: Authorization = "Bearer eyJ..."
    -> StreamableHttpTransport(headers={...})
    -> FastMCP Client connects
    -> tools discovered

Mid-session (token expires)
---------------------------
  LLM calls an MCP tool
    -> MCP server returns 401
    -> mcp_auth_refresh hook runs
    -> acquire-token.sh prints MCP_AUTH_TOKEN=eyK... (fresh token)
    -> framework updates os.environ["MCP_AUTH_TOKEN"]
    -> header templates re-resolved from stored ${MCP_AUTH_TOKEN}
    -> old MCP client closed
    -> new MCP client connected with fresh headers
    -> failed tool call retried
    -> success
```

## Configuration

The complete `agent.yaml` configuration:

```yaml
mcp_servers:
  - url: ${MCP_GATEWAY_URL}
    headers:
      Authorization: "Bearer ${MCP_AUTH_TOKEN}"

hooks:
  - event: pre_mcp_connect
    command: ./hooks/acquire-token.sh
    timeout: 10
  - event: mcp_auth_refresh
    command: ./hooks/acquire-token.sh
    timeout: 10
```

Both events use the same script -- this is intentional. The token acquisition logic is identical for initial connect and mid-session refresh. Keeping one script eliminates drift between the two paths.

## Hook script

`hooks/acquire-token.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

KEYCLOAK_URL="${KEYCLOAK_URL:-}"
CLIENT_ID="${MCP_CLIENT_ID:-}"
CLIENT_SECRET="${MCP_CLIENT_SECRET:-}"

# No Keycloak URL means auth is disabled (local dev). Exit cleanly.
[ -n "$KEYCLOAK_URL" ] || exit 0

TOKEN=$(curl -s -X POST "${KEYCLOAK_URL}/protocol/openid-connect/token" \
  -d "grant_type=client_credentials" \
  -d "client_id=${CLIENT_ID}" \
  -d "client_secret=${CLIENT_SECRET}" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

echo "MCP_AUTH_TOKEN=${TOKEN}"
```

Make it executable:

```bash
chmod +x hooks/acquire-token.sh
```

Key design choices in this script:

- `exit 0` on missing `KEYCLOAK_URL` allows the agent to start without auth during local development. The hook succeeds silently and the header template resolves to `Bearer ` (empty token), which is fine when the MCP server has no auth.
- The token is printed as `KEY=VALUE` on stdout. The framework parses each line, splitting on the first `=`, and sets the result in `os.environ`.
- `python3` is used for JSON parsing instead of `jq` because `jq` is not available in Red Hat UBI containers by default.
- Errors from `curl` or the Keycloak endpoint propagate naturally -- `set -euo pipefail` ensures the hook exits non-zero, which the framework logs as a warning.

## Header template resolution

The `${MCP_AUTH_TOKEN}` syntax in headers is not a one-time substitution. Here is how it works internally:

1. When `agent.yaml` is loaded, `_stash_header_templates()` parses the raw YAML a second time *without* env-var substitution and stores the original template strings (e.g., `"Bearer ${MCP_AUTH_TOKEN}"`) on each `McpServerConfig` as `_header_templates`.

2. At connection time, the templates are resolved against the current `os.environ` to produce the actual header values.

3. When `mcp_auth_refresh` fires, the hook updates `os.environ` with the fresh token. The framework then calls `substitute_env_vars()` on the stored `_header_templates` to produce new header values with the fresh token.

This is why the header value *must* use `${VAR}` syntax. A hardcoded token value would work for the initial connection but could never be refreshed.

## OpenShift deployment

Add the MCP gateway and Keycloak endpoint configuration to `chart/values.yaml`:

```yaml
config:
  MCP_GATEWAY_URL: "https://mcp-gateway.apps.cluster.example.com/mcp"
  KEYCLOAK_URL: "https://keycloak.apps.cluster.example.com/realms/mcp"
  MCP_CLIENT_ID: "my-agent"

env:
  - name: MCP_CLIENT_SECRET
    valueFrom:
      secretKeyRef:
        name: mcp-agent-credentials
        key: client-secret
```

Create the Secret containing the client credentials:

```bash
oc create secret generic mcp-agent-credentials \
  --from-literal=client-secret='<your-client-secret>' \
  --context=$CTX -n $NS
```

The `config:` values are injected as environment variables via the chart's ConfigMap. The client secret is mounted from a Kubernetes Secret so it never appears in plain text in version control or ConfigMap data.

## Scaffold shortcut

Running `/add-hook mcp-auth` generates all of the above automatically:

- `hooks/acquire-token.sh` -- the token acquisition script
- `hooks/acquire-token.yaml` -- `pre_mcp_connect` binding
- `hooks/refresh-token.yaml` -- `mcp_auth_refresh` binding
- `chart/values.yaml` env var placeholders for `KEYCLOAK_URL`, `MCP_CLIENT_ID`, `MCP_CLIENT_SECRET`, and `MCP_GATEWAY_URL`

It also updates the `mcp_servers` entry in `agent.yaml` with the `headers` block.

## Troubleshooting

### Agent starts but MCP tools return 401

Verify the hook produces a valid token by running it manually:

```bash
KEYCLOAK_URL=https://keycloak.example.com/realms/mcp \
MCP_CLIENT_ID=my-agent \
MCP_CLIENT_SECRET=secret123 \
./hooks/acquire-token.sh
```

Expected output: `MCP_AUTH_TOKEN=eyJ...` (a JWT). If the output is empty or the command fails, the `curl` call to Keycloak is not succeeding -- check the URL, client ID, and secret.

Then verify `agent.yaml` has the headers block:

```yaml
mcp_servers:
  - url: ${MCP_GATEWAY_URL}
    headers:
      Authorization: "Bearer ${MCP_AUTH_TOKEN}"
```

Without the `headers` block, the token is acquired but never sent.

### Token refresh fires but reconnect fails

Check agent logs for `"Failed to reconnect MCP server"`. This means the new token was acquired successfully but the MCP connection failed for another reason (network, DNS, TLS certificate).

To confirm the token itself is valid, add a debug line to the hook script. Stderr goes to the agent logs without interfering with the `KEY=VALUE` parsing on stdout:

```bash
echo "DEBUG: token=${TOKEN:0:20}..." >&2
```

### Config validation error: extra fields not permitted

`McpServerConfig` uses `extra="forbid"` and rejects unknown fields. Common mistakes:

- `transport: streamable-http` -- remove it; `url` implies streamable-http transport
- `name: my-server` -- not a valid field on `McpServerConfig`

The valid fields for HTTP transport are `url` and `headers` only.

### Token expires faster than expected

The default Keycloak access token lifespan is 5 minutes. For long-running agents, you have two options:

1. Increase the token lifespan in the Keycloak realm settings (Clients > your client > Advanced > Access Token Lifespan).
2. Accept that `mcp_auth_refresh` will fire periodically. The hook is designed for this -- it acquires a fresh token, reconnects transparently, and retries the failed call. The LLM never sees the 401.

### Multiple MCP servers with different auth

Each MCP server entry can have its own `headers` with different env var names. Use distinct variable names and, if needed, separate hook scripts:

```yaml
mcp_servers:
  - url: ${MCP_SEARCH_URL}
    headers:
      Authorization: "Bearer ${MCP_AUTH_TOKEN_SEARCH}"
  - url: ${MCP_DB_URL}
    headers:
      Authorization: "Bearer ${MCP_AUTH_TOKEN_DB}"
```

The `pre_mcp_connect` hook receives `MCP_SERVER_URL` in its environment, so a single script can branch on which server is connecting:

```bash
case "$MCP_SERVER_URL" in
  *search*) echo "MCP_AUTH_TOKEN_SEARCH=${TOKEN}" ;;
  *db*)     echo "MCP_AUTH_TOKEN_DB=${TOKEN}" ;;
esac
```

Alternatively, use separate hook scripts with different `matcher` patterns (though `pre_mcp_connect` fires for all servers by default -- the `MCP_SERVER_URL` env var is the dispatch mechanism).

## Further reading

- [docs/hooks.md](hooks.md) -- lifecycle hook reference (events, fields, env vars, debugging)
- [docs/architecture.md](architecture.md) -- BaseAgent internals, `McpServerConfig`, MCP integration
- [Kuadrant MCP Gateway](https://github.com/Kuadrant/mcp-gateway) -- upstream gateway documentation
