# Lifecycle Hooks

Lifecycle hooks are shell commands that fire at specific points during your agent's execution. They let you integrate external tools, enforce policies, or customize behavior without modifying the agent code.

## What are hooks?

Hooks are configured in `agent.yaml` or auto-discovered from `hooks/*.yaml` files. When an event fires, the framework runs your command as a subprocess, captures stdout, and makes decisions based on the exit code.

## Built-in events

| Event | When | stdout behavior | Special |
|-------|------|-----------------|---------|
| `setup_complete` | After `setup()` finishes | Injected as context message | — |
| `shutdown` | Before `shutdown()` cleanup | Logged only | — |
| `pre_tool_use` | Before tool execution | Logged only | Non-zero exit blocks the tool call |
| `post_tool_use` | After tool execution | Logged only | — |
| `pre_mcp_connect` | Before each MCP server connection | Logged only | Token acquisition, pre-flight checks |
| `post_mcp_connect` | After MCP server connected + tools discovered | Logged only | Audit, validate expected tools |
| `mcp_auth_refresh` | When an MCP tool call fails with 401/403 | Env vars updated from stdout `KEY=VALUE` lines | Token refresh, transparent retry |

## Environment variables

Every hook receives:

- `AGENT_NAME` — agent's configured name
- `AGENT_PROJECT_DIR` — resolved base directory
- `HOOK_EVENT` — event name (e.g., `setup_complete`)

Tool events (`pre_tool_use`, `post_tool_use`) add:

- `TOOL_NAME` — name of the tool being called
- `TOOL_ARGS` — JSON-encoded tool arguments

Post-tool events also receive:

- `TOOL_RESULT` — tool output (truncated to 4096 chars)

MCP events (`pre_mcp_connect`, `post_mcp_connect`) add:

- `MCP_SERVER_URL` — URL or label of the MCP server

Post-connect events also receive:

- `MCP_TOOLS_COUNT` — number of tools discovered
- `MCP_PROMPTS_COUNT` — number of prompts discovered
- `MCP_RESOURCES_COUNT` — number of resources discovered

Auth refresh events also receive:

- `MCP_SERVER_URL` — URL or label of the server that returned 401/403

## Configuration

### Method 1: agent.yaml

```yaml
hooks:
  - event: setup_complete
    command: ./hooks/fetch-memory.sh
    timeout: 10
  - event: pre_tool_use
    command: ./hooks/check-safety.sh
    timeout: 3
    matcher: "execute_*"
```

### Method 2: Auto-discovery

Create `.yaml` or `.yml` files in the `hooks/` directory. Each file declares one hook.

```yaml
# hooks/setup-memory.yaml
event: setup_complete
command: ./hooks/fetch-memory.sh
timeout: 10
name: memory-loader
```

Hooks configured in `agent.yaml` fire first, then file-discovered hooks in alphabetical order.

## Hook fields

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `event` | yes | — | Lifecycle event name |
| `command` | yes | — | Shell command to run |
| `timeout` | no | 10.0 | Max seconds before SIGTERM |
| `matcher` | no | null | fnmatch pattern for tool name (tool events only) |
| `name` | no | filename stem | Human-readable label for logs |

The `matcher` field filters tool events. For example, `matcher: "db_*"` only fires for tools starting with `db_`.

## Example: MemoryHub pre-loading

Load project memories at session start and inject them as agent context.

**hooks/setup-memory.yaml:**

```yaml
event: setup_complete
command: ./hooks/fetch-memory.sh
timeout: 10
```

**hooks/fetch-memory.sh:**

```bash
#!/usr/bin/env bash
set -euo pipefail

API_KEY_FILE="$HOME/.config/memoryhub/api-key"
[ -f "$API_KEY_FILE" ] || exit 0

export MEMORYHUB_API_KEY=$(tr -d '\n' < "$API_KEY_FILE")

memoryhub search \
  "project context architecture preferences decisions" \
  --project-id "$AGENT_NAME" \
  --output compact \
  --max 20 2>/dev/null || exit 0
```

The search results are printed to stdout. The framework captures them and injects them as a context message before the first user turn.

Make the script executable:

```bash
chmod +x hooks/fetch-memory.sh
```

## Example: Pre-tool safety gate

Block dangerous tool calls before execution.

**hooks/pre-tool-safety.yaml:**

```yaml
event: pre_tool_use
command: ./hooks/check-safety.sh
timeout: 3
matcher: "execute_*"
```

**hooks/check-safety.sh:**

```bash
#!/usr/bin/env bash
set -euo pipefail

TOOL_NAME="${TOOL_NAME:-}"
TOOL_ARGS="${TOOL_ARGS:-{}}"

# Block execution tools with risky patterns
if echo "$TOOL_ARGS" | grep -qE '(rm -rf|DROP TABLE|DELETE FROM)'; then
  echo "Blocked: dangerous operation detected in $TOOL_NAME" >&2
  exit 1
fi

exit 0
```

When this hook exits non-zero, the tool call is blocked. The agent receives a structured error message instead of the tool result.

## Example: Post-tool audit logging

Log all database tool calls to an external audit system.

**hooks/post-tool-audit.yaml:**

```yaml
event: post_tool_use
command: ./hooks/audit-db.sh
timeout: 5
matcher: "db_*"
```

**hooks/audit-db.sh:**

```bash
#!/usr/bin/env bash
set -euo pipefail

AUDIT_ENDPOINT="${AUDIT_ENDPOINT:-http://audit-service:8080/log}"

curl -s -X POST "$AUDIT_ENDPOINT" \
  -H "Content-Type: application/json" \
  -d "{
    \"agent\": \"$AGENT_NAME\",
    \"tool\": \"$TOOL_NAME\",
    \"args\": $TOOL_ARGS,
    \"result_preview\": \"${TOOL_RESULT:0:200}\",
    \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"
  }" || true

exit 0
```

## Example: MCP gateway auth

Acquire a Keycloak token before connecting to an authenticated MCP gateway.

**hooks/mcp-auth.yaml:**

```yaml
event: pre_mcp_connect
command: ./hooks/acquire-token.sh
timeout: 5
```

**hooks/acquire-token.sh:**

```bash
#!/usr/bin/env bash
set -euo pipefail

KEYCLOAK_URL="${KEYCLOAK_URL:-}"
CLIENT_ID="${MCP_CLIENT_ID:-}"
CLIENT_SECRET="${MCP_CLIENT_SECRET:-}"
[ -n "$KEYCLOAK_URL" ] || exit 0

TOKEN=$(curl -s -X POST "$KEYCLOAK_URL/protocol/openid-connect/token" \
  -d "grant_type=client_credentials" \
  -d "client_id=$CLIENT_ID" \
  -d "client_secret=$CLIENT_SECRET" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])" 2>/dev/null) || exit 0

echo "$TOKEN" > /tmp/mcp-token
```

**agent.yaml:**

```yaml
mcp_servers:
  - url: ${MCP_GATEWAY_URL}
    headers:
      Authorization: "Bearer ${MCP_AUTH_TOKEN}"
```

The startup script reads the token file and sets the env var before Python starts:

```bash
#!/bin/bash
./hooks/acquire-token.sh
export MCP_AUTH_TOKEN=$(cat /tmp/mcp-token 2>/dev/null)
exec python -m src.agent
```

The `pre_mcp_connect` hook also fires before each connection, so it can refresh the token file for any server that connects after the initial startup.

## Example: Auto-refresh expired MCP tokens

When an MCP tool call fails with 401/403, the framework fires `mcp_auth_refresh`. The hook should acquire a fresh token and print `KEY=VALUE` lines to stdout — the framework updates `os.environ` from these before reconnecting.

**hooks/refresh-token.yaml:**

```yaml
event: mcp_auth_refresh
command: ./hooks/refresh-token.sh
timeout: 10
```

**hooks/refresh-token.sh:**

```bash
#!/usr/bin/env bash
set -euo pipefail

KEYCLOAK_URL="${KEYCLOAK_URL:-}"
CLIENT_ID="${MCP_CLIENT_ID:-}"
CLIENT_SECRET="${MCP_CLIENT_SECRET:-}"
[ -n "$KEYCLOAK_URL" ] || exit 1

TOKEN=$(curl -s -X POST "$KEYCLOAK_URL/protocol/openid-connect/token" \
  -d "grant_type=client_credentials" \
  -d "client_id=$CLIENT_ID" \
  -d "client_secret=$CLIENT_SECRET" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# Framework parses KEY=VALUE lines from stdout into os.environ
echo "MCP_AUTH_TOKEN=$TOKEN"
```

The framework re-resolves `${MCP_AUTH_TOKEN}` in the header templates, creates a new MCP client, and retries the failed tool call once. If the retry also fails, the error reaches the LLM normally.

## Custom events

Fire custom events from your agent subclass:

```python
from fipsagents.baseagent import BaseAgent

class AuthenticatedAgent(BaseAgent):
    async def authenticate(self, user: str):
        results = await self.hooks.fire(
            "pre_auth",
            env_extra={"AUTH_USER": user},
        )
        
        if any(r.blocked for r in results):
            raise PermissionError(f"Authentication blocked by hook: {user}")
        
        # Proceed with authentication...
```

**hooks/pre-auth.yaml:**

```yaml
event: pre_auth
command: ./hooks/check-auth.sh
timeout: 3
```

**hooks/check-auth.sh:**

```bash
#!/usr/bin/env bash
set -euo pipefail

USER="${AUTH_USER:-}"

# Check against a blocklist
if grep -qx "$USER" /etc/agent/blocked-users.txt 2>/dev/null; then
  echo "User $USER is blocked" >&2
  exit 1
fi

exit 0
```

## Graceful degradation

Hooks are designed to fail safely:

- **Timeout:** Process is killed (SIGTERM), session continues
- **Non-zero exit:** Logged as warning (except `pre_tool_use`, which blocks the tool)
- **Command not found:** Logged, session continues
- **No hooks configured:** Zero overhead (short-circuit check)

Hooks should never crash your agent. If a hook misbehaves, the framework logs the failure and moves on.

## Best practices

1. **Keep hooks fast.** Default timeout is 10 seconds. Long-running tasks should be async or delegated to a queue.

2. **Use exit codes intentionally.** Exit 0 for success, non-zero to signal failure. Only `pre_tool_use` treats non-zero as a blocking condition.

3. **Write idempotent hooks.** Hooks may fire multiple times during testing or recovery scenarios.

4. **Log to stderr for diagnostics.** Stdout is captured for injection or inspection. Use stderr for warnings and errors.

5. **Test your hooks in isolation.** Run them manually with the expected environment variables set:

   ```bash
   AGENT_NAME=myagent \
   HOOK_EVENT=setup_complete \
   ./hooks/fetch-memory.sh
   ```

6. **Make scripts executable.** The framework runs them directly, not through `bash -c`.

   ```bash
   chmod +x hooks/*.sh
   ```

7. **Handle missing dependencies gracefully.** Check for required tools and exit 0 if unavailable:

   ```bash
   command -v memoryhub >/dev/null || exit 0
   ```

## Debugging

Enable hook debug logging:

```yaml
logging:
  level: DEBUG
```

You'll see detailed logs for each hook execution:

```
DEBUG fipsagents.baseagent.hooks: Firing hook memory-loader (setup_complete)
DEBUG fipsagents.baseagent.hooks: Hook memory-loader completed in 0.42s (exit 0)
```

Check the agent logs after deployment:

```bash
oc logs deployment/my-agent | grep hooks
```

## Security considerations

Hooks run with the same permissions as the agent process. Be careful with:

- **User input in environment variables.** The framework does not sanitize `TOOL_ARGS` or `TOOL_RESULT`. Validate and escape in your scripts.
- **Secret exposure.** Don't echo secrets to stdout. They may be logged or injected as context.
- **Command injection.** Use parameterized commands or explicit argument parsing. Avoid `eval` or unquoted variable expansion.

For production agents, consider running hooks in a restricted sandbox or using a dedicated service account with minimal privileges.

## Further reading

- See `docs/architecture.md` for the internal hook dispatch mechanism
- See `packages/fipsagents/src/fipsagents/baseagent/hooks.py` for the implementation
- See `templates/agent-loop/hooks/` for additional examples
