#!/usr/bin/env bash
# Bring up the full feedback-track stack pointed at the Anthropic API
# via the llm-adapter sidecar, and leave everything running for manual
# browser testing.
#
# Layout:
#
#   browser
#      │
#      ▼
#   ui-template        :13000  (static + /v1 reverse proxy)
#      │
#      ▼
#   gateway-template   :18090  (chat + feedback pass-through)
#      │
#      ▼
#   agent (real Claude Haiku, sqlite feedback store, tracing on)
#                      :18080
#      │
#      ▼
#   llm-adapter        :18081  (OpenAI → Anthropic translator)
#      │
#      ▼
#   api.anthropic.com
#
# Usage:
#   scripts/run-stack-anthropic.sh
#   scripts/run-stack-anthropic.sh --stop      # kill anything left running
#   MODEL=claude-haiku-4-5 scripts/run-stack-anthropic.sh
#
# Reads ANTHROPIC_API_KEY from $HOME/.secrets (sourced shell file).
# Ctrl-C to stop, or run again with --stop.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GATEWAY_REPO="${GATEWAY_REPO:-$HOME/Developer/AGENTS/gateway-template}"
UI_REPO="${UI_REPO:-$HOME/Developer/AGENTS/ui-template}"
ADAPTER_DIR="${ADAPTER_DIR:-$REPO_ROOT/packages/llm-adapter}"

# BaseAgent hardcodes the adapter sidecar at http://localhost:8081/v1
# (fipsagents.baseagent.config._ADAPTER_PORT). Stick to that port to
# match the production wiring; the runner monkey-patches it if you set
# ADAPTER_PORT to something else.
ADAPTER_PORT="${ADAPTER_PORT:-8081}"
AGENT_PORT="${AGENT_PORT:-18080}"
GATEWAY_PORT="${GATEWAY_PORT:-18090}"
UI_PORT="${UI_PORT:-13000}"

MODEL="${MODEL:-claude-haiku-4-5}"

PIDFILE="${TMPDIR:-/tmp}/fipsagents-stack.pids"

c_red()   { printf '\033[31m%s\033[0m' "$1"; }
c_green() { printf '\033[32m%s\033[0m' "$1"; }
c_dim()   { printf '\033[2m%s\033[0m' "$1"; }
say()     { echo "$(c_dim '==>') $1"; }
ok()      { echo "  $(c_green ok) $1"; }
die()     { echo "  $(c_red ERR) $1" >&2; exit 1; }

# ---- --stop subcommand -----------------------------------------------------

if [ "${1:-}" = "--stop" ]; then
  if [ ! -f "$PIDFILE" ]; then
    say "no $PIDFILE; nothing to stop"
    exit 0
  fi
  while read -r pid name; do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null && ok "killed $name pid=$pid" || true
    fi
  done <"$PIDFILE"
  rm -f "$PIDFILE"
  exit 0
fi

# Refuse to start if a previous run is still up.
if [ -f "$PIDFILE" ] && grep -q '[0-9]' "$PIDFILE"; then
  while read -r pid _; do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      die "previous run still alive (pid $pid). Run \`$0 --stop\` first."
    fi
  done <"$PIDFILE"
fi
: >"$PIDFILE"
record_pid() { echo "$1 $2" >>"$PIDFILE"; }

# ---- prereqs ---------------------------------------------------------------

say "Checking prerequisites"
for cmd in python3 go curl jq; do
  command -v "$cmd" >/dev/null 2>&1 || die "$cmd not in PATH"
done
[ -d "$GATEWAY_REPO" ] || die "GATEWAY_REPO not found: $GATEWAY_REPO"
[ -d "$UI_REPO" ]      || die "UI_REPO not found: $UI_REPO"
[ -d "$ADAPTER_DIR" ]  || die "ADAPTER_DIR not found: $ADAPTER_DIR"
[ -f "$HOME/.secrets" ] || die "$HOME/.secrets not found"

# ---- secrets ---------------------------------------------------------------

# shellcheck disable=SC1090
. "$HOME/.secrets"
[ -n "${ANTHROPIC_API_KEY:-}" ] || die "ANTHROPIC_API_KEY missing after sourcing ~/.secrets"
ok "ANTHROPIC_API_KEY loaded (${#ANTHROPIC_API_KEY} chars)"

# ---- workdir + venv --------------------------------------------------------

WORKDIR="$REPO_ROOT/.local/stack"
mkdir -p "$WORKDIR/agent/prompts" "$WORKDIR/agent/tools" "$WORKDIR/logs"
say "workdir: $WORKDIR"

if [ ! -d "$WORKDIR/.venv" ]; then
  say "Creating venv"
  python3 -m venv "$WORKDIR/.venv"
fi
# shellcheck disable=SC1091
source "$WORKDIR/.venv/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet -e "$REPO_ROOT/packages/fipsagents[server]" aiosqlite
pip install --quiet -e "$ADAPTER_DIR"
ok "fipsagents + llm-adapter installed"

# ---- agent files -----------------------------------------------------------

cat >"$WORKDIR/agent/prompts/system.md" <<'EOF'
---
name: system
description: System prompt for the local Anthropic-backed smoke agent
---

You are a helpful assistant running locally for end-to-end testing of
the user-feedback feature. Keep replies concise and friendly.
EOF

cat >"$WORKDIR/agent/agent.yaml" <<EOF
agent:
  name: anthropic-smoke
  description: "Real Claude Haiku via the local adapter sidecar"
  version: 0.0.0
model:
  # provider: anthropic tells the framework to route through the
  # adapter sidecar. The endpoint here is informational — the
  # framework rewrites it to http://localhost:\${ADAPTER_PORT}/v1.
  provider: anthropic
  endpoint: http://127.0.0.1:${ADAPTER_PORT}/v1
  name: ${MODEL}
  temperature: 0.7
  max_tokens: 1024
mcp_servers: []
tools:
  local_dir: ./tools
  visibility_default: agent_only
prompts:
  dir: ./prompts
  system: system
loop:
  max_iterations: 3
logging:
  level: INFO
server:
  host: 127.0.0.1
  port: ${AGENT_PORT}
  storage:
    backend: sqlite
    sqlite_path: ${WORKDIR}/agent/agent.db
  sessions:
    enabled: false
  traces:
    enabled: true
    sampling_rate: 1.0
  feedback:
    enabled: true
    max_age_hours: 720
  metrics:
    enabled: false
EOF

cat >"$WORKDIR/agent/run_agent.py" <<EOF
"""Tiny BaseAgent subclass for the Anthropic-backed smoke stack.

Monkey-patches the framework's hardcoded adapter port so the sidecar
can run on a non-standard port without colliding with whatever else
the developer has on :8081.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Override the hardcoded adapter port BEFORE BaseAgent imports it.
import fipsagents.baseagent.config as _bcfg
_bcfg._ADAPTER_PORT = ${ADAPTER_PORT}

from fipsagents.baseagent import BaseAgent, StepResult
from fipsagents.server import OpenAIChatServer


class AnthropicSmokeAgent(BaseAgent):
    async def step(self) -> StepResult:
        response = await self.call_model()
        return StepResult.done(response.content or "")


if __name__ == "__main__":
    cfg = Path(sys.argv[1])
    server = OpenAIChatServer(
        AnthropicSmokeAgent, config_path=cfg, base_dir=cfg.parent,
    )
    server.run(host="127.0.0.1", port=int(sys.argv[2]))
EOF

ok "agent files written to $WORKDIR/agent"

# ---- start adapter ---------------------------------------------------------

say "Starting llm-adapter on :$ADAPTER_PORT (provider=anthropic, model=$MODEL)"
ADAPTER_LOG="$WORKDIR/logs/adapter.log"
( exec env \
    ADAPTER_PROVIDER=anthropic \
    ADAPTER_PORT="$ADAPTER_PORT" \
    ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
    LOG_LEVEL=INFO \
    python -m llm_adapter.app \
) >"$ADAPTER_LOG" 2>&1 &
ADAPTER_PID=$!
record_pid "$ADAPTER_PID" "adapter"

for _ in $(seq 1 40); do
  if curl -fs "http://127.0.0.1:$ADAPTER_PORT/healthz" >/dev/null 2>&1; then
    ok "adapter ready (pid $ADAPTER_PID)"
    break
  fi
  sleep 0.25
done
curl -fs "http://127.0.0.1:$ADAPTER_PORT/healthz" >/dev/null 2>&1 \
  || die "adapter did not come up; tail of $ADAPTER_LOG"

# ---- start agent -----------------------------------------------------------

say "Starting agent on :$AGENT_PORT"
AGENT_LOG="$WORKDIR/logs/agent.log"
( cd "$WORKDIR/agent" \
    && exec env OPENAI_API_KEY="not-required" \
       python run_agent.py agent.yaml "$AGENT_PORT" \
) >"$AGENT_LOG" 2>&1 &
AGENT_PID=$!
record_pid "$AGENT_PID" "agent"

for _ in $(seq 1 60); do
  if curl -fs "http://127.0.0.1:$AGENT_PORT/healthz" >/dev/null 2>&1; then
    ok "agent ready (pid $AGENT_PID)"
    break
  fi
  sleep 0.25
done
curl -fs "http://127.0.0.1:$AGENT_PORT/healthz" >/dev/null 2>&1 \
  || die "agent did not come up; tail of $AGENT_LOG"

# ---- start gateway ---------------------------------------------------------

say "Building + starting gateway on :$GATEWAY_PORT"
GATEWAY_BIN="$WORKDIR/gateway-server"
( cd "$GATEWAY_REPO" && go build -o "$GATEWAY_BIN" ./cmd/server ) \
  || die "gateway build failed"

GATEWAY_LOG="$WORKDIR/logs/gateway.log"
( exec env BACKEND_URL="http://127.0.0.1:$AGENT_PORT" PORT="$GATEWAY_PORT" \
    "$GATEWAY_BIN" \
) >"$GATEWAY_LOG" 2>&1 &
GATEWAY_PID=$!
record_pid "$GATEWAY_PID" "gateway"

for _ in $(seq 1 40); do
  if curl -fs "http://127.0.0.1:$GATEWAY_PORT/healthz" >/dev/null 2>&1; then
    ok "gateway ready (pid $GATEWAY_PID)"
    break
  fi
  sleep 0.25
done
curl -fs "http://127.0.0.1:$GATEWAY_PORT/healthz" >/dev/null 2>&1 \
  || die "gateway did not come up; tail of $GATEWAY_LOG"

# ---- start UI --------------------------------------------------------------

say "Building + starting UI on :$UI_PORT"
UI_BIN="$WORKDIR/ui-server"
( cd "$UI_REPO" && go build -o "$UI_BIN" ./cmd/server ) \
  || die "UI build failed"

UI_LOG="$WORKDIR/logs/ui.log"
( exec env API_URL="http://127.0.0.1:$GATEWAY_PORT" PORT="$UI_PORT" \
    "$UI_BIN" \
) >"$UI_LOG" 2>&1 &
UI_PID=$!
record_pid "$UI_PID" "ui"

for _ in $(seq 1 40); do
  if curl -fs "http://127.0.0.1:$UI_PORT/healthz" >/dev/null 2>&1; then
    ok "UI ready (pid $UI_PID)"
    break
  fi
  sleep 0.25
done
curl -fs "http://127.0.0.1:$UI_PORT/healthz" >/dev/null 2>&1 \
  || die "UI did not come up; tail of $UI_LOG"

# ---- one-shot sanity round-trip --------------------------------------------

say "Sanity check: end-to-end chat through the whole stack"
SANITY_HDR="$WORKDIR/logs/sanity.headers"
SANITY_BODY="$WORKDIR/logs/sanity.body"
if curl -sf -D "$SANITY_HDR" -o "$SANITY_BODY" \
     -X POST "http://127.0.0.1:$UI_PORT/v1/chat/completions" \
     -H 'Content-Type: application/json' \
     -d '{"messages":[{"role":"user","content":"Reply with exactly: pong"}],"stream":false}' ; then
  TRACE_ID=$(awk -F': ' 'tolower($1)=="x-trace-id"{gsub(/\r/,"");print $2}' "$SANITY_HDR" | tr -d ' ')
  CONTENT=$(jq -r '.choices[0].message.content // empty' "$SANITY_BODY" 2>/dev/null)
  if [ -n "$TRACE_ID" ] && [ -n "$CONTENT" ]; then
    ok "real-API round trip succeeded"
    echo "      trace_id : $TRACE_ID"
    echo "      reply    : $(echo "$CONTENT" | head -c 80)"
  else
    echo "      $(c_red 'partial response — trace_id or content missing')"
    echo "      headers: $(tr -d '\r' <"$SANITY_HDR" | head -5)"
    echo "      body:    $(head -c 200 "$SANITY_BODY")"
  fi
else
  echo "      $(c_red 'sanity request failed') — see $SANITY_BODY"
fi

# ---- summary ---------------------------------------------------------------

cat <<EOF

$(c_green '✓ Stack is up.')

  UI       http://127.0.0.1:$UI_PORT     ← open this in a browser
  Gateway  http://127.0.0.1:$GATEWAY_PORT
  Agent    http://127.0.0.1:$AGENT_PORT
  Adapter  http://127.0.0.1:$ADAPTER_PORT  (Anthropic, model=$MODEL)

  PIDs     $(awk '{printf "%s=%s ", $2, $1}' "$PIDFILE")
  Logs     $WORKDIR/logs/

Chat from a terminal:
  curl -s -X POST http://127.0.0.1:$UI_PORT/v1/chat/completions \\
    -H 'Content-Type: application/json' \\
    -d '{"messages":[{"role":"user","content":"hi"}],"stream":false}' | jq

Inspect feedback after clicking thumbs in the browser:
  curl -s http://127.0.0.1:$AGENT_PORT/v1/feedback | jq

Stop everything:
  $0 --stop
EOF
