#!/usr/bin/env bash
# Smoke test for the user-feedback feature track.
#
# Spins up a stub agent + the gateway against your local checkout and
# exercises every interesting path:
#
#   1. X-Trace-Id header on sync chat completion
#   2. trace_id field on the final SSE usage chunk (streaming)
#   3. POST /v1/feedback round-trip via the agent (sqlite store)
#   4. POST /v1/feedback via the gateway with auth headers forwarded
#   5. POST /v1/feedback without trace_id (server synthesises one)
#   6. GET /v1/feedback?trace_id=... returns the record we wrote
#   7. GET /v1/feedback/stats?window=hour returns aggregated counts
#   8. (gateway) auth headers (Authorization, X-User-ID) reach the backend
#
# No real LLM is required — the stub agent returns a fixed canned reply
# so the script can run offline.
#
# Usage:
#   scripts/smoke-feedback.sh                      # run all tests
#   scripts/smoke-feedback.sh --keep-running       # leave services up at end
#   AGENT_PORT=18080 GATEWAY_PORT=18090 scripts/smoke-feedback.sh
#
# Exit code is the number of failures (0 = all passed).

set -uo pipefail

# ---- config ----------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GATEWAY_REPO="${GATEWAY_REPO:-$HOME/Developer/AGENTS/gateway-template}"
AGENT_PORT="${AGENT_PORT:-18080}"
GATEWAY_PORT="${GATEWAY_PORT:-18090}"
WORKDIR="$(mktemp -d -t fipsagents-smoke-XXXXXX)"
KEEP_RUNNING=0
[ "${1:-}" = "--keep-running" ] && KEEP_RUNNING=1

AGENT_LOG="$WORKDIR/agent.log"
GATEWAY_LOG="$WORKDIR/gateway.log"
AGENT_PID=""
GATEWAY_PID=""
PASS=0
FAIL=0

# ---- output helpers --------------------------------------------------------

c_red()   { printf '\033[31m%s\033[0m' "$1"; }
c_green() { printf '\033[32m%s\033[0m' "$1"; }
c_dim()   { printf '\033[2m%s\033[0m' "$1"; }

pass() { PASS=$((PASS+1)); echo "  $(c_green PASS) $1"; }
fail() {
  FAIL=$((FAIL+1))
  echo "  $(c_red FAIL) $1"
  shift
  for line in "$@"; do
    echo "       $(c_dim "$line")"
  done
}
section() { echo; echo "== $1 =="; }

cleanup() {
  if [ "$KEEP_RUNNING" = "1" ]; then
    echo
    echo "Services left running:"
    [ -n "$AGENT_PID" ]   && echo "  agent   pid=$AGENT_PID  port=$AGENT_PORT  log=$AGENT_LOG"
    [ -n "$GATEWAY_PID" ] && echo "  gateway pid=$GATEWAY_PID port=$GATEWAY_PORT log=$GATEWAY_LOG"
    echo "  workdir=$WORKDIR"
    echo "Stop with: kill $AGENT_PID $GATEWAY_PID"
    return
  fi
  [ -n "$AGENT_PID" ]   && kill "$AGENT_PID"   2>/dev/null || true
  [ -n "$GATEWAY_PID" ] && kill "$GATEWAY_PID" 2>/dev/null || true
  rm -rf "$WORKDIR"
}
trap cleanup EXIT

# ---- prereq checks ---------------------------------------------------------

section "Prerequisites"
need() {
  if command -v "$1" >/dev/null 2>&1; then
    pass "$1 available"
  else
    fail "$1 not found in PATH"
  fi
}
need python3
need go
need curl
need jq

if [ ! -d "$GATEWAY_REPO" ]; then
  fail "gateway-template not found at $GATEWAY_REPO" \
       "set GATEWAY_REPO=/path/to/gateway-template if it lives elsewhere"
fi

if [ "$FAIL" -gt 0 ]; then
  echo
  echo "Prerequisite checks failed; aborting."
  exit "$FAIL"
fi

# ---- venv + install --------------------------------------------------------

section "Setting up venv"
python3 -m venv "$WORKDIR/.venv"
# shellcheck disable=SC1091
source "$WORKDIR/.venv/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet -e "${REPO_ROOT}/packages/fipsagents[server]" aiosqlite
pass "fipsagents installed in editable mode (server + aiosqlite)"

# ---- write stub agent ------------------------------------------------------

section "Building stub agent"

mkdir -p "$WORKDIR/agent/prompts" "$WORKDIR/agent/tools"

cat >"$WORKDIR/agent/prompts/system.md" <<'EOF'
---
name: system
description: Stub system prompt for smoke testing
---

You are a stub assistant for the fipsagents feedback smoke test.
EOF

cat >"$WORKDIR/agent/agent.yaml" <<EOF
agent:
  name: smoke-stub
  description: "Stub agent for the feedback smoke test"
  version: 0.0.0
model:
  provider: openai
  endpoint: http://127.0.0.1:1/v1
  name: stub-model
  temperature: 0.0
  max_tokens: 16
mcp_servers: []
tools:
  local_dir: ./tools
  visibility_default: agent_only
prompts:
  dir: ./prompts
  system: system
loop:
  max_iterations: 1
logging:
  level: WARNING
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

# Stub agent + launcher.  Bypasses the real LLM by overriding astep_stream
# to yield a deterministic content delta + StreamComplete.
cat >"$WORKDIR/agent/run_stub.py" <<'EOF'
"""Run a stub OpenAIChatServer for smoke testing.  No LLM traffic."""
from __future__ import annotations

import sys
from pathlib import Path

from fipsagents.baseagent import BaseAgent, StepResult
from fipsagents.baseagent.config import load_config
from fipsagents.baseagent.events import ContentDelta, StreamComplete, StreamMetrics
from fipsagents.server import OpenAIChatServer


class StubAgent(BaseAgent):
    async def setup(self) -> None:
        # Load the config but skip LLM/MCP wiring so the smoke test can
        # run with no real model behind it.
        if self._provided_config is not None:
            self.config = self._provided_config
        else:
            self.config = load_config(self._config_path)
        self._setup_done = True

    async def shutdown(self) -> None:
        pass

    async def step(self) -> StepResult:  # type: ignore[override]
        return StepResult.done("stub response")

    async def astep_stream(self, *, max_iterations: int = 1, **_kw):
        yield ContentDelta(content="stub ")
        yield ContentDelta(content="response")
        yield StreamComplete(
            finish_reason="stop",
            metrics=StreamMetrics(
                prompt_tokens=4,
                completion_tokens=2,
                total_tokens=6,
                time_to_first_content=0.01,
                total_time=0.02,
            ),
        )


if __name__ == "__main__":
    cfg = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("agent.yaml")
    server = OpenAIChatServer(StubAgent, config_path=cfg, base_dir=cfg.parent)
    server.run(host="127.0.0.1", port=int(sys.argv[2]))
EOF

pass "stub agent and config written to $WORKDIR/agent"

# ---- start agent -----------------------------------------------------------

section "Starting agent on :$AGENT_PORT"
( cd "$WORKDIR/agent" && python run_stub.py agent.yaml "$AGENT_PORT" >"$AGENT_LOG" 2>&1 ) &
AGENT_PID=$!

# Wait for /healthz.
for _ in $(seq 1 40); do
  if curl -fs "http://127.0.0.1:$AGENT_PORT/healthz" >/dev/null 2>&1; then
    pass "agent ready (pid $AGENT_PID)"
    break
  fi
  sleep 0.25
done
if ! curl -fs "http://127.0.0.1:$AGENT_PORT/healthz" >/dev/null 2>&1; then
  fail "agent did not come up; tail of $AGENT_LOG:" \
       "$(tail -n 20 "$AGENT_LOG" 2>/dev/null | sed 's/^/        /')"
  exit "$FAIL"
fi

# ---- build + start gateway -------------------------------------------------

section "Starting gateway on :$GATEWAY_PORT"
GATEWAY_BIN="$WORKDIR/gateway-server"
( cd "$GATEWAY_REPO" && go build -o "$GATEWAY_BIN" ./cmd/server ) \
  && pass "gateway binary built" \
  || { fail "gateway build failed"; exit "$FAIL"; }

BACKEND_URL="http://127.0.0.1:$AGENT_PORT" PORT="$GATEWAY_PORT" \
  "$GATEWAY_BIN" >"$GATEWAY_LOG" 2>&1 &
GATEWAY_PID=$!

for _ in $(seq 1 40); do
  if curl -fs "http://127.0.0.1:$GATEWAY_PORT/healthz" >/dev/null 2>&1; then
    pass "gateway ready (pid $GATEWAY_PID)"
    break
  fi
  sleep 0.25
done
if ! curl -fs "http://127.0.0.1:$GATEWAY_PORT/healthz" >/dev/null 2>&1; then
  fail "gateway did not come up; tail of $GATEWAY_LOG:" \
       "$(tail -n 20 "$GATEWAY_LOG" 2>/dev/null | sed 's/^/        /')"
  exit "$FAIL"
fi

# ---- helpers used by the assertions ----------------------------------------

# trace_id syntax we generate: trace_<16 hex>.  Also accept W3C 32-hex
# trace IDs in case a propagated traceparent is in play.
TRACE_RE='^(trace_[0-9a-f]{16}|[0-9a-f]{32})$'

# ---- smoke tests -----------------------------------------------------------

section "Sync chat completion → X-Trace-Id header"
sync_resp_hdr="$WORKDIR/sync.headers"
sync_resp_body="$WORKDIR/sync.body"
curl -sf -D "$sync_resp_hdr" -o "$sync_resp_body" \
  -X POST "http://127.0.0.1:$AGENT_PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"stream":false}'
SYNC_TRACE=$(awk -F': ' 'tolower($1)=="x-trace-id"{gsub(/\r/,"");print $2}' "$sync_resp_hdr" | tr -d ' ')
if [[ "$SYNC_TRACE" =~ $TRACE_RE ]]; then
  pass "X-Trace-Id present and well-formed: $SYNC_TRACE"
else
  fail "X-Trace-Id missing or malformed" "got: '$SYNC_TRACE'" "headers: $(tr -d '\r' <"$sync_resp_hdr" | head -10)"
fi

section "Streaming chat completion → trace_id on usage chunk"
stream_body="$WORKDIR/stream.sse"
stream_hdr="$WORKDIR/stream.headers"
curl -sN -D "$stream_hdr" -o "$stream_body" \
  -X POST "http://127.0.0.1:$AGENT_PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"stream":true}'
STREAM_HDR_TRACE=$(awk -F': ' 'tolower($1)=="x-trace-id"{gsub(/\r/,"");print $2}' "$stream_hdr" | tr -d ' ')
if [[ "$STREAM_HDR_TRACE" =~ $TRACE_RE ]]; then
  pass "X-Trace-Id present on streaming response: $STREAM_HDR_TRACE"
else
  fail "X-Trace-Id missing on streaming response" "got: '$STREAM_HDR_TRACE'"
fi

# Pull trace_id out of the usage chunk (the empty-choices frame).
USAGE_TRACE=$(grep '^data: {' "$stream_body" \
  | sed 's/^data: //' \
  | jq -r 'select(.choices==[]) | .trace_id // empty' \
  | head -1)
if [[ "$USAGE_TRACE" =~ $TRACE_RE ]]; then
  pass "usage chunk carries trace_id: $USAGE_TRACE"
else
  fail "trace_id missing from usage chunk" "got: '$USAGE_TRACE'"
fi
if [ "$STREAM_HDR_TRACE" = "$USAGE_TRACE" ]; then
  pass "header and usage-chunk trace_id agree"
else
  fail "header and usage-chunk trace_id disagree" \
       "header=$STREAM_HDR_TRACE" "chunk=$USAGE_TRACE"
fi

section "POST /v1/feedback (with trace_id) → 201 + record retrievable"
TRACE_FOR_FB="$STREAM_HDR_TRACE"
fb_resp=$(curl -s -o "$WORKDIR/fb1.body" -w '%{http_code}' \
  -X POST "http://127.0.0.1:$AGENT_PORT/v1/feedback" \
  -H 'Content-Type: application/json' \
  -d "{\"trace_id\":\"$TRACE_FOR_FB\",\"rating\":1,\"comment\":\"smoke 👍\"}")
if [ "$fb_resp" = "201" ]; then
  FB1_ID=$(jq -r .feedback_id <"$WORKDIR/fb1.body")
  pass "POST /v1/feedback returned 201 (id=$FB1_ID)"
else
  fail "POST /v1/feedback returned $fb_resp" "$(cat "$WORKDIR/fb1.body")"
fi

LIST_RESP=$(curl -sf "http://127.0.0.1:$AGENT_PORT/v1/feedback?trace_id=$TRACE_FOR_FB")
if [ "$(echo "$LIST_RESP" | jq 'length')" = "1" ] \
   && [ "$(echo "$LIST_RESP" | jq -r '.[0].rating')" = "1" ] \
   && [ "$(echo "$LIST_RESP" | jq -r '.[0].trace_id')" = "$TRACE_FOR_FB" ]; then
  pass "GET /v1/feedback?trace_id=… returns the record"
else
  fail "GET /v1/feedback?trace_id=… returned unexpected payload" "$LIST_RESP"
fi

section "POST /v1/feedback without trace_id → server synthesises one"
fb2_resp=$(curl -s -o "$WORKDIR/fb2.body" -w '%{http_code}' \
  -X POST "http://127.0.0.1:$AGENT_PORT/v1/feedback" \
  -H 'Content-Type: application/json' \
  -d '{"rating":-1,"comment":"orphan"}')
if [ "$fb2_resp" = "201" ]; then
  pass "POST /v1/feedback (no trace_id) returned 201"
else
  fail "POST /v1/feedback (no trace_id) returned $fb2_resp" "$(cat "$WORKDIR/fb2.body")"
fi

section "GET /v1/feedback/stats?window=hour"
STATS=$(curl -sf "http://127.0.0.1:$AGENT_PORT/v1/feedback/stats?window=hour")
TOT=$(echo "$STATS" | jq '[.[].total]|add // 0')
UP=$(echo "$STATS"  | jq '[.[].thumbs_up]|add // 0')
DOWN=$(echo "$STATS"| jq '[.[].thumbs_down]|add // 0')
if [ "$TOT" = "2" ] && [ "$UP" = "1" ] && [ "$DOWN" = "1" ]; then
  pass "stats: total=$TOT up=$UP down=$DOWN"
else
  fail "stats unexpected" "got total=$TOT up=$UP down=$DOWN" "raw: $STATS"
fi

section "Gateway pass-through with auth headers"
gw_fb_resp=$(curl -s -o "$WORKDIR/gw_fb.body" -w '%{http_code}' \
  -X POST "http://127.0.0.1:$GATEWAY_PORT/v1/feedback" \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer smoke-token' \
  -H 'X-User-ID: smoke-tester' \
  -d "{\"trace_id\":\"$TRACE_FOR_FB\",\"rating\":1,\"comment\":\"via gateway\"}")
if [ "$gw_fb_resp" = "201" ]; then
  pass "POST /v1/feedback via gateway returned 201"
else
  fail "POST /v1/feedback via gateway returned $gw_fb_resp" "$(cat "$WORKDIR/gw_fb.body")"
fi

section "GET /v1/feedback via gateway returns 3 records"
GW_LIST=$(curl -sf "http://127.0.0.1:$GATEWAY_PORT/v1/feedback?limit=10")
GW_CT=$(echo "$GW_LIST" | jq 'length')
if [ "$GW_CT" = "3" ]; then
  pass "gateway list returned 3 records"
else
  fail "gateway list returned $GW_CT records (expected 3)" "$GW_LIST"
fi

section "GET /v1/feedback/stats via gateway"
GW_STATS=$(curl -sf "http://127.0.0.1:$GATEWAY_PORT/v1/feedback/stats?window=hour")
GW_TOT=$(echo "$GW_STATS" | jq '[.[].total]|add // 0')
if [ "$GW_TOT" = "3" ]; then
  pass "gateway stats: total=$GW_TOT"
else
  fail "gateway stats unexpected" "got total=$GW_TOT" "raw: $GW_STATS"
fi

section "Gateway rejects unsupported method"
M_CODE=$(curl -s -o /dev/null -w '%{http_code}' \
  -X DELETE "http://127.0.0.1:$GATEWAY_PORT/v1/feedback")
if [ "$M_CODE" = "405" ]; then
  pass "DELETE /v1/feedback → 405 (method not allowed)"
else
  fail "DELETE /v1/feedback returned $M_CODE (expected 405)"
fi

# ---- summary ---------------------------------------------------------------

echo
TOTAL=$((PASS+FAIL))
if [ "$FAIL" -eq 0 ]; then
  echo "$(c_green "All $TOTAL checks passed.")"
else
  echo "$(c_red "$FAIL of $TOTAL checks failed.")"
  echo
  echo "Logs:"
  echo "  agent:   $AGENT_LOG"
  echo "  gateway: $GATEWAY_LOG"
fi

exit "$FAIL"
