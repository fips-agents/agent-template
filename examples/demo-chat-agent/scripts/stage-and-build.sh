#!/usr/bin/env bash
# Stage the in-tree fipsagents source into ./vendor/ and run an
# OpenShift binary build. Cleans up afterwards regardless of outcome.
#
# Usage: scripts/stage-and-build.sh [-n <namespace>]
# Defaults: namespace=demo-chat-agent
set -euo pipefail

NAMESPACE="${1:-demo-chat-agent}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$DEMO_DIR/../.." && pwd)"
FIPSAGENTS_SRC="$REPO_ROOT/packages/fipsagents"
VENDOR_DIR="$DEMO_DIR/vendor"

if [ ! -d "$FIPSAGENTS_SRC" ]; then
    echo "Error: fipsagents source not found at $FIPSAGENTS_SRC" >&2
    exit 1
fi

cleanup() {
    rm -rf "$VENDOR_DIR"
}
trap cleanup EXIT

echo "==> Staging fipsagents from $FIPSAGENTS_SRC"
rm -rf "$VENDOR_DIR"
mkdir -p "$VENDOR_DIR/fipsagents"
# Copy only what's needed to install the package. Skip egg-info and
# caches.
cp "$FIPSAGENTS_SRC/pyproject.toml"    "$VENDOR_DIR/fipsagents/"
cp "$FIPSAGENTS_SRC/README.md"          "$VENDOR_DIR/fipsagents/" 2>/dev/null || true
cp "$FIPSAGENTS_SRC/LICENSE"            "$VENDOR_DIR/fipsagents/" 2>/dev/null || true
cp -R "$FIPSAGENTS_SRC/src"             "$VENDOR_DIR/fipsagents/"
# Remove build artifacts that would pollute the image.
find "$VENDOR_DIR/fipsagents" -type d -name __pycache__ -exec rm -rf {} +
find "$VENDOR_DIR/fipsagents" -type d -name '*.egg-info' -exec rm -rf {} +

cd "$DEMO_DIR"
echo "==> Starting binary build in namespace $NAMESPACE"
oc start-build demo-chat-agent --from-dir=. --follow -n "$NAMESPACE"
