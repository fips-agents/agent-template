#!/usr/bin/env bash
# =============================================================================
# deploy.sh — Deploy the agent to OpenShift
# =============================================================================
# Usage: ./deploy.sh <project-namespace>
#
# Prerequisites:
#   - oc CLI installed and logged in
#   - Target namespace already exists (or you have permission to create it)
#   - Container image built and available in a registry
#   - Helm chart or manifests in chart/ directory
# =============================================================================
set -euo pipefail

PROJECT="${1:?Usage: ./deploy.sh <project-namespace>}"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

echo "NOTE: Ensure your container image is pushed to a registry accessible by OpenShift."
echo "  podman push \${IMAGE_NAME}:\${IMAGE_TAG} quay.io/your-org/\${IMAGE_NAME}:\${IMAGE_TAG}"
echo ""

# NOTE: If using OpenShift BuildConfig instead of pre-built images, ensure
# dockerfilePath is set:
#   oc patch bc/<name> -p '{"spec":{"strategy":{"dockerStrategy":{"dockerfilePath":"Containerfile"}}}}' -n "$PROJECT"

if ! command -v oc &>/dev/null; then
    echo "Error: oc CLI not found. Install it from https://mirror.openshift.com/pub/openshift-v4/clients/ocp/"
    exit 1
fi

if ! oc whoami &>/dev/null; then
    echo "Error: Not logged in to OpenShift. Run 'oc login' first."
    exit 1
fi

# Ensure the namespace exists (create if missing and user has permission)
if ! oc get namespace "$PROJECT" &>/dev/null; then
    echo "Namespace '$PROJECT' not found. Creating..."
    oc new-project "$PROJECT" || {
        echo "Error: Could not create namespace '$PROJECT'."
        exit 1
    }
fi

# ---------------------------------------------------------------------------
# Deploy
# ---------------------------------------------------------------------------

CHART_DIR="chart"

if [ -d "$CHART_DIR" ]; then
    echo "Deploying agent to namespace '$PROJECT' from $CHART_DIR/..."

    # If helm is available and Chart.yaml exists, prefer helm install/upgrade
    if command -v helm &>/dev/null && [ -f "$CHART_DIR/Chart.yaml" ]; then
        APP_NAME="$(basename "$(pwd)")"
        helm upgrade --install "$APP_NAME" "$CHART_DIR" \
            -n "$PROJECT" \
            --wait
    else
        # Fallback: apply raw manifests from the chart templates directory
        if [ -d "$CHART_DIR/templates" ]; then
            oc apply -f "$CHART_DIR/templates/" -n "$PROJECT"
        else
            echo "Error: No templates found in $CHART_DIR/. Create your manifests first."
            exit 1
        fi
    fi
else
    echo "Error: $CHART_DIR/ directory not found."
    echo "Create your Helm chart or OpenShift manifests in $CHART_DIR/ before deploying."
    exit 1
fi

echo ""
echo "Deployment complete in namespace '$PROJECT'."
echo "Check status: oc get pods -n $PROJECT"
