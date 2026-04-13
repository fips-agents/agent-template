# Deploy Workflow

Build the container image and deploy the workflow to OpenShift. This command handles the full path from source code to a running pod.

**Prerequisites: The workflow must be implemented and tested. Run `/create-agent` and ideally `/exercise-agent` first. You also need `oc` CLI access and a target OpenShift cluster.**

## Process

### Step 1: Pre-flight Checks

1. **Tests pass**: Run `make test`. Do not deploy broken code.
2. **No uncommitted changes**: Run `git status`. Warn if there are uncommitted changes.
3. **Configuration review**: Verify `agent.yaml` uses env var substitution for endpoints, no hardcoded secrets.
4. **Containerfile exists**: Verify at project root.
5. **Helm chart exists**: Verify `chart/Chart.yaml` and `chart/values.yaml`.

### Step 2: Determine Build Strategy

- **Option A: Remote build (recommended on Mac)** -- delegate to `remote-builder` agent for x86_64.
- **Option B: Local build** -- `podman build --platform linux/amd64`.
- **Option C: OpenShift BuildConfig** -- trigger cluster-side build.

### Step 3: Prepare and Build

Ensure file permissions (644 for source files), then build with chosen strategy.

### Step 4: Push Image

Push to the registry accessible by OpenShift.

### Step 5: Configure Deployment

Update `chart/values.yaml` with image reference, resources, env var overrides, secrets.

### Step 6: Deploy

`make deploy PROJECT=<namespace>` or `helm upgrade --install`.

### Step 7: Verify

Check pod status, logs, and startup messages. Look for tool discovery, prompt loading, and "setup complete" indicators.

### Step 8: Report

Present deployment status, pod name, image tag, and instructions for monitoring and redeployment.
