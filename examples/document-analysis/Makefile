# Document Analysis Workflow — OpenShift deployment
#
# Usage:
#   make deploy PROJECT=document-analysis     # Full deploy (create project, build, deploy)
#   make build-oc PROJECT=document-analysis   # Build only (no Helm deploy)
#   make test                                  # Run unit tests locally
#   make run-local                             # Run the workflow locally
#
# All targets are idempotent — safe to re-run.

PROJECT         ?= document-analysis
IMAGE_NAME      ?= document-analysis
IMAGE_TAG       ?= latest
INTERNAL_REGISTRY = image-registry.openshift-image-registry.svc:5000
CHART_PATH      = ../../templates/agent-loop/chart

IMAGE_REF       = $(INTERNAL_REGISTRY)/$(PROJECT)/$(IMAGE_NAME):$(IMAGE_TAG)

.PHONY: deploy build-oc test run-local verify clean

# ---------------------------------------------------------------------------
# Full deployment: ensure project, build image, deploy via Helm
# ---------------------------------------------------------------------------

deploy: build-oc helm-deploy verify
	@echo ""
	@echo "Deployment complete. Route:"
	@oc get route -n $(PROJECT) -o jsonpath='{.items[0].spec.host}' 2>/dev/null || echo "(no route found)"
	@echo ""

# ---------------------------------------------------------------------------
# OpenShift build via BuildConfig (from local source)
# ---------------------------------------------------------------------------

build-oc: ensure-project
	@echo "--- Building image via BuildConfig ---"
	@if ! oc get bc $(IMAGE_NAME) -n $(PROJECT) >/dev/null 2>&1; then \
		echo "Creating BuildConfig..."; \
		oc new-build --name=$(IMAGE_NAME) --binary --strategy=docker -n $(PROJECT); \
	else \
		echo "BuildConfig $(IMAGE_NAME) already exists"; \
	fi
	oc start-build $(IMAGE_NAME) --from-dir=. --follow -n $(PROJECT)

# ---------------------------------------------------------------------------
# Helm deploy (idempotent via upgrade --install)
# ---------------------------------------------------------------------------

helm-deploy: ensure-project
	@echo "--- Deploying via Helm ---"
	helm upgrade --install $(IMAGE_NAME) $(CHART_PATH) \
		-f values-deploy.yaml \
		--set image.repository=$(INTERNAL_REGISTRY)/$(PROJECT)/$(IMAGE_NAME) \
		--set image.tag=$(IMAGE_TAG) \
		-n $(PROJECT) \
		--wait --timeout 120s

# ---------------------------------------------------------------------------
# Verify: wait for pod ready, print route
# ---------------------------------------------------------------------------

verify:
	@echo "--- Verifying deployment ---"
	oc rollout status deployment/$(IMAGE_NAME) -n $(PROJECT) --timeout=120s
	@echo "Checking readiness probe..."
	@ROUTE=$$(oc get route $(IMAGE_NAME) -n $(PROJECT) -o jsonpath='{.spec.host}' 2>/dev/null); \
	if [ -n "$$ROUTE" ]; then \
		echo "Route: https://$$ROUTE"; \
		echo "Testing /healthz..."; \
		curl -sf "https://$$ROUTE/healthz" && echo " OK" || echo " WARN: healthz not yet reachable (may need a moment)"; \
	else \
		echo "No route found — check oc get route -n $(PROJECT)"; \
	fi

# ---------------------------------------------------------------------------
# Ensure the OpenShift project exists
# ---------------------------------------------------------------------------

ensure-project:
	@if ! oc get project $(PROJECT) >/dev/null 2>&1; then \
		echo "Creating project $(PROJECT)..."; \
		oc new-project $(PROJECT); \
	fi

# ---------------------------------------------------------------------------
# Local development
# ---------------------------------------------------------------------------

test:
	pytest tests/ -v

run-local:
	PYTHONPATH=src:$$PYTHONPATH python run.py

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

clean:
	@echo "This will delete the entire $(PROJECT) namespace and all resources."
	@echo "Press Ctrl-C within 5 seconds to abort..."
	@sleep 5
	oc delete project $(PROJECT) --ignore-not-found
