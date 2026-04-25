# =============================================================================
# LLM Adapter
# =============================================================================

VENV        ?= .venv
PYTHON      ?= python3
IMAGE_NAME  ?= llm-adapter
IMAGE_TAG   ?= latest

PIP         := $(VENV)/bin/pip
PYTEST      := $(VENV)/bin/pytest

.DEFAULT_GOAL := help

.PHONY: install
install: ## Create venv and install all dependencies
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

.PHONY: test
test: ## Run tests
	$(PYTEST) -v

.PHONY: test-cov
test-cov: ## Run tests with coverage
	$(PYTEST) -v --cov=llm_adapter --cov-report=term-missing --cov-report=html

.PHONY: lint
lint: ## Run ruff linter
	@$(VENV)/bin/ruff check src/ tests/ 2>/dev/null \
		|| ($(PIP) install ruff && $(VENV)/bin/ruff check src/ tests/)

.PHONY: build
build: ## Build container image (linux/amd64)
	podman build --platform linux/amd64 -t $(IMAGE_NAME):$(IMAGE_TAG) -f Containerfile . --no-cache

.PHONY: run-local
run-local: ## Run adapter locally (set ADAPTER_PROVIDER and credentials)
	$(VENV)/bin/python -m llm_adapter.app

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
