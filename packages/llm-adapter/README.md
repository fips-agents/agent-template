# LLM Adapter

Sidecar service that translates OpenAI-compatible chat completion requests
to provider-native APIs.  Runs alongside a BaseAgent container in the same
pod, listening on `localhost:8081`.

## Supported Providers

| Provider | Status | Env Var |
|----------|--------|---------|
| Anthropic (Claude) | Implemented | `ANTHROPIC_API_KEY` |
| AWS Bedrock | Planned | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` |
| Azure OpenAI | Planned | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ADAPTER_PROVIDER` | `anthropic` | Target provider |
| `ADAPTER_PORT` | `8081` | Listening port |
| `LOG_LEVEL` | `INFO` | Python log level |

## Development

```bash
make install    # Create venv, install deps
make test       # Run tests
make lint       # Lint with ruff
make build      # Build container (linux/amd64)
make run-local  # Run locally (set provider env vars first)
```

## Deployment

The adapter deploys as a sidecar via the Helm chart in
`templates/agent-loop/chart/`.  Enable it in `values.yaml`:

```yaml
llm_adapter:
  enabled: true
  provider: anthropic
  image:
    repository: quay.io/your-org/llm-adapter
    tag: latest
  secretRef:
    name: anthropic-credentials
```

See `templates/agent-loop/chart/values.yaml` for full configuration.
