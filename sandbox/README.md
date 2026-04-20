# Code Execution Sandbox

The sandbox has been extracted to its own repository:

**https://github.com/fips-agents/code-sandbox**

The agent-template's Helm chart still supports the sandbox as a sidecar
(`sandbox.enabled: true` in `values.yaml`). The `code_executor` tool in
`tools/` is the client that talks to the sandbox API.

To deploy a sandbox alongside an agent, build the sandbox image from the
code-sandbox repo and reference it in your deployment values:

```yaml
sandbox:
  enabled: true
  image:
    repository: image-registry.openshift-image-registry.svc:5000/<ns>/code-sandbox
    tag: latest
```

Or scaffold a standalone sandbox instance:

```bash
pip install fips-agents-cli
fips-agents create sandbox my-sandbox
```
