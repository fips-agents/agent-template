FROM registry.access.redhat.com/ubi9/python-311:latest

LABEL name="code-sandbox-agent" \
      version="0.4.0" \
      description="Code Sandbox Agent — answers questions via sandboxed Python" \
      io.openshift.expose-services="8080:http"

WORKDIR /opt/app-root/src

# Install Python deps first for layer caching
COPY --chmod=644 requirements-container.txt .
RUN pip install --no-cache-dir -r requirements-container.txt

# Copy agent source, tools, prompts, and config
COPY --chmod=644 agent.yaml .
COPY --chmod=644 server.py .
COPY --chmod=644 src/__init__.py src/
COPY --chmod=644 src/agent.py src/
COPY --chmod=644 tools/code_executor.py tools/
COPY --chmod=644 prompts/system.md prompts/

EXPOSE 8080

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
