FROM registry.access.redhat.com/ubi9/python-311:latest

LABEL name="gemma-tool-calling" \
      version="0.1.0" \
      description="Gemma 4 native tool calling validation agent" \
      io.openshift.expose-services="8080:http"

WORKDIR /opt/app-root/src

# Copy wheel and requirements first for layer caching
COPY --chmod=644 fipsagents-0.6.0.dev0-py3-none-any.whl .
COPY --chmod=644 requirements-container.txt .
RUN pip install --no-cache-dir -r requirements-container.txt

# Copy agent source, prompts, tools, and config
COPY --chmod=644 agent.yaml .
COPY --chmod=644 run.py .
COPY --chmod=644 src/__init__.py src/
COPY --chmod=644 src/agent.py src/
COPY --chmod=644 src/server.py src/
COPY --chmod=644 prompts/system.md prompts/
COPY --chmod=644 tools/get_current_time.py tools/

EXPOSE 8080

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8080"]
