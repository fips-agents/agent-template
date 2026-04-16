FROM registry.access.redhat.com/ubi9/python-311:latest

LABEL name="demo-chat-agent" \
      version="0.1.0" \
      description="FIPS-Agents demo chat agent with MemoryHub-backed memory" \
      io.openshift.expose-services="8080:http"

WORKDIR /opt/app-root/src

# Install the vendored fipsagents source first. The deploy script
# stages packages/fipsagents -> vendor/fipsagents before running the
# build so we can install the in-development version instead of the
# PyPI release.
USER 0
COPY vendor/fipsagents/ /tmp/fipsagents-src/
# pip needs to write egg-info into the source dir during install; make
# the staged source writable by anyone before running the install.
RUN chmod -R a+w /tmp/fipsagents-src \
    && pip install --no-cache-dir /tmp/fipsagents-src/ \
    && rm -rf /tmp/fipsagents-src
USER 1001

# Install remaining Python deps. fipsagents is listed in requirements
# but pip will see the vendored version already installed and skip it
# as long as the version constraint is satisfied.
COPY --chmod=644 requirements-container.txt .
RUN pip install --no-cache-dir -r requirements-container.txt

# Copy agent artifacts.
COPY --chmod=644 agent.yaml .
# .memoryhub.yaml is rewritten in place by server.lifespan to expand
# ${VAR:-default} placeholders, so the running user (1001) needs write
# access. chmod 666 keeps the file readable by everyone but writable too.
COPY --chmod=666 .memoryhub.yaml .
COPY --chmod=644 src/__init__.py src/
COPY --chmod=644 src/agent.py src/
COPY --chmod=644 src/server.py src/
COPY --chmod=644 tools/get_current_time.py tools/
COPY --chmod=644 prompts/system.md prompts/

EXPOSE 8080

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8080"]
