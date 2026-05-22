"""Tests for trust scoreboard REST API endpoints."""

from __future__ import annotations

import pytest
import pytest_asyncio
import httpx

from starlette.applications import Starlette

from fipsagents.server.trust_routes import register_trust_routes
from fipsagents.baseagent.trust import TrustManager
from fipsagents.baseagent.skills import SkillLoader


# ---------------------------------------------------------------------------
# Mock Agent
# ---------------------------------------------------------------------------


class MockAgent:
    """Minimal agent mock for testing trust routes."""

    def __init__(self):
        self._trust_manager = TrustManager()
        self.skills = SkillLoader()
        self._discovered_capabilities = {}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def trust_client():
    """ASGI test client backed by a mock agent with trust manager."""
    agent = MockAgent()

    app = Starlette()
    register_trust_routes(app, lambda: agent)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, agent


@pytest_asyncio.fixture
async def no_trust_client():
    """ASGI test client where trust is not enabled (no _trust_manager)."""
    agent = type("Agent", (), {})()
    agent.skills = SkillLoader()
    agent._discovered_capabilities = {}

    app = Starlette()
    register_trust_routes(app, lambda: agent)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest_asyncio.fixture
async def no_agent_client():
    """ASGI test client where the agent is None."""
    app = Starlette()
    register_trust_routes(app, lambda: None)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# ---------------------------------------------------------------------------
# GET /v1/agent/trust
# ---------------------------------------------------------------------------


class TestGetTrust:
    @pytest.mark.asyncio
    async def test_returns_trust_state(self, trust_client):
        client, agent = trust_client

        # Record some events.
        agent._trust_manager.record_completion(quality_score=1.0, reason="test")
        agent._trust_manager.record_failure(severity=0.5, reason="minor error")

        resp = await client.get("/v1/agent/trust")
        assert resp.status_code == 200

        data = resp.json()
        assert data["level"] >= 0
        assert data["score"] >= 0.0
        assert data["completions"] == 1
        assert data["failures"] == 1
        assert data["violations"] == 0
        assert "history" in data
        assert len(data["history"]) == 2

        # Check history structure.
        event = data["history"][0]
        assert "timestamp" in event
        assert event["event_type"] == "completion"
        assert event["delta"] > 0
        assert event["reason"] == "test"

    @pytest.mark.asyncio
    async def test_returns_empty_state_initially(self, trust_client):
        client, _ = trust_client

        resp = await client.get("/v1/agent/trust")
        assert resp.status_code == 200

        data = resp.json()
        assert data["level"] == 0
        assert data["score"] == 0.0
        assert data["completions"] == 0
        assert data["failures"] == 0
        assert data["violations"] == 0
        assert data["last_promotion"] is None
        assert data["last_decay"] is None
        assert data["history"] == []

    @pytest.mark.asyncio
    async def test_no_trust_manager_returns_404(self, no_trust_client):
        client = no_trust_client

        resp = await client.get("/v1/agent/trust")
        assert resp.status_code == 404
        assert "Trust not enabled" in resp.json()["error"]

    @pytest.mark.asyncio
    async def test_no_agent_returns_404(self, no_agent_client):
        client = no_agent_client

        resp = await client.get("/v1/agent/trust")
        assert resp.status_code == 404
        assert "Agent not available" in resp.json()["error"]


# ---------------------------------------------------------------------------
# GET /v1/agent/skills
# ---------------------------------------------------------------------------


class TestGetSkills:
    @pytest.mark.asyncio
    async def test_returns_empty_list_initially(self, trust_client):
        client, _ = trust_client

        resp = await client.get("/v1/agent/skills")
        assert resp.status_code == 200

        data = resp.json()
        assert data["skills"] == []

    @pytest.mark.asyncio
    async def test_returns_skills_after_loading(self, trust_client, tmp_path):
        client, agent = trust_client

        # Create a test skill.
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        skill_dir = skills_dir / "test-skill"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            """---
name: test-skill
description: A test skill
triggers:
  - test
---
Full skill content here.
"""
        )

        # Load the skill.
        agent.skills.load_all(skills_dir)

        resp = await client.get("/v1/agent/skills")
        assert resp.status_code == 200

        data = resp.json()
        assert len(data["skills"]) == 1
        skill = data["skills"][0]
        assert skill["name"] == "test-skill"
        assert skill["description"] == "A test skill"
        assert skill["learned"] is False
        assert skill["activated"] is False

    @pytest.mark.asyncio
    async def test_reflects_activated_state(self, trust_client, tmp_path):
        client, agent = trust_client

        # Create and load a skill.
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        skill_dir = skills_dir / "activated-skill"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            """---
name: activated-skill
description: An activated skill
---
Skill content.
"""
        )

        agent.skills.load_all(skills_dir)

        # Activate it.
        agent.skills.activate("activated-skill")

        resp = await client.get("/v1/agent/skills")
        assert resp.status_code == 200

        skill = resp.json()["skills"][0]
        assert skill["name"] == "activated-skill"
        assert skill["activated"] is True

    @pytest.mark.asyncio
    async def test_no_agent_returns_404(self, no_agent_client):
        client = no_agent_client

        resp = await client.get("/v1/agent/skills")
        assert resp.status_code == 404
        assert "Agent not available" in resp.json()["error"]


# ---------------------------------------------------------------------------
# GET /v1/agent/capabilities
# ---------------------------------------------------------------------------


class TestGetCapabilities:
    @pytest.mark.asyncio
    async def test_returns_empty_list_initially(self, trust_client):
        client, _ = trust_client

        resp = await client.get("/v1/agent/capabilities")
        assert resp.status_code == 200

        data = resp.json()
        assert data["capabilities"] == []

    @pytest.mark.asyncio
    async def test_returns_discovered_capabilities(self, trust_client):
        client, agent = trust_client

        # Add some capabilities.
        agent._discovered_capabilities = {
            "mcp:search": 1.0,
            "skill:summarize": 1.0,
            "tool:python": 0.8,
        }

        resp = await client.get("/v1/agent/capabilities")
        assert resp.status_code == 200

        data = resp.json()
        caps = data["capabilities"]
        assert len(caps) == 3

        # Verify structure.
        cap_names = {c["name"] for c in caps}
        assert cap_names == {"mcp:search", "skill:summarize", "tool:python"}

        # Find specific capability.
        python_cap = next(c for c in caps if c["name"] == "tool:python")
        assert python_cap["value"] == 0.8

    @pytest.mark.asyncio
    async def test_no_agent_returns_404(self, no_agent_client):
        client = no_agent_client

        resp = await client.get("/v1/agent/capabilities")
        assert resp.status_code == 404
        assert "Agent not available" in resp.json()["error"]
