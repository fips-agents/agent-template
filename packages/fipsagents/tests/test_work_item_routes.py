"""Tests for work-item REST API endpoints."""

from __future__ import annotations

import pytest
import pytest_asyncio
import httpx

from starlette.applications import Starlette

from fipsagents.server.work_item_routes import register_work_item_routes
from fipsagents.server.work_item_stores.sqlite import SqliteWorkItemStore
from fipsagents.server.work_items import NullWorkItemStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def work_item_client(tmp_path):
    """ASGI test client backed by a real SqliteWorkItemStore."""
    store = SqliteWorkItemStore(str(tmp_path / "test.db"))
    # Trigger table creation.
    await store._get_db()

    app = Starlette()
    register_work_item_routes(app, lambda: store)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, store

    await store.close()


@pytest_asyncio.fixture
async def disabled_client():
    """ASGI test client where work items are disabled (NullWorkItemStore)."""
    store = NullWorkItemStore()
    app = Starlette()
    register_work_item_routes(app, lambda: store)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest_asyncio.fixture
async def none_store_client():
    """ASGI test client where the store is None."""
    app = Starlette()
    register_work_item_routes(app, lambda: None)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _create_item(client: httpx.AsyncClient, **overrides) -> dict:
    """Create a work item via the API and return its JSON body."""
    payload = {"title": "Test item", "description": "Do something"}
    payload.update(overrides)
    resp = await client.post("/v1/work-items", json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()


async def _checkout(
    client: httpx.AsyncClient,
    item_id: str,
    actor_id: str = "agent-1",
) -> dict:
    """Check out an item and return its JSON body."""
    resp = await client.post(
        f"/v1/work-items/{item_id}/checkout",
        json={"actor_id": actor_id},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# POST /v1/work-items (create)
# ---------------------------------------------------------------------------


class TestCreateWorkItem:
    @pytest.mark.asyncio
    async def test_creates_item(self, work_item_client):
        client, _ = work_item_client
        data = await _create_item(client, title="Build widget")
        assert data["title"] == "Build widget"
        assert data["status"] == "available"
        assert data["id"].startswith("wi_")

    @pytest.mark.asyncio
    async def test_with_optional_fields(self, work_item_client):
        client, _ = work_item_client
        data = await _create_item(
            client,
            title="Complex task",
            priority=5,
            parent_id="parent-1",
            depends_on=["dep-1"],
            acceptance_criteria=["passes tests"],
            max_tokens=1000,
            max_cost_usd=0.50,
            max_duration_seconds=600,
            required_capabilities=[{"name": "python", "value": 0.8}],
            created_by="ci-pipeline",
        )
        assert data["priority"] == 5
        assert data["parent_id"] == "parent-1"
        assert data["depends_on"] == ["dep-1"]
        assert data["acceptance_criteria"] == ["passes tests"]
        assert data["max_tokens"] == 1000
        assert data["created_by"] == "ci-pipeline"
        assert data["required_capabilities"] == [
            {"name": "python", "value": 0.8},
        ]

    @pytest.mark.asyncio
    async def test_missing_title_returns_400(self, work_item_client):
        client, _ = work_item_client
        resp = await client.post(
            "/v1/work-items", json={"description": "no title"},
        )
        assert resp.status_code == 400
        assert "title" in resp.json()["error"]


# ---------------------------------------------------------------------------
# GET /v1/work-items (list)
# ---------------------------------------------------------------------------


class TestListWorkItems:
    @pytest.mark.asyncio
    async def test_lists_available_items(self, work_item_client):
        client, _ = work_item_client
        await _create_item(client, title="Item A")
        await _create_item(client, title="Item B")

        resp = await client.get("/v1/work-items")
        assert resp.status_code == 200
        items = resp.json()
        assert len(items) == 2
        titles = {i["title"] for i in items}
        assert titles == {"Item A", "Item B"}

    @pytest.mark.asyncio
    async def test_filters_by_parent_id(self, work_item_client):
        client, _ = work_item_client
        await _create_item(client, title="Child", parent_id="parent-1")
        await _create_item(client, title="Other")

        resp = await client.get("/v1/work-items?parent_id=parent-1")
        assert resp.status_code == 200
        items = resp.json()
        assert len(items) == 1
        assert items[0]["title"] == "Child"

    @pytest.mark.asyncio
    async def test_max_results(self, work_item_client):
        client, _ = work_item_client
        for i in range(5):
            await _create_item(client, title=f"Item {i}")

        resp = await client.get("/v1/work-items?max_results=3")
        assert resp.status_code == 200
        assert len(resp.json()) == 3


# ---------------------------------------------------------------------------
# GET /v1/work-items/stats
# ---------------------------------------------------------------------------


class TestStatsWorkItems:
    @pytest.mark.asyncio
    async def test_stats_empty(self, work_item_client):
        client, _ = work_item_client
        resp = await client.get("/v1/work-items/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["counts"] == {}
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_stats_with_items(self, work_item_client):
        client, _ = work_item_client
        item_a = await _create_item(client, title="Item A")
        await _create_item(client, title="Item B")
        await _checkout(client, item_a["id"])

        resp = await client.get("/v1/work-items/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["counts"]["available"] == 1
        assert data["counts"]["checked_out"] == 1
        assert data["total"] == 2

    @pytest.mark.asyncio
    async def test_stats_disabled(self, disabled_client):
        resp = await disabled_client.get("/v1/work-items/stats")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_stats_none_store(self, none_store_client):
        resp = await none_store_client.get("/v1/work-items/stats")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /v1/work-items/{item_id} (get)
# ---------------------------------------------------------------------------


class TestGetWorkItem:
    @pytest.mark.asyncio
    async def test_returns_item(self, work_item_client):
        client, _ = work_item_client
        created = await _create_item(client, title="Fetch me")

        resp = await client.get(f"/v1/work-items/{created['id']}")
        assert resp.status_code == 200
        assert resp.json()["title"] == "Fetch me"

    @pytest.mark.asyncio
    async def test_not_found(self, work_item_client):
        client, _ = work_item_client
        resp = await client.get("/v1/work-items/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /v1/work-items/{item_id}/checkout
# ---------------------------------------------------------------------------


class TestCheckoutWorkItem:
    @pytest.mark.asyncio
    async def test_checkout_succeeds(self, work_item_client):
        client, _ = work_item_client
        created = await _create_item(client)
        data = await _checkout(client, created["id"])

        assert data["status"] == "checked_out"
        assert data["assignee"] == "agent-1"

    @pytest.mark.asyncio
    async def test_double_checkout_returns_409(self, work_item_client):
        client, _ = work_item_client
        created = await _create_item(client)
        await _checkout(client, created["id"], actor_id="agent-1")

        resp = await client.post(
            f"/v1/work-items/{created['id']}/checkout",
            json={"actor_id": "agent-2"},
        )
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_missing_actor_id(self, work_item_client):
        client, _ = work_item_client
        created = await _create_item(client)
        resp = await client.post(
            f"/v1/work-items/{created['id']}/checkout", json={},
        )
        assert resp.status_code == 400
        assert "actor_id" in resp.json()["error"]

    @pytest.mark.asyncio
    async def test_checkout_with_lease_duration(self, work_item_client):
        client, _ = work_item_client
        created = await _create_item(client)
        resp = await client.post(
            f"/v1/work-items/{created['id']}/checkout",
            json={"actor_id": "agent-1", "lease_duration_seconds": 600},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /v1/work-items/{item_id}/complete
# ---------------------------------------------------------------------------


class TestCompleteWorkItem:
    @pytest.mark.asyncio
    async def test_complete_succeeds(self, work_item_client):
        client, _ = work_item_client
        created = await _create_item(client)
        await _checkout(client, created["id"])

        resp = await client.post(
            f"/v1/work-items/{created['id']}/complete",
            json={"result": {"output": "done"}},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"

    @pytest.mark.asyncio
    async def test_complete_with_handoff_note(self, work_item_client):
        client, _ = work_item_client
        created = await _create_item(client)
        await _checkout(client, created["id"])

        resp = await client.post(
            f"/v1/work-items/{created['id']}/complete",
            json={
                "accomplished": ["built the widget"],
                "remaining": ["deploy"],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["handoff_note"]["accomplished"] == ["built the widget"]

    @pytest.mark.asyncio
    async def test_complete_with_review_required(self, work_item_client):
        client, _ = work_item_client
        created = await _create_item(client)
        await _checkout(client, created["id"])

        resp = await client.post(
            f"/v1/work-items/{created['id']}/complete",
            json={"review_required": True},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "review_pending"

    @pytest.mark.asyncio
    async def test_complete_not_checked_out_returns_409(self, work_item_client):
        client, _ = work_item_client
        created = await _create_item(client)

        resp = await client.post(
            f"/v1/work-items/{created['id']}/complete", json={},
        )
        assert resp.status_code == 409


# ---------------------------------------------------------------------------
# POST /v1/work-items/{item_id}/release
# ---------------------------------------------------------------------------


class TestReleaseWorkItem:
    @pytest.mark.asyncio
    async def test_release_succeeds(self, work_item_client):
        client, _ = work_item_client
        created = await _create_item(client)
        await _checkout(client, created["id"])

        resp = await client.post(
            f"/v1/work-items/{created['id']}/release",
            json={"remaining": ["finish the job"], "blockers": ["no API key"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "available"
        assert data["handoff_note"]["remaining"] == ["finish the job"]
        assert data["handoff_note"]["blockers"] == ["no API key"]

    @pytest.mark.asyncio
    async def test_release_not_checked_out_returns_409(self, work_item_client):
        client, _ = work_item_client
        created = await _create_item(client)

        resp = await client.post(
            f"/v1/work-items/{created['id']}/release", json={},
        )
        assert resp.status_code == 409


# ---------------------------------------------------------------------------
# POST /v1/work-items/{item_id}/accept
# ---------------------------------------------------------------------------


class TestAcceptWorkItem:
    @pytest.mark.asyncio
    async def test_accept_succeeds(self, work_item_client):
        client, _ = work_item_client
        created = await _create_item(client)
        await _checkout(client, created["id"])

        # Complete with review_required to get to review_pending state.
        await client.post(
            f"/v1/work-items/{created['id']}/complete",
            json={"review_required": True},
        )

        resp = await client.post(
            f"/v1/work-items/{created['id']}/accept", json={},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"

    @pytest.mark.asyncio
    async def test_accept_wrong_status_returns_409(self, work_item_client):
        client, _ = work_item_client
        created = await _create_item(client)

        resp = await client.post(
            f"/v1/work-items/{created['id']}/accept", json={},
        )
        assert resp.status_code == 409


# ---------------------------------------------------------------------------
# POST /v1/work-items/{item_id}/reject
# ---------------------------------------------------------------------------


class TestRejectWorkItem:
    @pytest.mark.asyncio
    async def test_reject_succeeds(self, work_item_client):
        client, _ = work_item_client
        created = await _create_item(client)
        await _checkout(client, created["id"])
        await client.post(
            f"/v1/work-items/{created['id']}/complete",
            json={"review_required": True},
        )

        resp = await client.post(
            f"/v1/work-items/{created['id']}/reject",
            json={"reason": "Tests fail"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "available"

    @pytest.mark.asyncio
    async def test_reject_missing_reason_returns_400(self, work_item_client):
        client, _ = work_item_client
        created = await _create_item(client)
        await _checkout(client, created["id"])
        await client.post(
            f"/v1/work-items/{created['id']}/complete",
            json={"review_required": True},
        )

        resp = await client.post(
            f"/v1/work-items/{created['id']}/reject", json={},
        )
        assert resp.status_code == 400
        assert "reason" in resp.json()["error"]


# ---------------------------------------------------------------------------
# DELETE /v1/work-items/{item_id}
# ---------------------------------------------------------------------------


class TestDeleteWorkItem:
    @pytest.mark.asyncio
    async def test_delete_returns_204(self, work_item_client):
        client, _ = work_item_client
        created = await _create_item(client)

        resp = await client.delete(f"/v1/work-items/{created['id']}")
        assert resp.status_code == 204

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_404(self, work_item_client):
        client, _ = work_item_client
        resp = await client.delete("/v1/work-items/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Disabled store guard
# ---------------------------------------------------------------------------


class TestDisabledStore:
    """All endpoints return 404 when work items are disabled."""

    @pytest.mark.asyncio
    async def test_create_disabled(self, disabled_client):
        resp = await disabled_client.post(
            "/v1/work-items", json={"title": "nope"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_list_disabled(self, disabled_client):
        resp = await disabled_client.get("/v1/work-items")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_disabled(self, disabled_client):
        resp = await disabled_client.get("/v1/work-items/some-id")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_checkout_disabled(self, disabled_client):
        resp = await disabled_client.post(
            "/v1/work-items/some-id/checkout",
            json={"actor_id": "a"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_complete_disabled(self, disabled_client):
        resp = await disabled_client.post(
            "/v1/work-items/some-id/complete", json={},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_release_disabled(self, disabled_client):
        resp = await disabled_client.post(
            "/v1/work-items/some-id/release", json={},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_accept_disabled(self, disabled_client):
        resp = await disabled_client.post(
            "/v1/work-items/some-id/accept", json={},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_reject_disabled(self, disabled_client):
        resp = await disabled_client.post(
            "/v1/work-items/some-id/reject", json={"reason": "x"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_disabled(self, disabled_client):
        resp = await disabled_client.delete("/v1/work-items/some-id")
        assert resp.status_code == 404


class TestNoneStore:
    """All endpoints return 404 when the store is None."""

    @pytest.mark.asyncio
    async def test_create_none(self, none_store_client):
        resp = await none_store_client.post(
            "/v1/work-items", json={"title": "nope"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_list_none(self, none_store_client):
        resp = await none_store_client.get("/v1/work-items")
        assert resp.status_code == 404
