"""Tests for HttpWebhookSource with HMAC-SHA256 verification."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from fipsagents.server.sources.webhook import HttpWebhookSource


# -- Helpers ----------------------------------------------------------------


def _make_config(
    path: str = "/events/test",
    secret: str | None = None,
    signature_header: str = "X-Hub-Signature-256",
    event_type_header: str = "X-GitHub-Event",
    max_events_per_second: float = 0,  # disable rate limiting in tests
) -> SimpleNamespace:
    return SimpleNamespace(
        path=path,
        secret=secret,
        signature_header=signature_header,
        event_type_header=event_type_header,
        max_events_per_second=max_events_per_second,
    )


def _sign(body: bytes, secret: str) -> str:
    """Compute sha256=<hex> HMAC signature."""
    return "sha256=" + hmac.new(
        secret.encode(), body, hashlib.sha256,
    ).hexdigest()


# -- Fixtures ---------------------------------------------------------------


@pytest.fixture
def app() -> FastAPI:
    return FastAPI()


@pytest.fixture
async def signed_source_and_client(app: FastAPI):
    """Source configured with HMAC secret + an httpx test client."""
    config = _make_config(secret="test-secret")
    source = HttpWebhookSource("test-webhook", config=config)
    await source.setup(app=app)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield source, client
    await source.close()


@pytest.fixture
async def unsigned_source_and_client(app: FastAPI):
    """Source configured without HMAC secret."""
    config = _make_config(secret=None)
    source = HttpWebhookSource("test-unsigned", config=config)
    await source.setup(app=app)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield source, client
    await source.close()


# -- HMAC verification tests ------------------------------------------------


class TestHmacVerification:
    async def test_valid_signature_returns_202(
        self, signed_source_and_client,
    ):
        source, client = signed_source_and_client
        body = json.dumps({"action": "opened"}).encode()
        sig = _sign(body, "test-secret")
        resp = await client.post(
            "/events/test",
            content=body,
            headers={
                "X-Hub-Signature-256": sig,
                "Content-Type": "application/json",
            },
        )
        assert resp.status_code == 202
        assert resp.json()["status"] == "accepted"

    async def test_invalid_signature_returns_401(
        self, signed_source_and_client,
    ):
        _source, client = signed_source_and_client
        body = json.dumps({"action": "opened"}).encode()
        resp = await client.post(
            "/events/test",
            content=body,
            headers={
                "X-Hub-Signature-256": "sha256=badhex",
                "Content-Type": "application/json",
            },
        )
        assert resp.status_code == 401
        assert "Invalid signature" in resp.json()["error"]

    async def test_missing_signature_header_returns_401(
        self, signed_source_and_client,
    ):
        _source, client = signed_source_and_client
        body = json.dumps({"action": "opened"}).encode()
        resp = await client.post(
            "/events/test",
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 401
        assert "Missing signature" in resp.json()["error"]

    async def test_no_secret_configured_skips_verification(
        self, unsigned_source_and_client,
    ):
        source, client = unsigned_source_and_client
        body = json.dumps({"action": "opened"}).encode()
        resp = await client.post(
            "/events/test",
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 202


# -- Consume tests ----------------------------------------------------------


class TestWebhookSourceConsume:
    async def test_posted_event_is_yielded(
        self, unsigned_source_and_client,
    ):
        source, client = unsigned_source_and_client
        body = json.dumps({"ref": "refs/heads/main"}).encode()
        resp = await client.post(
            "/events/test",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-GitHub-Event": "push",
            },
        )
        assert resp.status_code == 202

        event = await asyncio.wait_for(source._queue.get(), timeout=2.0)
        assert event.event_type == "push"
        assert event.payload["ref"] == "refs/heads/main"
        assert event.source == "test-unsigned"

    async def test_multiple_events_fifo_order(
        self, unsigned_source_and_client,
    ):
        source, client = unsigned_source_and_client
        for i in range(3):
            body = json.dumps({"seq": i}).encode()
            await client.post(
                "/events/test",
                content=body,
                headers={"Content-Type": "application/json"},
            )

        for i in range(3):
            event = await asyncio.wait_for(source._queue.get(), timeout=2.0)
            assert event.payload["seq"] == i

    async def test_event_type_from_header(
        self, unsigned_source_and_client,
    ):
        source, client = unsigned_source_and_client
        body = json.dumps({}).encode()
        await client.post(
            "/events/test",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-GitHub-Event": "pull_request",
            },
        )
        event = await asyncio.wait_for(source._queue.get(), timeout=2.0)
        assert event.event_type == "pull_request"

    async def test_event_type_defaults_to_webhook(
        self, unsigned_source_and_client,
    ):
        source, client = unsigned_source_and_client
        body = json.dumps({}).encode()
        await client.post(
            "/events/test",
            content=body,
            headers={"Content-Type": "application/json"},
        )
        event = await asyncio.wait_for(source._queue.get(), timeout=2.0)
        assert event.event_type == "webhook"

    async def test_event_ids_are_unique(
        self, unsigned_source_and_client,
    ):
        source, client = unsigned_source_and_client
        for _ in range(3):
            body = json.dumps({}).encode()
            await client.post(
                "/events/test",
                content=body,
                headers={"Content-Type": "application/json"},
            )

        ids = set()
        for _ in range(3):
            event = await asyncio.wait_for(source._queue.get(), timeout=2.0)
            ids.add(event.event_id)
        assert len(ids) == 3

    async def test_non_json_body_wrapped(
        self, unsigned_source_and_client,
    ):
        source, client = unsigned_source_and_client
        await client.post(
            "/events/test",
            content=b"plain text payload",
            headers={"Content-Type": "text/plain"},
        )
        event = await asyncio.wait_for(source._queue.get(), timeout=2.0)
        assert event.payload == {"raw": "plain text payload"}

    async def test_session_key_uses_path(
        self, unsigned_source_and_client,
    ):
        source, client = unsigned_source_and_client
        body = json.dumps({}).encode()
        await client.post(
            "/events/test",
            content=body,
            headers={"Content-Type": "application/json"},
        )
        event = await asyncio.wait_for(source._queue.get(), timeout=2.0)
        assert event.session_key == "event:/events/test"


# -- Lifecycle tests ---------------------------------------------------------


class TestWebhookSourceLifecycle:
    async def test_setup_without_app_raises(self):
        config = _make_config()
        source = HttpWebhookSource("no-app", config=config)
        with pytest.raises(ValueError, match="requires app="):
            await source.setup()

    async def test_setup_registers_route(self, app: FastAPI):
        config = _make_config(path="/hooks/ci")
        source = HttpWebhookSource("ci-hook", config=config)
        await source.setup(app=app)

        # Verify the route exists on the app
        routes = [r.path for r in app.routes]
        assert "/hooks/ci" in routes

    async def test_close_is_noop(self):
        config = _make_config()
        source = HttpWebhookSource("noop", config=config)
        await source.close()  # should not raise

    async def test_custom_headers(self, app: FastAPI):
        config = _make_config(
            secret="s3cret",
            signature_header="X-Custom-Sig",
            event_type_header="X-Custom-Event",
        )
        source = HttpWebhookSource("custom", config=config)
        await source.setup(app=app)
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test",
        ) as client:
            body = json.dumps({"ok": True}).encode()
            sig = _sign(body, "s3cret")

            # Custom signature header accepted
            resp = await client.post(
                "/events/test",
                content=body,
                headers={
                    "X-Custom-Sig": sig,
                    "X-Custom-Event": "deploy",
                    "Content-Type": "application/json",
                },
            )
            assert resp.status_code == 202

            event = await asyncio.wait_for(source._queue.get(), timeout=2.0)
            assert event.event_type == "deploy"

    async def test_consume_yields_queued_events(self, app: FastAPI):
        config = _make_config()
        source = HttpWebhookSource("consume-test", config=config)
        await source.setup(app=app)
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test",
        ) as client:
            body = json.dumps({"val": 42}).encode()
            await client.post(
                "/events/test",
                content=body,
                headers={"Content-Type": "application/json"},
            )

        # Use the consume() async generator directly
        gen = source.consume()
        event = await asyncio.wait_for(gen.__anext__(), timeout=2.0)
        assert event.payload["val"] == 42
