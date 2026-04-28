"""Tests for fipsagents.server.scanner."""


from __future__ import annotations

import httpx
import pytest

from fipsagents.server.scanner import (
    HttpScanner,
    NullScanner,
    ScanResult,
    VirusScanner,
    create_scanner,
)


# ---------------------------------------------------------------------------
# ScanResult
# ---------------------------------------------------------------------------


class TestScanResult:
    def test_clean(self):
        r = ScanResult.clean()
        assert r.infected is False
        assert r.viruses == []
        assert r.error is None

    def test_found(self):
        r = ScanResult.found(["EICAR-Test-File"])
        assert r.infected is True
        assert r.viruses == ["EICAR-Test-File"]

    def test_failed(self):
        r = ScanResult.failed("connection refused")
        assert r.infected is False
        assert r.error == "connection refused"


# ---------------------------------------------------------------------------
# NullScanner
# ---------------------------------------------------------------------------


class TestNullScanner:
    @pytest.mark.asyncio
    async def test_always_clean(self):
        scanner = NullScanner()
        result = await scanner.scan(b"anything", filename="x.bin")
        assert result.infected is False
        assert result.error is None


# ---------------------------------------------------------------------------
# HttpScanner — uses httpx.MockTransport for full-stack testing
# ---------------------------------------------------------------------------


def _make_scanner_with_handler(
    handler, *, timeout: float = 5.0,
) -> HttpScanner:
    """Construct an HttpScanner whose AsyncClient uses *handler*."""
    transport = httpx.MockTransport(handler)
    scanner = HttpScanner("http://scanner.invalid/scan", timeout_seconds=timeout)
    # Pre-seed the cached client so HttpScanner doesn't make a real one.
    scanner._client = httpx.AsyncClient(transport=transport, timeout=timeout)
    return scanner


class TestHttpScanner:
    def test_empty_url_rejected(self):
        with pytest.raises(ValueError, match="non-empty url"):
            HttpScanner("")

    @pytest.mark.asyncio
    async def test_clean_response(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["content-type"] == "application/octet-stream"
            assert request.headers["x-filename"] == "doc.pdf"
            return httpx.Response(
                200, json={"infected": False, "viruses": []},
            )

        scanner = _make_scanner_with_handler(handler)
        try:
            result = await scanner.scan(b"clean payload", filename="doc.pdf")
        finally:
            await scanner.close()

        assert result.infected is False
        assert result.error is None

    @pytest.mark.asyncio
    async def test_infected_via_200_with_body(self):
        def handler(request):
            return httpx.Response(
                200, json={"infected": True, "viruses": ["EICAR-Test-File"]},
            )

        scanner = _make_scanner_with_handler(handler)
        try:
            result = await scanner.scan(b"x5O...", filename="virus.txt")
        finally:
            await scanner.close()

        assert result.infected is True
        assert result.viruses == ["EICAR-Test-File"]

    @pytest.mark.asyncio
    async def test_infected_via_422_with_body(self):
        def handler(request):
            return httpx.Response(
                422, json={"infected": True, "viruses": ["WIN.Trojan.Foo"]},
            )

        scanner = _make_scanner_with_handler(handler)
        try:
            result = await scanner.scan(b"bad", filename="x.exe")
        finally:
            await scanner.close()

        assert result.infected is True
        assert result.viruses == ["WIN.Trojan.Foo"]

    @pytest.mark.asyncio
    async def test_infected_via_422_without_body_uses_unknown(self):
        def handler(request):
            return httpx.Response(422, content=b"")

        scanner = _make_scanner_with_handler(handler)
        try:
            result = await scanner.scan(b"bad", filename="x")
        finally:
            await scanner.close()

        assert result.infected is True
        assert result.viruses == ["unknown"]

    @pytest.mark.asyncio
    async def test_200_without_json_treated_as_clean(self):
        def handler(request):
            return httpx.Response(200, content=b"OK")

        scanner = _make_scanner_with_handler(handler)
        try:
            result = await scanner.scan(b"x", filename="a")
        finally:
            await scanner.close()

        assert result.infected is False
        assert result.error is None

    @pytest.mark.asyncio
    async def test_500_returns_failed(self):
        def handler(request):
            return httpx.Response(500, content=b"oops")

        scanner = _make_scanner_with_handler(handler)
        try:
            result = await scanner.scan(b"x", filename="a")
        finally:
            await scanner.close()

        assert result.infected is False
        assert result.error is not None
        assert "500" in result.error

    @pytest.mark.asyncio
    async def test_connection_error_returns_failed(self):
        def handler(request):
            raise httpx.ConnectError("nobody listening")

        scanner = _make_scanner_with_handler(handler)
        try:
            result = await scanner.scan(b"x", filename="a")
        finally:
            await scanner.close()

        assert result.infected is False
        assert result.error is not None
        assert "ConnectError" in result.error
        assert "nobody listening" in result.error

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self):
        scanner = _make_scanner_with_handler(
            lambda r: httpx.Response(200, json={"infected": False}),
        )
        await scanner.close()
        # Second close should be a no-op, not raise.
        await scanner.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreateScanner:
    def test_empty_url_returns_null(self):
        scanner = create_scanner(url="")
        assert isinstance(scanner, NullScanner)

    def test_non_empty_url_returns_http(self):
        scanner = create_scanner(url="http://x/y")
        assert isinstance(scanner, HttpScanner)

    def test_returns_virus_scanner_subtype(self):
        # Smoke test: factory output always satisfies the contract.
        assert isinstance(create_scanner(url=""), VirusScanner)
        assert isinstance(create_scanner(url="http://x/y"), VirusScanner)
