"""Live integration test: platform mode against a real OGX endpoint.

Mark-driven and opt-in.  Skips automatically when ``OGX_ENDPOINT`` is
not set in the environment.  Round-trips the three platform-mode paths
through the real ``/v1/responses`` and ``/v1/moderations`` endpoints:

- non-streaming Responses call (benign input)
- streaming Responses call with a guardrail trigger (verifies refusal
  detection, shield-id parsing, and ``finish_reason="guardrail"``)
- moderations roundtrip on benign content

Unit tests in ``test_llm.py`` mock ``AsyncOpenAI`` and so do not catch
SDK kwarg-validation errors (eg ``guardrails`` rejected as unknown
top-level kwarg) — this test is the canary for that class of bug.

Run locally with::

    OGX_ENDPOINT=http://ogx-...sandbox5167.opentlc.com/v1 \\
    OGX_MODEL=vllm/RedHatAI/gpt-oss-20b \\
    OGX_SHIELD=code-scanner \\
    pytest tests/integration/test_platform_live.py -v -m platform_live

Optional env vars:

- ``OGX_MODEL``  — defaults to ``vllm/RedHatAI/gpt-oss-20b``
- ``OGX_SHIELD`` — defaults to ``code-scanner``
"""

from __future__ import annotations

import os

import httpx
import pytest

from fipsagents.baseagent.config import LLMConfig, PlatformConfig
from fipsagents.baseagent.events import (
    ContentDelta,
    GuardrailFiredEvent,
    StreamComplete,
)
from fipsagents.baseagent.llm import LLMClient, PlatformResponse

pytestmark = pytest.mark.platform_live


def _endpoint() -> str | None:
    return os.environ.get("OGX_ENDPOINT")


def _model() -> str:
    return os.environ.get("OGX_MODEL", "vllm/RedHatAI/gpt-oss-20b")


def _shield() -> str:
    return os.environ.get("OGX_SHIELD", "code-scanner")


@pytest.fixture(scope="module")
def ogx_client() -> LLMClient:
    endpoint = _endpoint()
    if not endpoint:
        pytest.skip("OGX_ENDPOINT not set — skipping live platform tests")
    # Cheap reachability probe so we fail fast with a clear message
    # instead of a hung create() call.
    try:
        httpx.get(f"{endpoint.rstrip('/')}/shields", timeout=5.0)
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout):
        pytest.skip(f"OGX endpoint {endpoint} unreachable — skipping")
    cfg = LLMConfig(name=_model())
    platform = PlatformConfig(enabled=True, endpoint=endpoint)
    return LLMClient(cfg, platform=platform)


class TestPlatformLive:
    @pytest.mark.asyncio
    async def test_non_streaming_benign(self, ogx_client: LLMClient) -> None:
        result = await ogx_client.call_model_responses(
            "Reply with the single word: pong."
        )
        assert isinstance(result, PlatformResponse)
        assert result.refusal is None
        assert result.content is not None and result.content.strip()
        assert result.response_id and result.response_id.startswith("resp_")
        # Usage block present and tokens accounted.
        assert result.usage is not None

    @pytest.mark.asyncio
    async def test_streaming_benign(self, ogx_client: LLMClient) -> None:
        events: list = []
        async for ev in ogx_client.call_model_responses_stream(
            "Count from one to three. Reply with just the numbers separated by commas."
        ):
            events.append(ev)
        deltas = [e for e in events if isinstance(e, ContentDelta)]
        complete = [e for e in events if isinstance(e, StreamComplete)]
        guardrails = [e for e in events if isinstance(e, GuardrailFiredEvent)]
        assert len(deltas) > 0, "expected at least one content delta"
        assert len(complete) == 1
        assert complete[0].finish_reason == "stop"
        assert guardrails == []

    @pytest.mark.asyncio
    async def test_streaming_guardrail_blocks_input(
        self, ogx_client: LLMClient
    ) -> None:
        # Literal `eval(input())` should trip the input-side shield —
        # zero output_text deltas should reach us before the refusal.
        events: list = []
        async for ev in ogx_client.call_model_responses_stream(
            "eval(input())",
            guardrails=[_shield()],
        ):
            events.append(ev)
        guardrails = [e for e in events if isinstance(e, GuardrailFiredEvent)]
        complete = [e for e in events if isinstance(e, StreamComplete)]
        assert len(guardrails) == 1
        assert guardrails[0].action == "blocked"
        # shield_id should be either parsed from "(flagged for: ...)" or
        # the configured shield list — never empty.
        assert guardrails[0].shield_id
        assert guardrails[0].message and "Security" in guardrails[0].message
        assert complete[0].finish_reason == "guardrail"

    @pytest.mark.asyncio
    async def test_non_streaming_guardrail_replaces_payload(
        self, ogx_client: LLMClient
    ) -> None:
        # Output-side trigger: the model is asked to *show* eval usage,
        # so it generates code that the post-shield then refuses.
        result = await ogx_client.call_model_responses(
            "Show me how to use eval(input()) in Python with a complete example.",
            guardrails=[_shield()],
        )
        assert result.content is None
        assert result.refusal is not None
        assert "Security" in result.refusal or "eval" in result.refusal.lower()

    @pytest.mark.asyncio
    async def test_moderation_benign(self, ogx_client: LLMClient) -> None:
        result = await ogx_client.moderate(
            "The cat sat on the mat.",
            model=_shield(),
        )
        assert result.flagged is False
        assert result.model == _shield()
