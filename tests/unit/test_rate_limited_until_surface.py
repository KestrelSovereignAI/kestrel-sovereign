"""A declined advised wait reaches the caller as a rate limit with a reset time (#3127).

Before, the retry loop spent sixteen minutes on capped attempts against advice
to wait hours, and the invoke endpoint then answered ``500 invoke_failed``:
the caller learned nothing about when a retry could succeed. The reset time is
the provider's number, not caller content or provider prose, so it may cross
the safe error boundaries: the HTTP path as ``429`` with ``Retry-After``, the
streaming paths as the constant route-guidance message plus that time.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from kestrel_sovereign.api_errors import rate_limited_until
from kestrel_sovereign.llm.retry import AdvisedWaitExceedsRetryBudget
from kestrel_sovereign.llm.streaming_errors import (
    agent_stream_error_block,
    bridge_sse_error_event,
    safe_streaming_error_message,
)

pytestmark = pytest.mark.usefixtures("isolated_process_rate_limiter")

PROVIDER_PROSE = "Error code: 429 - rate_limit_error WITHHELD-PROVIDER-TEXT-4c1d"
RESET = datetime(2026, 8, 26, 21, 14, 5, tzinfo=UTC)


class _Throttle(Exception):
    status_code = 429


def _declined(advised: float = 6832.4) -> AdvisedWaitExceedsRetryBudget:
    return AdvisedWaitExceedsRetryBudget(
        _Throttle(PROVIDER_PROSE),
        advised_seconds=advised,
        budget_seconds=840,
        retry_at=RESET,
    )


def _wrapped(declined: AdvisedWaitExceedsRetryBudget) -> RuntimeError:
    """The shape the service hands up: ``RuntimeError(...) from declined``."""
    try:
        raise RuntimeError(f"Model x failed: {declined}") from declined
    except RuntimeError as wrapped:
        return wrapped


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def test_rate_limited_until_is_a_429_with_retry_after_rounded_up():
    exc = rate_limited_until(_declined(6832.4))
    assert exc.status_code == 429
    assert exc.code == "rate_limited"
    assert exc.headers == {"Retry-After": "6833"}
    assert "2026-08-26T21:14:05+00:00" in exc.message
    assert PROVIDER_PROSE not in exc.message


def test_retry_after_is_at_least_one_second():
    assert rate_limited_until(_declined(0.2)).headers == {"Retry-After": "1"}


# ---------------------------------------------------------------------------
# Streams
# ---------------------------------------------------------------------------


def test_the_stream_message_names_the_reset_time_and_nothing_from_the_provider():
    message = safe_streaming_error_message(_wrapped(_declined()))
    assert message.startswith("Your selected model route is rate limited.")
    assert "wait until 2026-08-26T21:14:05+00:00" in message
    assert PROVIDER_PROSE not in message
    assert "WITHHELD" not in message


def test_the_agent_stream_block_and_bridge_event_carry_the_same_message():
    wrapped = _wrapped(_declined())
    block = agent_stream_error_block(wrapped)
    assert block.startswith("\n\n---\n⚠️ **Your selected model route is rate limited.**")
    assert "2026-08-26T21:14:05+00:00" in block

    event = bridge_sse_error_event(wrapped)
    assert event.startswith("data: ") and event.endswith("\n\n")
    payload = json.loads(event[len("data: "):])
    assert payload["type"] == "error"
    assert payload["message"] == safe_streaming_error_message(wrapped)
    assert PROVIDER_PROSE not in event


def test_an_unrelated_failure_still_gets_the_generic_constant():
    message = safe_streaming_error_message(RuntimeError(PROVIDER_PROSE))
    assert message.startswith("Error generating response.")
    assert PROVIDER_PROSE not in message


# ---------------------------------------------------------------------------
# The invoke endpoint
# ---------------------------------------------------------------------------

def _boot_app(process_input_error: Exception):
    from server import app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original = {
        "lifespan": app.router.lifespan_context,
        "agent": getattr(app.state, "agent", None),
        "manager": getattr(app.state, "agent_manager", None),
    }
    agent = MagicMock()
    agent.agent_id = "did:pkh:eip155:1:0xabc"
    agent.privacy_mode = MagicMock()
    agent.privacy_mode.value = "NORMAL"
    agent.features = {}
    agent.process_input = AsyncMock(side_effect=process_input_error)
    agent.register_active_request = MagicMock()
    agent.is_request_cancelled = MagicMock(return_value=False)
    agent._cleanup_cancelled_request = MagicMock()
    agent.storage.resolve_session_id = AsyncMock(return_value="sess-1")
    app.router.lifespan_context = noop_lifespan
    app.state.agent = agent
    app.state.agent_manager = None

    def restore():
        app.router.lifespan_context = original["lifespan"]
        app.state.agent = original["agent"]
        app.state.agent_manager = original["manager"]

    return app, restore


def _invoke(app):
    with patch.dict(os.environ, {"KESTREL_API_KEY": "test-key"}), TestClient(app) as client:
        return client.post(
            "/api/agent/invoke",
            json={"input": "merge PR #3112"},
            headers={"X-API-Key": "test-key"},
        )


def test_invoke_answers_429_with_retry_after_when_the_route_declined_to_wait():
    """Emma's instance: advised 6832 s; before this the caller got
    ``500 invoke_failed`` after sixteen minutes."""
    app, restore = _boot_app(_wrapped(_declined(6832.4)))
    try:
        response = _invoke(app)
    finally:
        restore()
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "6833"
    body = response.json()
    assert body["error"]["code"] == "rate_limited"
    assert "2026-08-26T21:14:05+00:00" in body["detail"]
    assert PROVIDER_PROSE not in response.text and "WITHHELD" not in response.text


def test_invoke_still_answers_500_for_any_other_failure():
    app, restore = _boot_app(RuntimeError(PROVIDER_PROSE))
    try:
        response = _invoke(app)
    finally:
        restore()
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "invoke_failed"
    assert PROVIDER_PROSE not in response.text
