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
from datetime import UTC, datetime, timedelta
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
NOW = datetime(2026, 8, 26, 19, 20, 12, tzinfo=UTC)
RESET = datetime(2026, 8, 26, 21, 14, 5, tzinfo=UTC)  # 6833 s after NOW


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
    exc = rate_limited_until(_declined(6832.4), now=NOW)
    assert exc.status_code == 429
    assert exc.code == "rate_limited"
    assert exc.headers == {"Retry-After": "6833"}
    assert exc.message.startswith("The model route is rate limited until 2026-08-26T21:14:05+00:00")
    assert PROVIDER_PROSE not in exc.message


def test_retry_after_is_at_least_one_second():
    assert rate_limited_until(_declined(0.2), now=RESET).headers == {"Retry-After": "1"}


# ---------------------------------------------------------------------------
# Streams
# ---------------------------------------------------------------------------


def test_the_stream_message_names_the_reset_time_and_nothing_from_the_provider():
    message = safe_streaming_error_message(_wrapped(_declined()))
    assert message.startswith("The model route is rate limited.")
    assert "wait until 2026-08-26T21:14:05+00:00" in message
    assert PROVIDER_PROSE not in message
    assert "WITHHELD" not in message


def test_the_agent_stream_block_and_bridge_event_carry_the_same_message():
    wrapped = _wrapped(_declined())
    block = agent_stream_error_block(wrapped)
    assert block.startswith("\n\n---\n⚠️ **The model route is rate limited.**")
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
    live = AdvisedWaitExceedsRetryBudget(
        _Throttle(PROVIDER_PROSE), advised_seconds=6832.4, budget_seconds=840,
        retry_at=datetime.now(UTC) + timedelta(seconds=6832.4),
    )
    app, restore = _boot_app(_wrapped(live))
    try:
        response = _invoke(app)
    finally:
        restore()
    assert response.status_code == 429
    assert int(response.headers["Retry-After"]) in (6832, 6833)
    body = response.json()
    assert body["error"]["code"] == "rate_limited"
    assert live.retry_at.isoformat(timespec="seconds") in body["detail"]
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


def test_the_invoke_endpoint_logs_the_decline_at_its_own_logger():
    """The endpoint's logger does not always propagate to the root handler
    once the project's logging is configured, so the line is read at the
    logger itself."""
    from kestrel_sovereign.endpoints import agent as agent_endpoints

    live = AdvisedWaitExceedsRetryBudget(
        _Throttle(PROVIDER_PROSE), advised_seconds=300, budget_seconds=240,
        retry_at=datetime.now(UTC) + timedelta(seconds=300),
    )
    app, restore = _boot_app(_wrapped(live))
    try:
        with patch.object(agent_endpoints.logger, "error") as log_error:
            response = _invoke(app)
    finally:
        restore()
    assert response.status_code == 429
    rendered = [call.args[0] % tuple(call.args[1:]) for call in log_error.call_args_list]
    assert any(line.startswith("Agent invocation declined: model route rate limited until") for line in rendered)
    assert not any(PROVIDER_PROSE in line for line in rendered)


def test_chat_completions_answers_429_with_retry_after_when_the_route_declined_to_wait():
    """The OpenAI-compatible surface, whose clients honour Retry-After on a
    429, answered 500 for the same aggregate."""
    live = AdvisedWaitExceedsRetryBudget(
        _Throttle(PROVIDER_PROSE), advised_seconds=6832.4, budget_seconds=840,
        retry_at=datetime.now(UTC) + timedelta(seconds=6832.4),
    )
    app, restore = _boot_app(_wrapped(live))
    try:
        with patch.dict(os.environ, {"KESTREL_API_KEY": "test-key"}), TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"model": "any", "messages": [{"role": "user", "content": "hello"}]},
                headers={"X-API-Key": "test-key"},
            )
    finally:
        restore()
    assert response.status_code == 429, response.text
    assert int(response.headers["Retry-After"]) in (6832, 6833)
    body = response.json()
    assert body["error"]["code"] == "rate_limited"
    assert PROVIDER_PROSE not in response.text
