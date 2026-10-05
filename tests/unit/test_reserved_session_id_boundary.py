"""A request cannot claim a reserved non-interactive session id (#3284).

``NON_INTERACTIVE_SESSION_IDS`` tells the security hook that no human is
attached, so an ASK-gated tool in such a session is refused instead of queued.
``/api/agent/invoke`` and ``/api/agent/stream`` took ``session_id`` straight
from the body, so a chat caller who sent ``"scheduler"`` or
``"feature-lifecycle"`` had every ASK-gated tool in the turn refused with "no
interactive approver" while a human sat at the console, and the turn's history
was filed under the reserved label.

Both doors now refuse such a value before the turn is registered. The cases
iterate the hook's own set, so a future reserved id is covered here the moment
it is added there.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kestrel_sovereign.features.security.hooks import (
    FEATURE_LIFECYCLE_SESSION_ID,
    NON_INTERACTIVE_SESSION_IDS,
    reserved_session_id,
)

pytestmark = pytest.mark.usefixtures("isolated_process_rate_limiter")

RESERVED = sorted(NON_INTERACTIVE_SESSION_IDS)
PADDED = [f"  {value}\t" for value in RESERVED]
ORDINARY = ["8f14e45f-ceea-467f-a0e6-5a1a2b3c4d5e", "scheduler2", "my-feature-lifecycle"]
# Each case: the value as sent, and the reserved id the refusal must name.
REFUSED = [*zip(RESERVED, RESERVED), *zip(PADDED, RESERVED)]


def _agent() -> MagicMock:
    async def stream(*_args, **_kwargs):
        yield "streamed reply"

    agent = MagicMock()
    agent.process_input = AsyncMock(return_value="invoked reply")
    agent.process_input_streaming = MagicMock(side_effect=stream)
    agent.register_active_request = MagicMock()
    agent._cleanup_cancelled_request = MagicMock()
    agent.is_request_cancelled = MagicMock(return_value=False)
    agent._conversation_response_identity = MagicMock(
        return_value={"model": "m", "provider": "p"}
    )
    agent.storage.resolve_session_id = AsyncMock(side_effect=lambda value: value)
    return agent


def _client(agent) -> TestClient:
    from kestrel_sovereign.api_errors import register_api_error_handlers
    from kestrel_sovereign.endpoints.agent import router
    from kestrel_sovereign.rate_limit import limiter

    app = FastAPI()
    app.state.limiter = limiter
    app.state.agent = agent
    app.include_router(router)
    register_api_error_handlers(app)
    return TestClient(app)


def _assert_refused_before_the_turn(response, agent, reserved: str) -> None:
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["code"] == "reserved_session_id"
    assert f"'{reserved}'" in error["message"]
    # Refused at the boundary: no request registered, no session resolved,
    # no turn started, so nothing was filed under the reserved label.
    agent.register_active_request.assert_not_called()
    agent._cleanup_cancelled_request.assert_not_called()
    agent.storage.resolve_session_id.assert_not_awaited()
    agent.process_input.assert_not_awaited()
    agent.process_input_streaming.assert_not_called()


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------


def test_the_feature_lifecycle_id_is_one_of_the_cases():
    """#3280 added the more human-plausible id; the doors must refuse it too."""
    assert FEATURE_LIFECYCLE_SESSION_ID in RESERVED
    assert "scheduler" in RESERVED


@pytest.mark.parametrize("value", RESERVED)
def test_a_reserved_id_names_itself(value):
    assert reserved_session_id(value) == value


@pytest.mark.parametrize("value, expected", list(zip(PADDED, RESERVED)))
def test_surrounding_whitespace_does_not_hide_a_reserved_id(value, expected):
    """The turn lifecycle strips the session it binds, so a padded value would
    reach the turn's features as the bare reserved id."""
    assert reserved_session_id(value) == expected


@pytest.mark.parametrize("value", [*ORDINARY, None, "", 1314, ["scheduler"]])
def test_anything_else_is_not_reserved(value):
    assert reserved_session_id(value) is None


# ---------------------------------------------------------------------------
# POST /api/agent/invoke
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value, reserved", REFUSED)
def test_invoke_refuses_a_reserved_session_id(value, reserved):
    agent = _agent()

    response = _client(agent).post(
        "/api/agent/invoke",
        json={"input": "list my files", "session_id": value},
    )

    _assert_refused_before_the_turn(response, agent, reserved)


@pytest.mark.parametrize("value", ORDINARY)
def test_invoke_still_runs_an_ordinary_session_id(value):
    agent = _agent()

    response = _client(agent).post(
        "/api/agent/invoke",
        json={"input": "list my files", "session_id": value},
    )

    assert response.status_code == 200, response.text
    assert response.json()["session_id"] == value
    agent.process_input.assert_awaited_once()
    assert agent.process_input.await_args.kwargs["session_id"] == value


def test_invoke_without_a_session_id_is_unaffected():
    agent = _agent()

    response = _client(agent).post("/api/agent/invoke", json={"input": "hello"})

    assert response.status_code == 200, response.text
    agent.process_input.assert_awaited_once()


# ---------------------------------------------------------------------------
# POST /api/agent/stream
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value, reserved", REFUSED)
def test_stream_refuses_a_reserved_session_id(value, reserved):
    agent = _agent()

    response = _client(agent).post(
        "/api/agent/stream",
        json={"input": "list my files", "session_id": value},
    )

    _assert_refused_before_the_turn(response, agent, reserved)
    assert "X-Session-Id" not in response.headers


@pytest.mark.parametrize("value", ORDINARY)
def test_stream_still_runs_an_ordinary_session_id(value):
    agent = _agent()

    response = _client(agent).post(
        "/api/agent/stream",
        json={"input": "list my files", "session_id": value},
    )

    assert response.status_code == 200, response.text
    assert response.text == "streamed reply"
    assert response.headers["X-Session-Id"] == value
    agent.process_input_streaming.assert_called_once()
    assert agent.process_input_streaming.call_args.kwargs["session_id"] == value
