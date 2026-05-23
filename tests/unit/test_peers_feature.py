"""Direct contracts for the Peers feature."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.features.peers.feature import PeersFeature, _discover_host_url


def test_discover_host_url_from_env(monkeypatch):
    monkeypatch.setenv("KESTREL_HOST_URL", "http://localhost:9999/")

    assert _discover_host_url() == "http://localhost:9999"


@pytest.mark.asyncio
async def test_list_peers_filters_out_self():
    agent = SimpleNamespace(_agent_name="emma")
    feature = PeersFeature(agent)
    feature._host_url = "http://multi_agent"
    feature._api_key = ""
    feature._own_name = "emma"

    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = [
        {"name": "emma", "status": "online"},
        {"name": "claw", "status": "online", "description": "peer"},
    ]

    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = False
    client.get.return_value = response

    with patch("kestrel_sovereign.features.peers.feature.httpx.AsyncClient", return_value=client):
        result = await feature.list_peers()

    assert result.status is ToolResultStatus.OK
    assert result.data["self"] == "emma"
    assert result.data["peers"] == [{"name": "claw", "status": "online", "description": "peer"}]


@pytest.mark.asyncio
async def test_ask_agent_rejects_self_target():
    agent = SimpleNamespace(_agent_name="emma")
    feature = PeersFeature(agent)
    feature._host_url = "http://multi_agent"
    feature._api_key = ""
    feature._own_name = "emma"

    result = await feature.ask_agent("emma", "hello")

    assert result.status is ToolResultStatus.ERROR
    assert "yourself" in result.error


@pytest.mark.asyncio
async def test_ask_agent_reports_offline_peer():
    agent = SimpleNamespace(_agent_name="emma")
    feature = PeersFeature(agent)
    feature._host_url = "http://multi_agent"
    feature._api_key = ""
    feature._own_name = "emma"

    response = MagicMock(status_code=503)
    response.raise_for_status.return_value = None

    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = False
    client.post.return_value = response

    with patch("kestrel_sovereign.features.peers.feature.httpx.AsyncClient", return_value=client):
        result = await feature.ask_agent("claw", "hello")

    assert result.status is ToolResultStatus.ERROR
    assert "offline" in result.error


@pytest.mark.asyncio
async def test_ask_agent_returns_peer_response():
    agent = SimpleNamespace(_agent_name="emma")
    feature = PeersFeature(agent)
    feature._host_url = "http://multi_agent"
    feature._api_key = "key"
    feature._own_name = "emma"

    response = MagicMock(status_code=200)
    response.raise_for_status.return_value = None
    response.json.return_value = {"response": "hi from claw"}

    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = False
    client.post.return_value = response

    with patch("kestrel_sovereign.features.peers.feature.httpx.AsyncClient", return_value=client):
        result = await feature.ask_agent("claw", "hello")

    assert result.status is ToolResultStatus.OK
    assert result.data == {"agent": "claw", "response": "hi from claw"}


def _make_a2a_feature(name="emma"):
    agent = SimpleNamespace(_agent_name=name)
    feature = PeersFeature(agent)
    feature._host_url = "http://multi_agent"
    feature._api_key = ""
    feature._own_name = name
    return feature


def _mock_post_response(task_id="t1", session_id="s1", state="submitted"):
    response = MagicMock(status_code=200)
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "id": task_id,
        "sessionId": session_id,
        "status": {"state": state},
    }
    return response


def _async_client_with(post_resp=None, get_resp=None):
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = False
    if post_resp is not None:
        client.post.return_value = post_resp
    if get_resp is not None:
        client.get.return_value = get_resp
    return client


@pytest.mark.asyncio
async def test_send_a2a_message_fire_and_forget():
    """``send_a2a_message`` POSTs to /tasks/send with NO skill_id and
    returns immediately. Contrast with send_a2a_task (skill_id set,
    same return shape) and send_a2a_question (sync wait)."""
    feature = _make_a2a_feature()
    client = _async_client_with(post_resp=_mock_post_response(task_id="m1"))
    with patch(
        "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
        return_value=client,
    ):
        result = await feature.send_a2a_message("meridian", "FYI I shipped PR 42")

    assert result.status is ToolResultStatus.OK
    assert result.data["sent"] is True
    assert result.data["task_id"] == "m1"
    assert result.data["recipient"] == "meridian"
    # NO skill_id was attached on the wire.
    posted_body = client.post.call_args.kwargs["json"]
    assert "skill" not in posted_body["metadata"], (
        "send_a2a_message must NOT set skill_id — that signals "
        "informational, not work-assignment"
    )
    assert posted_body["metadata"]["sender"] == "emma"


@pytest.mark.asyncio
async def test_send_a2a_message_rejects_self_target():
    feature = _make_a2a_feature("emma")
    result = await feature.send_a2a_message("emma", "test")
    assert result.status is ToolResultStatus.ERROR
    assert "yourself" in result.error.lower()


@pytest.mark.asyncio
async def test_send_a2a_question_waits_for_terminal_state_and_returns_answer():
    """``send_a2a_question`` polls /tasks/{id} until terminal, then
    returns the answer text from status.message.parts. The POST shape
    matches send_a2a_message (no skill_id); the difference is the
    sync-wait + answer-extraction on the caller side."""
    feature = _make_a2a_feature()

    # First the POST that creates the task.
    post_resp = _mock_post_response(task_id="q1", state="submitted")
    # Then polling: first returns still-working, second returns
    # completed with an answer.
    get_working = MagicMock(status_code=200)
    get_working.json.return_value = {
        "id": "q1", "sessionId": "s1",
        "status": {"state": "working"},
    }
    get_done = MagicMock(status_code=200)
    get_done.json.return_value = {
        "id": "q1", "sessionId": "s1",
        "status": {
            "state": "completed",
            "message": {
                "role": "agent",
                "parts": [{"type": "text", "text": "yes, three open PRs"}],
            },
        },
    }

    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = False
    client.post.return_value = post_resp
    client.get = AsyncMock(side_effect=[get_working, get_done])

    with patch(
        "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
        return_value=client,
    ):
        result = await feature.send_a2a_question(
            "meridian", "any open PRs?", timeout_seconds=10,
        )

    assert result.status is ToolResultStatus.OK
    assert result.data["answered"] is True
    assert result.data["answer"] == "yes, three open PRs"
    assert result.data["state"] == "completed"


@pytest.mark.asyncio
async def test_send_a2a_question_returns_failed_on_canceled_state():
    """A task that terminates as FAILED or CANCELED returns
    ToolResult.failed (not ok) — the caller's calling cognition
    turn should know the answer is unreliable, not just shaped
    differently."""
    feature = _make_a2a_feature()
    post_resp = _mock_post_response(task_id="q2", state="submitted")
    get_failed = MagicMock(status_code=200)
    get_failed.json.return_value = {
        "id": "q2",
        "status": {
            "state": "failed",
            "message": {
                "role": "agent",
                "parts": [{"type": "text", "text": "permission denied"}],
            },
        },
    }
    client = _async_client_with(post_resp=post_resp, get_resp=get_failed)

    with patch(
        "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
        return_value=client,
    ):
        result = await feature.send_a2a_question(
            "meridian", "do X", timeout_seconds=5,
        )

    assert result.status is ToolResultStatus.ERROR
    assert result.data["state"] == "failed"
    assert "permission denied" in result.data["answer"]


@pytest.mark.asyncio
async def test_send_a2a_question_partial_on_timeout():
    """When the recipient doesn't reach terminal state in the
    allotted time, return ToolResult.partial — the task is still
    live (caller can poll task_id manually), but our sync wait is up."""
    feature = _make_a2a_feature()
    post_resp = _mock_post_response(task_id="q3", state="submitted")
    get_still_working = MagicMock(status_code=200)
    get_still_working.json.return_value = {
        "id": "q3", "status": {"state": "working"},
    }
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = False
    client.post.return_value = post_resp
    client.get = AsyncMock(return_value=get_still_working)

    with patch(
        "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
        return_value=client,
    ):
        # 1-second timeout; first poll interval is 0.5s so at least
        # one GET happens before the deadline.
        result = await feature.send_a2a_question(
            "meridian", "what's the answer?", timeout_seconds=1,
        )

    assert result.status is ToolResultStatus.PARTIAL
    assert result.data["state"] == "timeout"
    assert result.data["task_id"] == "q3"


@pytest.mark.asyncio
async def test_send_a2a_task_carries_skill_id_and_returns_task_id():
    """``send_a2a_task`` is the work-delegation verb — skill_id IS
    attached to metadata (distinguishing it from send_a2a_message).
    Returns the task_id for caller-side tracking."""
    feature = _make_a2a_feature()
    client = _async_client_with(post_resp=_mock_post_response(task_id="w1"))

    with patch(
        "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
        return_value=client,
    ):
        result = await feature.send_a2a_task(
            "talon", "claim issue 42",
            skill_id="workflow.assign",
        )

    assert result.status is ToolResultStatus.OK
    assert result.data["task_id"] == "w1"
    posted_body = client.post.call_args.kwargs["json"]
    assert posted_body["metadata"]["skill"] == "workflow.assign", (
        "send_a2a_task MUST carry skill_id on the wire — that's the "
        "workflow-routing hook the receiver uses to pick the handler"
    )
