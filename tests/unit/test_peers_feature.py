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
    """Build a PeersFeature against a fake agent that exposes the
    minimum surface ``send_a2a_question`` (#1444) and the legacy verbs
    rely on. ``pending_a2a_questions`` and ``dispatcher`` are wired
    with AsyncMocks so the fire-and-resume path runs end-to-end
    without a real DB or dispatcher hop. ``_track_background_task``
    runs coroutines synchronously to keep tests deterministic — the
    happy path doesn't actually need to wait on the SSE supervisor."""
    agent = MagicMock()
    agent._agent_name = name
    agent.did = f"did:test:{name}"
    agent._provide_causation_chain = MagicMock(return_value=None)
    agent._get_current_turn_id = MagicMock(return_value=None)
    agent.pending_a2a_questions = MagicMock()
    agent.pending_a2a_questions.insert = AsyncMock(return_value=None)
    agent.pending_a2a_questions.mark_resolved = AsyncMock(return_value=True)
    agent.dispatcher = MagicMock()
    agent.dispatcher.enqueue_signal = AsyncMock()

    def _track_bg(coro, *, name=""):
        # Close the coroutine immediately — tests asserting on POST-
        # time behavior don't want the SSE loop firing. Tests that DO
        # care about the supervisor run it directly via
        # _supervise_a2a_question (see test_send_a2a_question_supervisor).
        coro.close()
        return MagicMock()
    agent._track_background_task = _track_bg

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
async def test_send_a2a_question_returns_awaiting_reply_after_post():
    """Under the #1444 fire-and-resume contract, ``send_a2a_question``
    returns IMMEDIATELY after the POST with ``awaiting_reply=True``
    and ``resume_via='a2a.question_answered'``. The wait happens on
    the dispatcher's signal rail, not in this tool call."""
    feature = _make_a2a_feature()
    post_resp = _mock_post_response(task_id="q1", state="submitted")
    client = _async_client_with(post_resp=post_resp)

    with patch(
        "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
        return_value=client,
    ):
        result = await feature.send_a2a_question(
            "meridian", "any open PRs?", timeout_seconds=300,
        )

    assert result.status is ToolResultStatus.OK
    assert result.data["sent"] is True
    assert result.data["awaiting_reply"] is True
    assert result.data["task_id"] == "q1"
    assert result.data["recipient"] == "meridian"
    assert result.data["resume_via"] == "a2a.question_answered"
    assert "expires_at" in result.data


@pytest.mark.asyncio
async def test_send_a2a_question_records_pending_row_and_spawns_supervisor():
    """The fire-and-resume contract has three side effects at POST
    time: insert pending_a2a_questions row, spawn supervisor task,
    return immediately. All three must happen — without the pending
    row, the startup-replay sweep cannot recover from a crash; without
    the supervisor, the answered signal never fires."""
    feature = _make_a2a_feature()
    post_resp = _mock_post_response(task_id="q-spawn", state="submitted")
    client = _async_client_with(post_resp=post_resp)

    spawned = []

    def _track_bg(coro, *, name=""):
        spawned.append((coro, name))
        coro.close()
        return MagicMock()
    feature.agent._track_background_task = _track_bg

    with patch(
        "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
        return_value=client,
    ):
        result = await feature.send_a2a_question(
            "meridian", "the question?",
        )

    assert result.status is ToolResultStatus.OK
    # Pending row insert.
    feature.agent.pending_a2a_questions.insert.assert_awaited_once()
    insert_kwargs = (
        feature.agent.pending_a2a_questions.insert.await_args.kwargs
    )
    assert insert_kwargs["task_id"] == "q-spawn"
    assert insert_kwargs["recipient"] == "meridian"
    assert insert_kwargs["original_question"] == "the question?"
    # Supervisor spawn — task name must let `kill -SIGUSR1` ps grep
    # find it during ops.
    assert len(spawned) == 1
    _, spawn_name = spawned[0]
    assert "a2a_question_supervisor" in spawn_name
    assert "meridian" in spawn_name
    assert "q-spawn" in spawn_name


@pytest.mark.asyncio
async def test_send_a2a_question_fails_when_pending_store_insert_fails():
    """Codex round 3 P2d on PR #1453: if the pending-questions store
    rejects the correlation row, the tool must NOT return
    ``awaiting_reply=True`` — without the row, the supervisor's
    mark_resolved returns False and silently drops the answered
    signal as a duplicate, so the asking lineage never resumes
    despite the receiver answering. Surface as failure so the
    caller knows fire-and-resume is broken."""
    feature = _make_a2a_feature()
    post_resp = _mock_post_response(task_id="q-insert-fails", state="submitted")
    client = _async_client_with(post_resp=post_resp)
    feature.agent.pending_a2a_questions.insert = AsyncMock(
        side_effect=RuntimeError("disk full"),
    )

    with patch(
        "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
        return_value=client,
    ):
        result = await feature.send_a2a_question(
            "meridian", "the question?",
        )

    assert result.status is ToolResultStatus.ERROR, (
        f"Insert failure must yield ToolResult.failed, not ok. Got "
        f"{result.status}: {result.error}"
    )
    assert result.data["sent"] is True, (
        "The task WAS POSTed — the caller should know that."
    )
    assert result.data["awaiting_reply"] is False, (
        "Without a pending row the supervisor's mark_resolved returns "
        "False and the resumption signal gets dropped — awaiting_reply "
        "must NOT be True or the agent will end its turn waiting for a "
        "signal that will never fire."
    )
    assert "get_peer_task_result" in (result.error or ""), (
        "Error should tell the caller the recovery path is "
        "get_peer_task_result so they can fetch the answer manually."
    )


@pytest.mark.asyncio
async def test_send_a2a_question_stamps_a2a_verb_metadata():
    """Receiver-side verb discrimination still must not depend solely
    on skill_id / reply_expected — the explicit ``a2a_verb='question'``
    tag survives the fire-and-resume refactor so receiver prompts can
    still branch on it (codex P2 on PR #1380)."""
    feature = _make_a2a_feature()
    post_resp = _mock_post_response(task_id="qV1", state="submitted")
    client = _async_client_with(post_resp=post_resp)

    with patch(
        "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
        return_value=client,
    ):
        await feature.send_a2a_question("meridian", "ping")

    posted_body = client.post.call_args.kwargs["json"]
    assert posted_body["metadata"]["a2a_verb"] == "question"
    assert posted_body["metadata"]["reply_expected"] is True


@pytest.mark.asyncio
async def test_send_a2a_message_stamps_a2a_verb_metadata():
    feature = _make_a2a_feature()
    client = _async_client_with(post_resp=_mock_post_response(task_id="mV1"))
    with patch(
        "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
        return_value=client,
    ):
        await feature.send_a2a_message("meridian", "FYI shipped")
    posted_body = client.post.call_args.kwargs["json"]
    assert posted_body["metadata"]["a2a_verb"] == "message"


@pytest.mark.asyncio
async def test_send_a2a_task_stamps_a2a_verb_metadata():
    feature = _make_a2a_feature()
    client = _async_client_with(post_resp=_mock_post_response(task_id="tV1"))
    with patch(
        "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
        return_value=client,
    ):
        await feature.send_a2a_task(
            "talon", "claim issue 42",
            skill_id="workflow.assign",
        )
    posted_body = client.post.call_args.kwargs["json"]
    assert posted_body["metadata"]["a2a_verb"] == "task"
    assert posted_body["metadata"]["skill"] == "workflow.assign"


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
