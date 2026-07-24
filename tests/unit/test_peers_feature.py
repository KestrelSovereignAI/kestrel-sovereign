"""Direct contracts for the Peers feature."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.features.peers.feature import (
    MAX_OUTBOUND_ARTIFACT_BYTES,
    PeersFeature,
    _discover_host_url,
)


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
    client.get.return_value = _mock_directory_response()

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
    client.get.return_value = _mock_directory_response()

    with patch("kestrel_sovereign.features.peers.feature.httpx.AsyncClient", return_value=client):
        result = await feature.ask_agent("claw", "hello")

    assert result.status is ToolResultStatus.OK
    assert result.data == {"agent": "claw", "response": "hi from claw"}


@pytest.mark.asyncio
async def test_local_host_routes_resolved_routing_name_not_display_name():
    """The local adapter resolves the public peer name before routing it."""
    agent = SimpleNamespace(_agent_name="emma")
    feature = PeersFeature(agent)
    feature._host_url = "http://multi_agent"
    feature._api_key = ""
    feature._own_name = "emma"

    directory = MagicMock(status_code=200)
    directory.raise_for_status.return_value = None
    directory.json.return_value = {
        "agents": [
            {"id": "emma", "name": "emma", "routing_name": "emma"},
            {
                "id": "did:test:claw",
                "name": "Claw Display Name",
                "routing_name": "claw-stable-route",
            },
        ],
    }
    response = MagicMock(status_code=200)
    response.raise_for_status.return_value = None
    response.json.return_value = {"response": "routed"}
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = False
    client.get.return_value = directory
    client.post.return_value = response

    with patch(
        "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
        return_value=client,
    ):
        result = await feature.ask_agent("Claw Display Name", "hello")

    assert result.status is ToolResultStatus.OK
    assert "/api/agents/claw-stable-route/api/agent/invoke" in client.post.call_args.args[0]
    assert "Claw%20Display%20Name" not in client.post.call_args.args[0]


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
    # This fixture models the intentional pre-ceremony compatibility tier.
    # MagicMock would otherwise synthesize a truthy ``identity.is_hybrid`` and
    # incorrectly present itself as a loaded hybrid signer (#2475).
    agent.identity = None
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


def _mock_directory_response():
    """Automatic directory used before each local peer route."""
    response = MagicMock(status_code=200)
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "agents": [
            {"id": "did:test:emma", "name": "emma", "routing_name": "emma"},
            {"id": "did:test:claw", "name": "claw", "routing_name": "claw"},
            {"id": "did:test:meridian", "name": "meridian", "routing_name": "meridian"},
            {"id": "did:test:nellie", "name": "nellie", "routing_name": "nellie"},
            {"id": "did:test:talon", "name": "talon", "routing_name": "talon"},
        ],
    }
    return response


def _async_client_with(post_resp=None, get_resp=None):
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = False
    if post_resp is not None:
        client.post.return_value = post_resp
    if get_resp is not None:
        client.get.side_effect = [_mock_directory_response(), get_resp]
    else:
        client.get.return_value = _mock_directory_response()
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


@pytest.mark.asyncio
async def test_send_a2a_task_attaches_sender_artifacts_on_wire():
    """Send-side artifact support (#1525): a sender can attach durable
    handoff payload to an outgoing task. The artifacts land on the wire
    payload with names, ordering, and structured metadata preserved."""
    feature = _make_a2a_feature()
    client = _async_client_with(post_resp=_mock_post_response(task_id="art1"))

    with patch(
        "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
        return_value=client,
    ):
        result = await feature.send_a2a_task(
            "meridian", "Please orchestrate this plan.",
            skill_id="workflow.assign",
            artifacts=[
                {"name": "plan", "text": "step one", "index": 0,
                 "last_chunk": True, "metadata": {"origin": "saved_item"}},
                {"name": "evidence", "data": {"diff": "+1 -1"}},
            ],
        )

    assert result.status is ToolResultStatus.OK
    assert result.data["artifacts_attached"] == 2
    posted_body = client.post.call_args.kwargs["json"]
    arts = posted_body["artifacts"]
    assert len(arts) == 2
    # Ordering + names + metadata preserved.
    assert arts[0]["name"] == "plan"
    assert arts[0]["parts"] == [{"type": "text", "text": "step one"}]
    assert arts[0]["index"] == 0
    assert arts[0]["lastChunk"] is True
    assert arts[0]["metadata"] == {"origin": "saved_item"}
    # Structured data part carries metadata, not just raw text.
    assert arts[1]["name"] == "evidence"
    assert arts[1]["parts"] == [{"type": "data", "data": {"diff": "+1 -1"}}]


@pytest.mark.asyncio
async def test_send_a2a_task_attaches_references_as_data_artifacts():
    """Durable references serialize into the ``references`` artifact
    group as structured DataParts with monotonic indices (#1525)."""
    feature = _make_a2a_feature()
    client = _async_client_with(post_resp=_mock_post_response(task_id="ref1"))

    with patch(
        "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
        return_value=client,
    ):
        result = await feature.send_a2a_task(
            "meridian", "Use these.",
            references=[
                {"ref_type": "memory", "id": "m1", "label": "plan"},
                {"ref_type": "recall", "id": "r2"},
            ],
        )

    assert result.status is ToolResultStatus.OK
    assert result.data["artifacts_attached"] == 2
    arts = client.post.call_args.kwargs["json"]["artifacts"]
    assert [a["name"] for a in arts] == ["references", "references"]
    assert [a["index"] for a in arts] == [0, 1]
    assert all(a["metadata"] == {"kind": "reference"} for a in arts)
    assert arts[0]["parts"][0] == {
        "type": "data",
        "data": {"ref_type": "memory", "id": "m1", "label": "plan"},
    }
    assert arts[1]["parts"][0]["data"] == {"ref_type": "recall", "id": "r2"}


@pytest.mark.asyncio
async def test_send_a2a_question_attaches_references_as_data_artifacts():
    """send_a2a_question preserves structured references as bounded
    DataPart artifacts, same as send_a2a_task."""
    feature = _make_a2a_feature()
    client = _async_client_with(post_resp=_mock_post_response(task_id="q-ref"))

    with patch(
        "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
        return_value=client,
    ):
        result = await feature.send_a2a_question(
            "meridian",
            "Use this context?",
            references={"ref_type": "memory", "id": "m1", "label": "brief"},
        )

    assert result.status is ToolResultStatus.OK
    arts = client.post.call_args.kwargs["json"]["artifacts"]
    assert len(arts) == 1
    assert arts[0]["name"] == "references"
    assert arts[0]["index"] == 0
    assert arts[0]["metadata"] == {"kind": "reference"}
    assert arts[0]["parts"][0] == {
        "type": "data",
        "data": {"ref_type": "memory", "id": "m1", "label": "brief"},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("method_name", ["send_a2a_task", "send_a2a_question"])
async def test_send_a2a_rejects_string_references_before_dispatch(method_name):
    """A JSON-serialized string is not a references list. It must fail
    before dispatch instead of becoming one artifact per character."""
    feature = _make_a2a_feature()
    client_factory = MagicMock(
        return_value=_async_client_with(post_resp=_mock_post_response())
    )

    with patch(
        "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
        client_factory,
    ):
        result = await getattr(feature, method_name)(
            "meridian",
            "use refs",
            references='[{"ref_type":"memory","id":"m1"}]',
        )

    assert result.status is ToolResultStatus.ERROR
    assert result.data["sent"] is False
    assert result.data["error_type"] == "invalid_a2a_references"
    assert result.data["error_code"] == "references_must_be_structured"
    client_factory.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("method_name", ["send_a2a_task", "send_a2a_question"])
async def test_send_a2a_rejects_oversized_references_before_dispatch(method_name):
    feature = _make_a2a_feature()
    client_factory = MagicMock(
        return_value=_async_client_with(post_resp=_mock_post_response())
    )
    huge_ref = {
        "ref_type": "memory",
        "id": "m-big",
        "body": "x" * MAX_OUTBOUND_ARTIFACT_BYTES,
    }

    with patch(
        "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
        client_factory,
    ):
        result = await getattr(feature, method_name)(
            "meridian",
            "use refs",
            references=[huge_ref],
        )

    assert result.status is ToolResultStatus.ERROR
    assert result.data["sent"] is False
    assert result.data["error_type"] == "invalid_a2a_references"
    assert result.data["error_code"] == "payload_too_large"
    client_factory.assert_not_called()


@pytest.mark.parametrize("tool_method,tool_name", [
    ("send_a2a_question", "send_a2a_question"),
    ("send_a2a_task", "send_a2a_task"),
])
def test_send_a2a_tool_schema_advertises_arrays_for_artifacts_and_references(
    tool_method, tool_name,
):
    """Codex round 2 P2: the LLM-facing tool schema for the A2A send
    surfaces MUST advertise ``artifacts`` and ``references`` as
    arrays-of-objects, not strings. Otherwise the LLM is told to pass
    a string, which the strict validator (#1624) now rejects, leaving
    references effectively unusable through the tool path.

    Triggered by the @tool decorator's schema generator mapping
    ``Union`` (the ``get_origin`` of ``Optional[...]``) to
    ``"string"`` via its type_map fallback. Annotation kept as
    ``List[Any]`` (without Optional) so the origin resolves to
    ``list`` → ``array``.
    """
    method = getattr(PeersFeature, tool_method)
    schema = getattr(method, "_tool_schema", None)
    assert schema is not None, "method missing @tool schema"
    params = {p.name: p for p in schema["parameters"]}
    for field in ("artifacts", "references"):
        assert field in params, f"{tool_name} schema missing {field} param"
        assert params[field].type == "array", (
            f"{tool_name}.{field} schema must advertise 'array', "
            f"got {params[field].type!r}"
        )


@pytest.mark.asyncio
async def test_send_a2a_rejects_nan_in_references_with_typed_error():
    """Codex round 1 P2: a NaN/Inf float passes default json.dumps but
    httpx encodes with allow_nan=False and would fail downstream as a
    generic send error. The validation encoder must match httpx so the
    typed invalid_a2a_* result fires instead."""
    feature = _make_a2a_feature()
    client_factory = MagicMock(
        return_value=_async_client_with(post_resp=_mock_post_response())
    )
    bad_ref = {"ref_type": "memory", "id": "m1", "score": float("nan")}

    with patch(
        "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
        client_factory,
    ):
        result = await feature.send_a2a_task(
            "meridian", "use refs", references=[bad_ref],
        )

    assert result.status is ToolResultStatus.ERROR
    assert result.data["sent"] is False
    assert result.data["error_type"] == "invalid_a2a_artifacts"
    assert result.data["error_code"] == "not_json_serializable"
    client_factory.assert_not_called()


@pytest.mark.asyncio
async def test_send_a2a_task_without_artifacts_omits_wire_key():
    """No artifacts/references → the ``artifacts`` key is absent from the
    payload so legacy recipients see an unchanged wire shape (#1525)."""
    feature = _make_a2a_feature()
    client = _async_client_with(post_resp=_mock_post_response(task_id="bare1"))

    with patch(
        "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
        return_value=client,
    ):
        result = await feature.send_a2a_task("meridian", "no payload")

    assert result.data["artifacts_attached"] == 0
    assert "artifacts" not in client.post.call_args.kwargs["json"]


@pytest.mark.asyncio
async def test_send_a2a_question_attaches_sender_artifacts_on_wire():
    """send_a2a_question also supports send-side artifacts so a question
    can carry the context the recipient needs to answer (#1525)."""
    feature = _make_a2a_feature()
    client = _async_client_with(post_resp=_mock_post_response(task_id="q-art"))

    with patch(
        "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
        return_value=client,
    ):
        result = await feature.send_a2a_question(
            "meridian", "Does this plan look right?",
            artifacts=[{"name": "plan", "text": "the plan body"}],
        )

    assert result.status is ToolResultStatus.OK
    arts = client.post.call_args.kwargs["json"]["artifacts"]
    assert arts[0]["name"] == "plan"
    assert arts[0]["parts"] == [{"type": "text", "text": "the plan body"}]


@pytest.mark.asyncio
async def test_get_peer_task_result_reassembles_artifacts_in_index_order():
    """When the receiver chunked a long reply into multiple Artifacts
    (because per-tool argument cap is 10K), ``get_peer_task_result``
    must reassemble them in ``index`` order — NOT in the order the
    underlying transport happened to return them. The receiver-side
    ``attach_artifact_to_a2a_task`` tool stamps ``index`` per segment
    for this reason."""
    feature = _make_a2a_feature()
    # Recipient's GET response: artifacts deliberately out of insertion
    # order to prove the sort is index-based.
    get_resp = MagicMock(status_code=200)
    get_resp.json.return_value = {
        "id": "t-chunked",
        "status": "completed",
        "message": "See attached artifacts (3 segments of reply_body).",
        "artifacts": [
            {"name": "reply_body", "index": 2, "lastChunk": True,
             "parts": [{"type": "text", "text": "third"}]},
            {"name": "reply_body", "index": 0, "lastChunk": False,
             "parts": [{"type": "text", "text": "first-"}]},
            {"name": "reply_body", "index": 1, "lastChunk": False,
             "parts": [{"type": "text", "text": "second-"}]},
        ],
    }
    client = _async_client_with(get_resp=get_resp)
    with patch(
        "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
        return_value=client,
    ):
        result = await feature.get_peer_task_result("meridian", "t-chunked")

    assert result.status is ToolResultStatus.OK
    assert result.data["artifact_segment_count"] == 3
    assert result.data["artifact_body"] == "first-second-third", (
        f"Artifact reassembly must walk index ascending. Got "
        f"{result.data['artifact_body']!r}."
    )
    assert result.data["artifact_body_complete"] is True, (
        "lastChunk=True on the index=2 segment means the body is "
        "complete; resumed turn can use it without waiting for more."
    )


@pytest.mark.asyncio
async def test_get_peer_task_result_flags_incomplete_chunked_body():
    """If the receiver is still mid-stream (no segment carries
    ``lastChunk=True``), ``artifact_body_complete`` must be False so
    the resumed turn doesn't act on a partial body."""
    feature = _make_a2a_feature()
    get_resp = MagicMock(status_code=200)
    get_resp.json.return_value = {
        "id": "t-partial",
        "status": "working",
        "artifacts": [
            {"name": "reply_body", "index": 0, "lastChunk": False,
             "parts": [{"type": "text", "text": "first-"}]},
            {"name": "reply_body", "index": 1, "lastChunk": False,
             "parts": [{"type": "text", "text": "second-"}]},
        ],
    }
    client = _async_client_with(get_resp=get_resp)
    with patch(
        "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
        return_value=client,
    ):
        result = await feature.get_peer_task_result("meridian", "t-partial")

    assert result.data["artifact_body"] == "first-second-"
    assert result.data["artifact_body_complete"] is False, (
        "Without any segment carrying lastChunk=True the body is "
        "partial — the resumed turn must know not to use it as final."
    )


@pytest.mark.asyncio
async def test_get_peer_task_result_falls_back_to_inline_message_when_no_artifacts():
    """Backwards-compat: short replies (no artifacts at all) must keep
    surfacing ``reply_text`` from the inline ``message`` field — the
    existing fire-and-resume contract for ≤10K replies is unchanged."""
    feature = _make_a2a_feature()
    get_resp = MagicMock(status_code=200)
    get_resp.json.return_value = {
        "id": "t-short",
        "status": "completed",
        "message": "Rome",
        "artifacts": [],
    }
    client = _async_client_with(get_resp=get_resp)
    with patch(
        "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
        return_value=client,
    ):
        result = await feature.get_peer_task_result("meridian", "t-short")

    assert result.data["reply_text"] == "Rome"
    assert result.data["artifact_body"] == ""
    assert result.data["artifact_segment_count"] == 0
    assert result.data["artifact_body_complete"] is True, (
        "No artifacts at all means the body IS complete via the inline "
        "message field — short-reply path must not get flagged partial."
    )


@pytest.mark.asyncio
async def test_get_peer_task_result_legacy_terminal_artifacts_complete():
    """Codex round 1 P2 on the artifact PR: a peer task in a TERMINAL
    state (completed/failed/canceled) whose artifacts predate the
    ``lastChunk`` chunking convention must still be flagged
    ``artifact_body_complete=True``. Otherwise the resumed turn
    would wait or refetch indefinitely on legacy peers that ship
    artifact text without the explicit terminal-segment marker."""
    feature = _make_a2a_feature()
    get_resp = MagicMock(status_code=200)
    get_resp.json.return_value = {
        "id": "t-legacy",
        "status": "completed",  # ← TERMINAL
        "message": "",  # answer lives in the artifact, not inline
        "artifacts": [
            # Legacy shape: no `lastChunk`, no `index` even — just the
            # answer text in parts[].
            {"name": "result", "parts": [{"type": "text", "text": "42"}]},
        ],
    }
    client = _async_client_with(get_resp=get_resp)
    with patch(
        "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
        return_value=client,
    ):
        result = await feature.get_peer_task_result("meridian", "t-legacy")

    assert result.data["artifact_body"] == "42"
    assert result.data["artifact_body_complete"] is True, (
        "Terminal task with legacy artifact (no lastChunk) must NOT "
        "flag incomplete — the resumed turn would otherwise loop "
        "waiting for chunks that will never arrive."
    )


@pytest.mark.asyncio
async def test_get_peer_task_result_multi_group_artifacts_isolate_reply_body():
    """Codex round 2 P2 on the artifact PR: when the peer attaches
    ``reply_body`` chunks AND a separate ``debug_log`` artifact,
    ``artifact_body`` must contain ONLY the reply_body group
    reassembled in index order — never the debug_log content.
    Otherwise the resumed turn would treat unrelated artifacts as
    part of the answer."""
    feature = _make_a2a_feature()
    get_resp = MagicMock(status_code=200)
    get_resp.json.return_value = {
        "id": "t-multi",
        "status": "completed",
        "message": "See attached artifacts (2 segments of reply_body).",
        "artifacts": [
            {"name": "debug_log", "index": 0, "lastChunk": True,
             "parts": [{"type": "text", "text": "<UNRELATED LOG OUTPUT>"}]},
            {"name": "reply_body", "index": 1, "lastChunk": True,
             "parts": [{"type": "text", "text": "second"}]},
            {"name": "reply_body", "index": 0, "lastChunk": False,
             "parts": [{"type": "text", "text": "first-"}]},
        ],
    }
    client = _async_client_with(get_resp=get_resp)
    with patch(
        "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
        return_value=client,
    ):
        result = await feature.get_peer_task_result("meridian", "t-multi")

    assert result.data["artifact_body"] == "first-second", (
        f"artifact_body must contain only the reply_body group "
        f"reassembled in index order, got {result.data['artifact_body']!r}. "
        f"Concatenating across groups would corrupt the answer."
    )
    assert "<UNRELATED LOG OUTPUT>" not in result.data["artifact_body"]
    # All groups are still surfaced individually for callers that need
    # the non-reply artifacts.
    assert result.data["artifact_bodies"] == {
        "debug_log": "<UNRELATED LOG OUTPUT>",
        "reply_body": "first-second",
    }
    assert result.data["artifact_group_complete"]["reply_body"] is True
    assert result.data["artifact_segment_count"] == 3, (
        "segment_count counts ALL artifact segments across groups, "
        "not just the primary reply group."
    )


@pytest.mark.asyncio
async def test_get_peer_task_result_legacy_single_unnamed_group_still_works():
    """Backwards-compat: a single artifact group with no recognized
    name (or empty name) still becomes the primary body — so peers
    that don't follow the ``reply_body`` convention yet aren't broken."""
    feature = _make_a2a_feature()
    get_resp = MagicMock(status_code=200)
    get_resp.json.return_value = {
        "id": "t-unnamed",
        "status": "completed",
        "message": "",
        "artifacts": [
            {"name": "result", "parts": [{"type": "text", "text": "the answer"}]},
        ],
    }
    client = _async_client_with(get_resp=get_resp)
    with patch(
        "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
        return_value=client,
    ):
        result = await feature.get_peer_task_result("meridian", "t-unnamed")

    assert result.data["artifact_body"] == "the answer"
    assert result.data["artifact_body_complete"] is True


# ---------------------------------------------------------------------------
# Startup-replay supervisors are FEATURE-owned so boot rollback / soft disable
# cancel them — they must not outlive the torn-down feature and later fire a
# resumption signal (kestrel-sovereign#2522 P1).
# ---------------------------------------------------------------------------


class _TrackingAgent:
    """Agent double with the REAL background-task lifecycle: tasks are actual
    ``asyncio.Task`` objects tracked in a global set that auto-discards on
    completion, exactly like ``KestrelAgent._track_background_task``. This lets
    the test prove a spawned supervisor is a live, cancellable task (a coroutine
    that a stub ``close()``s could never leak)."""

    def __init__(self, name="emma"):
        import asyncio

        self._agent_name = name
        self.did = f"did:test:{name}"
        self._background_tasks: set = set()
        self._asyncio = asyncio

    def _track_background_task(self, coro, *, name=""):
        task = self._asyncio.ensure_future(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task


def _within_deadline_waiting_row(task_id="t-replay", recipient="claw"):
    from datetime import datetime, timedelta, timezone

    deadline = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    return SimpleNamespace(
        task_id=task_id,
        recipient=recipient,
        original_question="still waiting?",
        origin_session_id="sess-1",
        deadline=deadline,
        retry_state=None,
    )


@pytest.mark.asyncio
async def test_replay_supervisor_is_feature_owned_and_cancelled_on_shutdown():
    """Boot startup-replay of a within-deadline WAITING row spawns the
    resubscription supervisor as a FEATURE-owned background task, so
    ``Feature.shutdown()`` (what boot rollback / soft disable call) cancels it —
    leaving NO task behind that could later dispatch a resumption signal against
    a torn-down feature (#2522 P1)."""
    import asyncio

    agent = _TrackingAgent()
    feature = PeersFeature(agent)
    feature._host_url = "http://multi_agent"
    feature._api_key = ""
    feature._own_name = "emma"

    # Replace the real SSE supervisor with one that blocks forever, so the
    # spawned task stays PENDING and we can prove teardown cancels it (the real
    # supervisor would make network calls and complete immediately, hiding the
    # leak the fix addresses). The REAL replay code path still runs and is what
    # decides to spawn — and how it tracks — the task.
    async def _never_returns(**kwargs):
        await asyncio.Event().wait()

    feature._supervise_a2a_question = _never_returns

    store = MagicMock()
    store.list_waiting = AsyncMock(
        return_value=[_within_deadline_waiting_row()]
    )

    await feature._replay_pending_a2a_questions(store)

    # The supervisor is tracked as a FEATURE-owned task (the crux of the fix —
    # before it, replay used the agent-global tracker and shutdown could not
    # reach it) AND lives in the agent's global reap set.
    owned = list(feature._owned_background_tasks)
    assert len(owned) == 1
    supervisor = owned[0]
    assert not supervisor.done()
    assert supervisor in agent._background_tasks

    # Boot rollback / soft disable both call Feature.shutdown().
    await feature.shutdown()

    assert supervisor.done()
    assert supervisor.cancelled()
    assert feature._owned_background_tasks == []
    # Dropped from the agent's global set too (no dangling reference).
    assert supervisor not in agent._background_tasks


@pytest.mark.asyncio
async def test_replay_past_deadline_row_spawns_no_supervisor():
    """A past-deadline WAITING row is expired in place (synthetic signal), never
    resubscribed — so no supervisor task is owned. Guards the replay branch that
    feeds the ownership fix so a regression can't silently spawn nothing."""
    from datetime import datetime, timedelta, timezone

    agent = _TrackingAgent()
    feature = PeersFeature(agent)
    feature._host_url = "http://multi_agent"
    feature._api_key = ""
    feature._own_name = "emma"
    feature._handle_expired_row = AsyncMock(return_value=None)

    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    row = SimpleNamespace(
        task_id="t-old",
        recipient="claw",
        original_question="q",
        origin_session_id="s",
        deadline=past,
        retry_state=None,
    )
    store = MagicMock()
    store.list_waiting = AsyncMock(return_value=[row])

    await feature._replay_pending_a2a_questions(store)

    feature._handle_expired_row.assert_awaited_once()
    assert getattr(feature, "_owned_background_tasks", []) == []
    assert agent._background_tasks == set()


# ---------------------------------------------------------------------------
# The LIVE ``send_a2a_question`` supervisor and the restored-payload retry loop
# are the two sibling long-lived Peers coroutines that must share the startup-
# replay supervisor's ownership: both are agent-tracked for full-shutdown reap
# but ALSO feature-owned, so runtime disable / boot rollback / soft disable
# cancel them via ``Feature.shutdown()``. An agent-only task would keep running
# (and could still fire a resumption signal) against a torn-down feature
# (kestrel-sovereign#2522 P1).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_question_supervisor_is_feature_owned_and_cancelled_on_shutdown():
    """The live ``send_a2a_question`` POST spawns its SSE resumption supervisor
    as a FEATURE-owned background task, so ``Feature.shutdown()`` (what boot
    rollback / soft disable call) cancels it — no supervisor outlives the
    torn-down feature to later dispatch a resumption signal (#2522 P1). This is
    the direct-ask analogue of the startup-replay supervisor guarantee."""
    import asyncio

    feature = _make_a2a_feature()

    # Swap the fixture's coro-closing stub for the REAL background-task
    # lifecycle (actual ``asyncio.Task`` in a self-discarding global set, like
    # ``KestrelAgent._track_background_task``) so we can prove the supervisor is
    # a live, cancellable task — a coroutine a stub ``close()``s could never leak.
    tracked: set = set()

    def _track_bg(coro, *, name=""):
        task = asyncio.ensure_future(coro)
        tracked.add(task)
        task.add_done_callback(tracked.discard)
        return task
    feature.agent._track_background_task = _track_bg

    # Block the supervisor forever so the spawned task stays PENDING and teardown
    # must cancel it (the real SSE supervisor would make network calls and finish
    # immediately, hiding the leak the fix addresses). The REAL send path still
    # runs and is what decides to spawn — and how it tracks — the task.
    async def _never_returns(**kwargs):
        await asyncio.Event().wait()
    feature._supervise_a2a_question = _never_returns

    post_resp = _mock_post_response(task_id="q-live", state="submitted")
    client = _async_client_with(post_resp=post_resp)
    with patch(
        "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
        return_value=client,
    ):
        result = await feature.send_a2a_question("meridian", "still there?")

    assert result.status is ToolResultStatus.OK
    assert result.data["awaiting_reply"] is True

    # The supervisor is tracked as a FEATURE-owned task (the crux of the fix —
    # before it, the live path used the agent-global tracker and shutdown could
    # not reach it) AND lives in the agent's global reap set.
    owned = list(feature._owned_background_tasks)
    assert len(owned) == 1
    supervisor = owned[0]
    assert not supervisor.done()
    assert supervisor in tracked

    # Boot rollback / soft disable both call Feature.shutdown().
    await feature.shutdown()

    assert supervisor.done()
    assert supervisor.cancelled()
    assert feature._owned_background_tasks == []
    # Dropped from the agent's global set too (no dangling reference).
    assert supervisor not in tracked


@pytest.mark.asyncio
async def test_question_answered_retry_is_feature_owned_and_cancelled_on_shutdown():
    """When a restored WAITING row's wakeup signal fails to enqueue, the
    near-term retry loop is scheduled as a FEATURE-owned background task. That
    loop is reachable during boot startup-replay, so ``Feature.shutdown()``
    (boot rollback / soft disable) must cancel it rather than leave it retrying
    — and eventually firing a resumption signal — against a torn-down feature
    (#2522 P1). Driven through the real restore path that boot replay hits."""
    import asyncio

    agent = _TrackingAgent()
    feature = PeersFeature(agent)
    feature._host_url = "http://multi_agent"
    feature._api_key = ""
    feature._own_name = "emma"

    # Block the retry loop forever so the spawned task stays PENDING and teardown
    # must cancel it (the real loop sleeps, refires, then completes — which would
    # hide the leak). The REAL restore→schedule code path still runs.
    async def _never_returns(**kwargs):
        await asyncio.Event().wait()
    feature._retry_restored_question_answered_signal = _never_returns

    # Real restore path: mark_waiting_for_retry succeeds, so
    # ``_restore_pending_question_waiting`` schedules the retry — exactly what
    # boot replay reaches when a wakeup-signal enqueue fails.
    store = MagicMock()
    store.mark_waiting_for_retry = AsyncMock(return_value=True)
    agent.pending_a2a_questions = store

    await feature._restore_pending_question_waiting(
        "t-retry",
        state="completed",
        reply_text="the answer",
        recipient="claw",
        original_question="still waiting?",
        sess_id="sess-1",
        causation_chain=None,
        schedule_retry=True,
    )

    store.mark_waiting_for_retry.assert_awaited_once()

    # The retry loop is a live FEATURE-owned task (before the fix it used the
    # agent-global tracker — or, worse, an untracked ``create_task`` fallback —
    # neither of which ``Feature.shutdown()`` could reach) AND is agent-tracked.
    owned = list(feature._owned_background_tasks)
    assert len(owned) == 1
    retry_task = owned[0]
    assert not retry_task.done()
    assert retry_task in agent._background_tasks

    # Boot rollback / soft disable both call Feature.shutdown().
    await feature.shutdown()

    assert retry_task.done()
    assert retry_task.cancelled()
    assert feature._owned_background_tasks == []
    # Dropped from the agent's global set too (no dangling reference).
    assert retry_task not in agent._background_tasks
