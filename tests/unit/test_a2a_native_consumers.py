"""Task tools, commands and pre-turn context consume actual native storage."""

from types import SimpleNamespace

import pytest
import pytest_asyncio
from a2a.helpers.proto_helpers import new_data_artifact, new_text_message
from a2a.types import Part, Role, SendMessageRequest, TaskState
from kestrel_sdk.tools.result import ToolResult, ToolResultStatus

from kestrel_sovereign.a2a.lifecycle import task_state_from_label
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.agent.preturn_state import _a2a_inbox_section
from kestrel_sovereign.command_handler import CommandHandler
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.tasks.feature import TaskFeature
from tests.utils.native_a2a import manager_for


@pytest_asyncio.fixture
async def inbox(native_a2a_backend_factory):
    backend = native_a2a_backend_factory()
    await backend.connect()
    scope = NativeRecipientScope("did:kestrel:recipient", "tenant")
    manager = manager_for(backend, scope)
    agent = SimpleNamespace(did=scope.recipient_id, task_manager=manager)
    feature = TaskFeature(agent)
    feature.set_task_manager(manager)
    try:
        await manager.initialize()
        yield manager, feature, agent
    finally:
        await manager.close()


async def admit(manager):
    message = new_text_message("first request part", role=Role.ROLE_USER)
    message.parts.append(Part(text="second request part"))
    return await manager.create_task(
        SendMessageRequest(
            tenant="tenant",
            message=message,
            metadata={
                "sender": "forged-sender",
                "sender_verified": True,
                "agent_id": "forged-agent",
                "skill": "peer-request",
                "task_type": "peer",
            },
        ),
        creator_agent_id="did:kestrel:verified-sender",
    )


async def test_status_commands_and_preturn_share_native_owner_and_text(inbox):
    manager, feature, agent = inbox
    task = await admit(manager)
    result = await feature.check_task_status(task.id)
    assert result.status is ToolResultStatus.OK
    assert result.data["status"] == "submitted"
    assert result.data["request_content"] == "first request part\nsecond request part"
    assert result.data["message"] is None
    assert result.data["sender"] == "did:kestrel:verified-sender"
    assert not task.status.HasField("timestamp")
    assert result.data["status_timestamp"] is None
    assert "created_at" not in result.data
    listing = await feature.list_my_tasks(status="submitted", task_type="peer")
    assert listing.status is ToolResultStatus.OK
    assert listing.data["count"] == 1
    assert listing.data["tasks"][0]["sender"] == "did:kestrel:verified-sender"
    command = await CommandHandler(agent)._cmd_tasks("!tasks submitted")
    assert "submitted" in command and task.id[:8] in command
    assert "did:kestrel:verified-sender" in command and "forged" not in command
    snapshot = await _a2a_inbox_section(agent)
    assert "did:kestrel:verified-sender" in snapshot and "forged" not in snapshot
    assert "first request part second request part" in snapshot


@pytest.mark.parametrize(
    "label",
    ["working", "input-required", "auth-required", "completed", "failed", "rejected"],
)
async def test_all_native_states_filter_and_render_without_private_enum_values(
    inbox, label
):
    manager, feature, agent = inbox
    task = await admit(manager)
    await manager.update_status(
        task.id, task_state_from_label(label), recipient_agent_id=agent.did
    )
    listing = await feature.list_my_tasks(status=label)
    assert listing.status is ToolResultStatus.OK
    assert [row["status"] for row in listing.data["tasks"]] == [label]
    command = await CommandHandler(agent)._cmd_tasks("!tasks " + label)
    assert task.id[:8] in command and label in command


@pytest.mark.parametrize("label", ["completed", "failed", "rejected"])
async def test_responder_commits_one_native_transition_and_refuses_second_terminal_reply(
    inbox, label
):
    manager, feature, agent = inbox
    task = await admit(manager)
    result = await feature.respond_to_a2a_task(task.id, "real reply", state=label)
    assert result.status is ToolResultStatus.OK
    assert result.data["state"] == label
    stored = await manager.get_task_snapshot(task.id, recipient_agent_id=agent.did)
    assert stored.lifecycle_revision == 1
    assert stored.task.status.message.message_id
    assert stored.task.status.message.role == Role.ROLE_AGENT
    assert stored.task.status.message.task_id == task.id
    assert stored.task.status.message.context_id == task.context_id
    assert stored.task.status.message.parts[0].text == "real reply"
    again = await feature.respond_to_a2a_task(task.id, "replacement", state="completed")
    assert again.status is ToolResultStatus.ERROR
    assert "already terminal" in again.error


async def test_results_preserve_official_artifact_json_and_arbitrary_json_values(inbox):
    manager, feature, agent = inbox
    task = await admit(manager)
    value = {"nested": [None, {}, [], False, 0, ""]}
    artifact = new_data_artifact("structured result", value)
    await manager.add_artifact(task.id, artifact, recipient_agent_id=agent.did)
    await feature.respond_to_a2a_task(task.id, "done")
    result = await feature.get_task_result(task.id)
    assert result.status is ToolResultStatus.OK
    assert result.data["artifacts"][0]["artifactId"] == artifact.artifact_id
    assert result.data["artifacts"][0]["parts"][0]["data"] == value
    assert result.data["message"] == "done"


async def test_foreign_recipient_and_unspecified_filter_cannot_expose_tasks(inbox):
    manager, feature, agent = inbox
    task = await admit(manager)
    foreign = TaskFeature(SimpleNamespace(did="did:kestrel:elsewhere"))
    foreign.set_task_manager(manager)
    assert (await foreign.check_task_status(task.id)).status is ToolResultStatus.ERROR
    assert (
        await foreign.respond_to_a2a_task(task.id, "forged reply")
    ).status is ToolResultStatus.ERROR
    assert (await foreign.list_my_tasks(status="submitted")).data["tasks"] == []
    assert (
        await feature.list_my_tasks(status="unspecified")
    ).status is ToolResultStatus.ERROR
    assert "Unknown task status" in await CommandHandler(agent)._cmd_tasks(
        "!tasks unspecified"
    )
    assert (
        await manager.get_task_for_recipient(task.id, agent.did)
    ).status.state == TaskState.TASK_STATE_SUBMITTED


async def test_catalog_commands_and_workflows_use_registered_native_skills(inbox):
    manager, tasks, agent = inbox

    class NativeWorkflowFeature(Feature):
        @property
        def tool_description(self):
            return "Native workflow tool results"

        async def initialize(self):
            pass

        @tool(
            name="native_workflow_echo",
            description="Echo JSON unchanged",
            command_prefix="!native-echo",
        )
        async def echo(self, value: dict) -> dict:
            return value

        @tool(name="native_workflow_fail", description="Reject this workflow step")
        async def reject(self) -> ToolResult:
            return ToolResult.failed("deliberate failure")

    provider = NativeWorkflowFeature(agent)
    manager.register_provider(
        provider.name,
        provider.get_agent_skills(),
        provider,
        command_prefixes={"!native-echo": "native_workflow_echo"},
    )
    catalog = await tasks.list_available_skills()
    assert catalog.status is ToolResultStatus.OK
    assert catalog.data["skill_count"] == 2
    assert provider.name in catalog.data["features"]
    assert CommandHandler(agent)._get_feature_commands() == {
        "!native-echo": "Echo JSON unchanged"
    }
    exposed = manager.get_provider_skills()
    exposed[provider.name][0].description = "changed by consumer"
    assert all(
        skill.description != "changed by consumer"
        for skill in manager.get_agent_skills()
    )
    value = {"nested": [None, {}, [], False]}
    outcome = await tasks.run_workflow(
        [
            {
                "feature": provider.name,
                "skill": "native_workflow_echo",
                "args": {"value": value},
            },
        ]
    )
    assert outcome.status is ToolResultStatus.OK
    assert outcome.data["completed"] == 1 and outcome.data["failed"] == 0
    assert outcome.data["results"][0]["result"]["result"] == value
    failed = await tasks.run_workflow(
        [
            {"feature": provider.name, "skill": "native_workflow_fail", "args": {}},
        ]
    )
    assert failed.status is ToolResultStatus.ERROR
    assert failed.data["failed"] == 1 and failed.data["completed"] == 0
    manager.unregister_provider(provider.name)
    assert manager.get_commands() == {} and manager.get_provider_skills() == {}


@pytest.mark.parametrize(
    "label,outcome,final",
    [
        ("rejected", "failed", True),
        ("completed", "done", True),
        ("failed", "failed", True),
        ("canceled", "failed", True),
        ("submitted", "pending", False),
        ("working", "pending", False),
        ("input-required", "pending", False),
        ("auth-required", "pending", False),
    ],
)
async def test_wait_provider_uses_native_terminal_and_interrupted_semantics(
    inbox, label, outcome, final
):
    from kestrel_sovereign.features.tasks.wait_provider import TaskWaitable
    from kestrel_sovereign.waits.engine import TERMINAL_EVENT_FINAL_KEY

    manager, feature, agent = inbox
    task = await admit(manager)
    if label == "canceled":
        await manager.cancel_task(task.id, agent_name=agent.did)
    elif label != "submitted":
        await manager.update_status(
            task.id, task_state_from_label(label), recipient_agent_id=agent.did
        )
    status = await TaskWaitable(feature).poll(task.id)
    assert status.outcome.value == outcome
    assert status.data.get(TERMINAL_EVENT_FINAL_KEY, False) is final


async def test_operator_inbox_routes_share_native_projection_and_ignore_claimed_sender(
    inbox,
):
    import httpx
    from fastapi import FastAPI
    from kestrel_sovereign.endpoints.task_inbox import router

    manager, feature, agent = inbox
    task = await admit(manager)
    app = FastAPI()
    app.state.agent = agent
    app.include_router(router)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://operator.test"
    ) as client:
        listed = await client.get(
            "/api/agent/task-inbox", params={"status": "submitted", "limit": 50}
        )
        assert listed.status_code == 200
        [entry] = listed.json()["tasks"]
        status = await feature.check_task_status(task.id)
        assert entry == {
            key: value
            for key, value in status.data.items()
            if key not in {"ok", "artifacts"}
        }
        assert entry["sender"] == "did:kestrel:verified-sender"
        fetched = await client.get(f"/api/agent/task-inbox/{task.id}")
        assert fetched.status_code == 200
        assert fetched.json()["task"]["contextId"] == task.context_id
        assert (
            fetched.json()["summary"]["request_content"]
            == "first request part\nsecond request part"
        )
        for invalid in ["submitted-legacy", "unspecified", ""]:
            assert (
                await client.get("/api/agent/task-inbox", params={"status": invalid})
            ).status_code == 400
        for limit in [0, 101]:
            assert (
                await client.get("/api/agent/task-inbox", params={"limit": limit})
            ).status_code == 422
        assert (await client.get("/api/agent/tasks")).status_code == 404
        assert (await client.get(f"/api/agent/tasks/{task.id}")).status_code == 404
        agent.did = "did:kestrel:other"
        assert (await client.get("/api/agent/task-inbox")).status_code == 503
        assert (await client.get(f"/api/agent/task-inbox/{task.id}")).status_code == 503
