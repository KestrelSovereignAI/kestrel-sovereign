"""Native admission, lifecycle, and streaming through the canonical manager."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import replace

import pytest
import pytest_asyncio
from kestrel_sovereign._vendor.a2a.helpers.proto_helpers import new_data_artifact, new_text_message
from kestrel_sovereign._vendor.a2a.types import (
    AgentSkill,
    Part,
    Role,
    SendMessageRequest,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from kestrel_sovereign._vendor.a2a.utils.errors import InvalidParamsError
from google.protobuf.json_format import MessageToDict, ParseDict

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.a2a.lifecycle import TERMINAL_STATES
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.a2a.stores.unified.context_store import (
    ContextOwnershipError,
    ContextStore,
)
from kestrel_sovereign.a2a.stores.unified.observability_store import ObservabilityStore
from kestrel_sovereign.a2a.stores.unified.session_service import SessionService
from kestrel_sovereign.a2a.stores.unified.task_store import TaskStore
from kestrel_sovereign.a2a.task_manager import (
    TaskCancellationAuthorizationError,
    TaskManager,
)
from kestrel_sovereign.storage.db.interface import QueryError, TransactionError
from tests.utils.native_a2a import manager_for


@pytest_asyncio.fixture
async def manager(native_a2a_backend_factory):
    backend = native_a2a_backend_factory()
    await backend.connect()
    instance = manager_for(
        backend, NativeRecipientScope("did:kestrel:recipient", "tenant")
    )
    try:
        await instance.initialize()
        yield instance
    finally:
        await instance.close()


def request(context_id="opaque-context", **metadata):
    return SendMessageRequest(
        tenant="tenant",
        message=new_text_message(
            "Please do this", context_id=context_id, role=Role.ROLE_USER
        ),
        metadata=metadata,
    )


async def admit(manager, params=None, creator="did:kestrel:sender"):
    return await manager.create_task(
        request() if params is None else params, creator_agent_id=creator
    )


async def test_context_admission_cannot_use_a_different_connection(
    native_a2a_backend_factory,
):
    first = native_a2a_backend_factory()
    second = native_a2a_backend_factory()
    with pytest.raises(ValueError, match="same database backend"):
        TaskManager(
            task_store=TaskStore(
                first, scope=NativeRecipientScope("did:kestrel:recipient")
            ),
            context_store=ContextStore(second),
            session_service=SessionService(first),
            observability_store=ObservabilityStore(first),
        )


async def test_retiring_a_manager_does_not_close_the_primary_backend_it_borrows(
    native_a2a_backend_factory,
):
    backend = native_a2a_backend_factory()
    await backend.connect()
    instance = manager_for(
        backend,
        NativeRecipientScope("did:kestrel:recipient", "tenant"),
        owns_backend=False,
    )
    try:
        await instance.initialize()
        accepted = await admit(instance)
        await instance.close()
        assert backend.is_connected
        assert await backend.fetch_all("SELECT id FROM a2a_tasks") == [(accepted.id,)]
    finally:
        await backend.close()


async def test_private_requests_and_missing_required_message_ids_fail_before_admission(
    manager,
):
    with pytest.raises(TypeError, match="official A2A SendMessageRequest"):
        await manager.create_task(
            {"sessionId": "legacy"}, creator_agent_id="did:kestrel:sender"
        )
    params = request()
    params.message.message_id = ""
    with pytest.raises(InvalidParamsError):
        await admit(manager, params)
    assert (
        await manager.task_store.backend.fetch_all(
            "SELECT context_id FROM a2a_contexts"
        )
        == []
    )
    assert await manager.task_store.backend.fetch_all("SELECT id FROM a2a_tasks") == []


@pytest.mark.parametrize("continuation", [False, True])
@pytest.mark.parametrize("with_data", [False, True])
async def test_empty_text_refuses_new_and_continued_input_before_storage(
    manager, continuation, with_data
):
    params = request()
    original = await admit(manager) if continuation else None
    if original is not None:
        params.message.task_id = original.id
    params.message.parts[0].text = ""
    if with_data:
        params.message.parts.append(ParseDict({"data": {"value": 1}}, Part()))
    before = {
        table: await manager.task_store.backend.fetch_all(f"SELECT * FROM {table}")
        for table in ("a2a_tasks", "a2a_contexts", "a2a_task_inputs", "a2a_task_changes")
    }
    with pytest.raises(InvalidParamsError, match="text.*empty"):
        if continuation:
            await manager.continue_task(params, creator_agent_id="did:kestrel:sender")
        else:
            await admit(manager, params)
    for table, rows in before.items():
        assert await manager.task_store.backend.fetch_all(f"SELECT * FROM {table}") == rows


async def test_native_task_ids_are_server_generated_and_message_identity_is_preserved(
    manager,
):
    params = request(nested={"array": [None, {}, []]})
    before = params.SerializeToString()
    first = await admit(manager, params)
    second = await admit(manager, params)
    assert isinstance(first, Task)
    assert first.id and second.id and first.id != second.id
    assert first.context_id == second.context_id == params.message.context_id
    assert first.history[0] == params.message
    assert MessageToDict(first.metadata) == MessageToDict(params.metadata)
    assert params.SerializeToString() == before
    assert await manager.context_store.require_owned(
        first.context_id,
        manager.task_store.scope,
        creator_agent_id="did:kestrel:sender",
    )


@pytest.mark.parametrize(
    "field,value", [("task_id", "attacker-selected"), ("role", Role.ROLE_AGENT)]
)
async def test_new_task_rejects_peer_selected_task_ids_and_non_user_messages(
    manager, field, value
):
    params = request()
    setattr(params.message, field, value)
    with pytest.raises(InvalidParamsError):
        await admit(manager, params)
    assert await manager.task_store.backend.fetch_all("SELECT id FROM a2a_tasks") == []
    assert (
        await manager.task_store.backend.fetch_all(
            "SELECT context_id FROM a2a_contexts"
        )
        == []
    )


async def test_wrong_tenant_or_unverified_creator_has_no_admission_side_effect(manager):
    params = request()
    params.tenant = "other"
    with pytest.raises(InvalidParamsError, match="tenant"):
        await admit(manager, params)
    with pytest.raises(ValueError, match="verified creator"):
        await admit(manager, creator=" ")
    assert await manager.task_store.backend.fetch_all("SELECT id FROM a2a_tasks") == []
    assert (
        await manager.task_store.backend.fetch_all(
            "SELECT context_id FROM a2a_contexts"
        )
        == []
    )


async def test_context_ownership_refuses_another_sender_without_creating_a_task(
    manager,
):
    first = await admit(manager)
    with pytest.raises(ContextOwnershipError, match="not owned"):
        await admit(manager, creator="did:kestrel:other-sender")
    assert await manager.task_store.backend.fetch_all("SELECT id FROM a2a_tasks") == [
        (first.id,)
    ]


async def test_failed_task_insert_rolls_back_context_ownership_and_does_not_wake_the_recipient(
    manager,
):
    backend = manager.task_store.backend
    if manager.task_store.is_sqlite:
        await backend.execute_script("""
            CREATE TRIGGER reject_native_test BEFORE INSERT ON a2a_tasks
            WHEN NEW.task_type = 'reject-admission'
            BEGIN SELECT RAISE(ABORT, 'native admission test refusal'); END;
        """)
    else:
        await backend.execute(
            "ALTER TABLE a2a_tasks ADD CONSTRAINT reject_native_test CHECK (task_type <> 'reject-admission')"
        )
    submissions = []
    manager._on_task_submitted = submissions.append
    with pytest.raises((QueryError, TransactionError)):
        await admit(manager, request(task_type="reject-admission"))
    assert submissions == []
    assert await backend.fetch_all("SELECT id FROM a2a_tasks") == []
    assert await backend.fetch_all("SELECT context_id FROM a2a_contexts") == []


async def test_a_session_projection_failure_does_not_turn_a_committed_task_into_a_send_failure(
    manager,
):
    await manager.task_store.backend.execute("DROP TABLE a2a_sessions")
    submissions = []
    manager._on_task_submitted = submissions.append
    task = await admit(manager)
    assert [item.task.id for item in submissions] == [task.id]
    assert submissions[0].ownership.creator_agent_id == "did:kestrel:sender"
    assert (
        await manager.task_store.get_for_creator(
            task.id,
            "did:kestrel:sender",
            recipient_agent_id=manager.task_store.scope.recipient_id,
        )
        == task
    )


async def test_opaque_peer_context_cannot_append_to_an_existing_host_chat_session(
    manager,
):
    await manager.session_service.create_session(
        agent_name="host",
        session_id="opaque-context",
        metadata={"private": "host-chat"},
    )
    task = await admit(manager)
    host = await manager.session_service.get_session("opaque-context")
    projection = await manager.session_service.get_session(
        manager.session_id_for_context(task.context_id)
    )
    assert host.events == []
    assert host.metadata == {"private": "host-chat"}
    assert len(projection.events) == 1
    assert projection.events[0]["data"]["messageId"] == task.history[0].message_id


@pytest.mark.parametrize(
    "changes", [{"recipient_id": "did:kestrel:other"}, {"tenant": "other"}]
)
async def test_projection_namespaces_include_both_recipient_and_tenant(
    manager, changes
):
    foreign = manager_for(
        manager.task_store.backend, replace(manager.task_store.scope, **changes)
    )
    assert manager.session_id_for_context("opaque") != foreign.session_id_for_context(
        "opaque"
    )


@pytest.mark.parametrize(
    "state", [TaskState.TASK_STATE_INPUT_REQUIRED, TaskState.TASK_STATE_AUTH_REQUIRED]
)
async def test_interrupted_tasks_can_resume_then_complete(manager, state):
    task = await admit(manager)
    recipient = manager.task_store.scope.recipient_id
    interrupted = await manager.update_status(
        task.id, state, recipient_agent_id=recipient
    )
    assert interrupted.status.state == state
    working = await manager.update_status(
        task.id, TaskState.TASK_STATE_WORKING, recipient_agent_id=recipient
    )
    assert working.status.state == TaskState.TASK_STATE_WORKING
    completed = await manager.complete_task(
        task.id, "Done", recipient_agent_id=recipient
    )
    assert completed.status.state == TaskState.TASK_STATE_COMPLETED
    assert completed.status.message.message_id
    assert completed.history[-1].parts[0].text == "Done"


@pytest.mark.parametrize(
    "state", sorted(TERMINAL_STATES - {TaskState.TASK_STATE_CANCELED})
)
async def test_native_stream_starts_with_a_task_and_closes_on_each_terminal_state(
    manager, state
):
    task = await admit(manager)
    recipient = manager.task_store.scope.recipient_id
    stream = manager.subscribe(
        task.id, creator_agent_id="did:kestrel:sender", recipient_agent_id=recipient
    )
    initial = await anext(stream)
    assert isinstance(initial, Task) and initial.id == task.id
    await manager.update_status(task.id, state, recipient_agent_id=recipient)
    event = await asyncio.wait_for(anext(stream), 1)
    assert isinstance(event, TaskStatusUpdateEvent)
    assert event.task_id == task.id and event.context_id == task.context_id
    assert event.status.state == state
    assert "final" not in MessageToDict(event)
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(anext(stream), 1)
    assert task.id not in manager._subscribers


async def test_native_artifact_events_have_context_and_preserve_structured_values(
    manager,
):
    task = await admit(manager)
    recipient = manager.task_store.scope.recipient_id
    stream = manager.subscribe(task.id, recipient_agent_id=recipient)
    await anext(stream)
    artifact = new_data_artifact("result", {"nested": [None, {}, []]})
    updated = await manager.add_artifact(
        task.id, artifact, recipient_agent_id=recipient
    )
    event = await asyncio.wait_for(anext(stream), 1)
    assert isinstance(event, TaskArtifactUpdateEvent)
    assert event.task_id == task.id and event.context_id == task.context_id
    assert event.artifact == updated.artifacts[0] == artifact
    await stream.aclose()
    assert task.id not in manager._subscribers


async def test_creator_cancellation_is_native_and_retries_do_not_repeat_projections(
    manager,
):
    task = await admit(manager)
    first = await manager.cancel_task(task.id, agent_name="did:kestrel:sender")
    projection_id = manager.session_id_for_context(task.context_id)
    events = (await manager.session_service.get_session(projection_id)).events
    retry = await manager.cancel_task(task.id, agent_name="did:kestrel:sender")
    assert first == retry
    assert first.status.state == TaskState.TASK_STATE_CANCELED
    assert (
        MessageToDict(first.metadata)["cancellation_receipt"]["actor_agent_id"]
        == "did:kestrel:sender"
    )
    assert (await manager.session_service.get_session(projection_id)).events == events
    with pytest.raises(TaskCancellationAuthorizationError):
        await manager.cancel_task(task.id, agent_name="did:kestrel:other")


class CompletingProvider:
    def __init__(self):
        self.calls = 0

    async def handle_task(self, task, *, arguments=None):
        self.calls += 1
        task.artifacts.append(
            new_data_artifact("result", {"ok": True, "nested": [None, {}, []]})
        )
        task.status.CopyFrom(TaskStatus(state=TaskState.TASK_STATE_COMPLETED))
        return task

    def get_skill_for_command(self, command):
        return None


class GatedCompletingProvider(CompletingProvider):
    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def handle_task(self, task, *, arguments=None):
        self.started.set()
        await self.release.wait()
        return await super().handle_task(task, arguments=arguments)


class FailingProvider(CompletingProvider):
    async def handle_task(self, task, *, arguments=None):
        self.calls += 1
        raise ValueError("The requested tool operation failed")


@pytest.mark.parametrize("reject_failed", [False, True])
@pytest.mark.parametrize("owns_backend", [False, True])
async def test_background_storage_failure_is_not_a_tool_failure_and_survives_worker_retirement(
    native_a2a_backend_factory,
    reject_failed,
    owns_backend,
):
    backend = native_a2a_backend_factory()
    await backend.connect()
    completions = []
    instance = manager_for(
        backend,
        NativeRecipientScope("did:kestrel:recipient", "tenant"),
        owns_backend=owns_backend,
        on_task_complete=completions.append,
    )
    provider = GatedCompletingProvider()
    execution = None
    try:
        await instance.initialize()
        rejected = "'completed', 'failed'" if reject_failed else "'completed'"
        if instance.task_store.is_sqlite:
            await backend.execute_script(f"""
                CREATE TRIGGER reject_native_async_result BEFORE UPDATE ON a2a_tasks
                WHEN NEW.status IN ({rejected})
                BEGIN SELECT RAISE(ABORT, 'native async result test refusal'); END;
            """)
        else:
            await backend.execute(
                f"ALTER TABLE a2a_tasks ADD CONSTRAINT reject_native_async_result CHECK (status NOT IN ({rejected}))"
            )
        skill = AgentSkill(
            id="native-tool",
            name="Native tool",
            description="Native async dispatch",
            tags=["test"],
        )
        instance.register_provider("feature", [skill], provider)
        accepted = await instance.execute_skill("feature", skill.id, {}, sync=False)
        await asyncio.wait_for(provider.started.wait(), 1)
        execution = next(iter(instance._execution_tasks))
        provider.release.set()
        outcome = await asyncio.wait_for(await_owned_task(execution), 1)
        # Completion may retire the Task before a later drain observes it.
        # Its infrastructure failure must remain owned by the manager.
        assert execution not in instance._execution_tasks
        assert isinstance(outcome.error, TransactionError)
        assert isinstance(outcome.error.__cause__, QueryError)
        assert provider.calls == 1
        assert completions == []
        assert await backend.fetch_all("SELECT id, status FROM a2a_tasks") == [
            (accepted.id, "submitted")
        ]
        for _ in range(2):
            with pytest.raises(TransactionError) as drained:
                await instance.drain_execution_tasks()
            assert drained.value is outcome.error
        with pytest.raises(TransactionError) as closed:
            await instance.close()
        assert closed.value is outcome.error
        assert backend.is_connected is not owns_backend
    finally:
        provider.release.set()
        if execution is not None:
            await await_owned_task(execution)
        # Cleanup must also work while running the deliberately failing
        # pre-fix counterexample, before its close assertion is reached.
        with suppress(TransactionError):
            await instance.close()
        await backend.close()


@pytest.mark.parametrize(
    "provider_type,state",
    [
        (CompletingProvider, TaskState.TASK_STATE_COMPLETED),
        (FailingProvider, TaskState.TASK_STATE_FAILED),
    ],
)
async def test_background_handler_outcome_commits_and_notifies_once(
    manager, provider_type, state
):
    provider = provider_type()
    completions = []
    manager._on_task_complete = completions.append
    skill = AgentSkill(
        id="native-tool",
        name="Native tool",
        description="Native async dispatch",
        tags=["test"],
    )
    manager.register_provider("feature", [skill], provider)
    accepted = await manager.execute_skill("feature", skill.id, {}, sync=False)
    await manager.drain_execution_tasks()
    await manager.drain_execution_tasks()
    persisted = await manager.get_task_for_recipient(accepted.id, manager.did)
    assert persisted.status.state == state
    assert provider.calls == 1
    assert len(completions) == 1
    assert completions[0].task == persisted
    assert completions[0].ownership.creator_agent_id == manager.did
    if state == TaskState.TASK_STATE_FAILED:
        assert persisted.status.message.task_id == persisted.id
        assert persisted.status.message.context_id == persisted.context_id
        assert (
            persisted.status.message.parts[0].text
            == "The requested tool operation failed"
        )


async def test_background_shutdown_joins_a_canceled_handler_and_its_durable_receipt(
    manager,
):
    provider = GatedCompletingProvider()
    skill = AgentSkill(
        id="native-tool",
        name="Native tool",
        description="Native async dispatch",
        tags=["test"],
    )
    manager.register_provider("feature", [skill], provider)
    accepted = await manager.execute_skill("feature", skill.id, {}, sync=False)
    await asyncio.wait_for(provider.started.wait(), 1)
    execution = next(iter(manager._execution_tasks))
    await manager.drain_execution_tasks(cancel=True)
    await manager.drain_execution_tasks()
    persisted = await manager.get_task_for_recipient(accepted.id, manager.did)
    assert persisted.status.state == TaskState.TASK_STATE_CANCELED
    assert provider.calls == 0
    assert execution.cancelled()
    assert manager._execution_tasks == set()


async def test_peer_context_names_cannot_reserve_the_local_command_session_namespace(
    manager,
):
    await admit(manager, request("local-chat-session"))
    provider = CompletingProvider()
    skill = AgentSkill(
        id="native-tool",
        name="Native tool",
        description="Test native dispatch",
        tags=["test"],
    )
    manager.register_provider("feature", [skill], provider)
    result = await manager.execute_skill(
        "feature", skill.id, {}, session_id="local-chat-session"
    )
    assert result.status.state == TaskState.TASK_STATE_COMPLETED
    assert result.context_id != "local-chat-session"
    assert provider.calls == 1


async def test_a_failed_outcome_write_never_returns_a_fabricated_completed_task(
    manager,
):
    backend = manager.task_store.backend
    if manager.task_store.is_sqlite:
        await backend.execute_script("""
            CREATE TRIGGER reject_native_result BEFORE UPDATE ON a2a_tasks
            WHEN NEW.status = 'completed'
            BEGIN SELECT RAISE(ABORT, 'native result test refusal'); END;
        """)
    else:
        await backend.execute(
            "ALTER TABLE a2a_tasks ADD CONSTRAINT reject_native_result CHECK (status <> 'completed')"
        )
    provider = CompletingProvider()
    skill = AgentSkill(
        id="native-tool",
        name="Native tool",
        description="Test native dispatch",
        tags=["test"],
    )
    manager.register_provider("feature", [skill], provider)
    with pytest.raises(TransactionError):
        await manager.execute_skill("feature", skill.id, {})
    assert provider.calls == 1
    assert await backend.fetch_all("SELECT status FROM a2a_tasks") == [("submitted",)]


async def test_feature_skills_use_native_types_and_execute_once_without_a_second_submission_wake(
    manager,
):
    provider = CompletingProvider()
    skill = AgentSkill(
        id="native-tool",
        name="Native tool",
        description="Test native dispatch",
        tags=["test"],
    )
    manager.register_provider("feature", [skill], provider, {"!native": skill.id})
    skill.name = "Changed after registration"
    published = manager.get_agent_skills()
    assert published[0].name == "Native tool"
    published[0].name = "Changed after publication"
    assert manager.get_agent_skills()[0].name == "Native tool"
    submissions = []
    manager._on_task_submitted = submissions.append
    result = await manager.execute_skill(
        "feature", "native-tool", {"nested": [None, {}, []]}
    )
    assert result.status.state == TaskState.TASK_STATE_COMPLETED
    assert provider.calls == 1 and submissions == []
    assert await manager.execute_command("!native") == {
        "ok": True,
        "nested": [None, {}, []],
    }
    assert provider.calls == 2
    manager.unregister_provider("feature")
    assert manager.get_agent_skills() == []
    assert manager.get_agent_for_command("!native") is None


async def test_local_skill_completion_retains_the_current_turn_causation_extension(
    manager,
):
    from datetime import UTC, datetime

    from kestrel_sdk.signals import CausationFrame

    from kestrel_sovereign.a2a.extensions import (
        CAUSATION_URI,
        serialize_chain_for_metadata,
    )
    from kestrel_sovereign.signals.sources.a2a import build_signal_for_completed_task

    cause = CausationFrame(
        agent_id=manager.did,
        source="cron.followup",
        signal_id="signal-one",
        turn_id="turn-one",
        depth=1,
        emitted_at=datetime(2026, 10, 20, tzinfo=UTC),
    )
    manager._causation_chain_provider = lambda: serialize_chain_for_metadata([cause])
    completions = []
    manager._on_task_complete = completions.append
    skill = AgentSkill(
        id="native-tool",
        name="Native tool",
        description="Native causal dispatch",
        tags=["test"],
    )
    manager.register_provider("feature", [skill], CompletingProvider())
    result = await manager.execute_skill("feature", skill.id, {}, sync=False)
    await manager.drain_execution_tasks()
    signal = build_signal_for_completed_task(completions[0], manager.did)
    assert signal.causation_chain == [cause]
    persisted = await manager.get_task_for_recipient(result.id, manager.did)
    metadata = MessageToDict(persisted.metadata)
    assert CAUSATION_URI in metadata and "causation_chain" not in metadata


@pytest.mark.parametrize("readback", ["available", "outage", "missing"])
async def test_lost_commit_ack_reconciles_only_the_canonical_payload(
    manager, monkeypatch, readback
):
    from kestrel_sovereign._vendor.a2a.helpers.proto_helpers import new_text_artifact

    from kestrel_sovereign.a2a.stores.unified.task_store import (
        TaskMutationAuthorizationError,
    )

    provider = CompletingProvider()
    skill = AgentSkill(
        id="native-tool", name="Native tool", description="Ack test", tags=["test"]
    )
    manager.register_provider("feature", [skill], provider)
    canonical_save = manager.task_store.save_recipient_terminal_outcome
    canonical_read = manager.task_store.get_for_recipient
    ack_error = RuntimeError("lost terminal commit acknowledgement")
    read_error = QueryError("canonical payload unavailable")
    committed_id = None
    writes = 0

    async def commit_then_lose_ack(task, **kwargs):
        nonlocal committed_id, writes
        writes += 1
        await manager.task_store.add_artifact(
            task.id,
            new_text_artifact("concurrent", "keep"),
            recipient_agent_id=manager.did,
        )
        saved = await canonical_save(task, **kwargs)
        assert saved is not None
        committed_id = task.id
        raise ack_error

    async def read(task_id, recipient_agent_id):
        if task_id == committed_id:
            if readback == "outage":
                raise read_error
            if readback == "missing":
                return None
        return await canonical_read(task_id, recipient_agent_id)

    monkeypatch.setattr(
        manager.task_store, "save_recipient_terminal_outcome", commit_then_lose_ack
    )
    monkeypatch.setattr(manager.task_store, "get_for_recipient", read)
    if readback == "available":
        returned = await manager.execute_skill("feature", skill.id, {})
        assert returned == await canonical_read(committed_id, manager.did)
        assert {item.name for item in returned.artifacts} == {"concurrent", "result"}
    else:
        with pytest.raises(RuntimeError) as failed:
            await manager.execute_skill("feature", skill.id, {})
        assert failed.value is ack_error
        if readback == "outage":
            assert failed.value.__cause__ is read_error
        else:
            assert isinstance(failed.value.__cause__, TaskMutationAuthorizationError)
    assert provider.calls == writes == 1
    persisted = await canonical_read(committed_id, manager.did)
    assert persisted.status.state == TaskState.TASK_STATE_COMPLETED
    assert {item.name for item in persisted.artifacts} == {"concurrent", "result"}


async def test_denied_local_skill_returns_a_native_rejection_without_running_the_tool(
    manager,
):
    from kestrel_sdk.hooks.base import Hook, HookEvent, HookOutput

    from kestrel_sovereign.hooks.manager import HooksManager

    class DenyExport(Hook):
        def __init__(self):
            super().__init__("deny-native-export", events=[HookEvent.PRE_TOOL_USE])

        async def execute(self, input):
            return HookOutput.deny("This recipient does not permit export")

    manager.hooks_manager = HooksManager()
    manager.hooks_manager.register(DenyExport())
    provider = CompletingProvider()
    manager.register_provider(
        "feature",
        [
            AgentSkill(
                id="native-tool",
                name="Native tool",
                description="Native deny",
                tags=["test"],
            )
        ],
        provider,
    )
    result = await manager.execute_skill("feature", "native-tool", {})
    assert provider.calls == 0
    assert result.status.state == TaskState.TASK_STATE_REJECTED
    assert result.status.message.task_id == result.id
    assert result.status.message.context_id == result.context_id
    assert MessageToDict(result.metadata)["denied"] is True
    assert await manager.get_task_for_recipient(result.id, manager.did) == result


@pytest.mark.parametrize(
    "state", [TaskState.TASK_STATE_INPUT_REQUIRED, TaskState.TASK_STATE_AUTH_REQUIRED]
)
@pytest.mark.parametrize("already_interrupted", [False, True])
async def test_native_stream_closes_on_an_interruption_without_making_the_task_terminal(
    manager, state, already_interrupted
):
    task = await admit(manager)
    recipient = manager.did
    if already_interrupted:
        await manager.update_status(task.id, state, recipient_agent_id=recipient)
    stream = manager.subscribe(
        task.id, creator_agent_id="did:kestrel:sender", recipient_agent_id=recipient
    )
    try:
        initial = await anext(stream)
        assert isinstance(initial, Task) and initial.id == task.id
        if not already_interrupted:
            await manager.update_status(task.id, state, recipient_agent_id=recipient)
            event = await asyncio.wait_for(anext(stream), 1)
            assert isinstance(event, TaskStatusUpdateEvent)
            assert event.status.state == state
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(anext(stream), 1)
        assert task.id not in manager._subscribers
        resumed = await manager.update_status(
            task.id, TaskState.TASK_STATE_WORKING, recipient_agent_id=recipient
        )
        assert resumed.status.state == TaskState.TASK_STATE_WORKING
    finally:
        await stream.aclose()


async def test_manager_owner_is_derived_from_authoritative_scope_and_cannot_be_overwritten(
    manager,
):
    assert manager.did == manager.task_store.scope.recipient_id
    with pytest.raises(AttributeError):
        manager.did = "did:forged:display-name"
    assert manager.did == "did:kestrel:recipient"
    accepted = await admit(manager)
    [event] = await manager.task_store.backend.fetch_all(
        "SELECT agent_name FROM a2a_observability WHERE tool_name = 'task_create'"
    )
    assert event == (manager.did,)
    assert accepted.id


async def test_failure_and_transition_telemetry_remain_owned_by_the_recipient_did(
    manager,
):
    accepted = await admit(manager)
    await manager.fail_task(
        accepted.id,
        "Deliberate failure",
        agent_name="A reporting display name",
        recipient_agent_id=manager.did,
    )
    rows = await manager.task_store.backend.fetch_all(
        "SELECT agent_name FROM a2a_observability"
    )
    assert len(rows) >= 3 and all(row == (manager.did,) for row in rows)
