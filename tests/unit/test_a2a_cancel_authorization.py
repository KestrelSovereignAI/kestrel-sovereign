"""Authority and atomicity regressions for A2A task cancellation (#3134)."""

import asyncio
import sqlite3
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from kestrel_sdk.hooks.base import HookEvent, HookOutput, PermissionDecision
from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.a2a.task_manager import (
    TaskCancellationAuthorizationError,
    create_task_manager,
)
from kestrel_sovereign.a2a.outbound_store import OutboundTaskRouteAmbiguousError
from kestrel_sovereign.a2a.stores.unified.task_store import TaskStore
from kestrel_sovereign.a2a.agent_card import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from kestrel_sovereign.a2a.types import (
    Artifact,
    Message,
    Task,
    TaskStatus,
    TaskSendParams,
    TaskState,
    TextPart,
)
from kestrel_sovereign.agent.event_manager import EventManagerMixin
from kestrel_sovereign.features.tasks.feature import TaskFeature
from kestrel_sovereign.features.peers.directory import (
    LocalHostPeerDirectory,
    PeerAccessDeniedError,
    PeerIdentity,
    PeerRequester,
    PeerTaskConflictError,
)
from kestrel_sovereign.features.peers.feature import PeersFeature
from kestrel_sovereign.multi_agent.agent_manager import AgentManager


def _params(task_id: str, *, metadata=None) -> TaskSendParams:
    return TaskSendParams(
        id=task_id,
        sessionId=f"session-{task_id}",
        message=Message(role="user", parts=[TextPart(text="Do the work")]),
        metadata=metadata or {},
    )


def _feature(manager, agent_did: str) -> TaskFeature:
    feature = TaskFeature(SimpleNamespace(did=agent_did))
    feature.set_task_manager(manager)
    return feature


@pytest.mark.asyncio
async def test_task_tool_requires_peer_route_for_creator_but_allows_recipient(tmp_path):
    manager = await create_task_manager(str(tmp_path / "tasks.db"))
    try:
        await manager.create_task(
            _params("by-owner"),
            agent_name="did:test:recipient",
            creator_agent_id="did:test:creator",
        )

        assert await manager.is_task_recipient(
            "by-owner", "did:test:recipient"
        )
        assert not await manager.is_task_recipient(
            "by-owner", "did:test:creator"
        )
        await manager.create_task(
            _params("by-delegate"),
            agent_name="did:test:recipient",
            creator_agent_id="did:test:creator",
        )

        owner_result = await _feature(manager, "did:test:creator").cancel_task(
            "by-owner", reason="assignment withdrawn"
        )
        delegate_result = await _feature(manager, "did:test:recipient").cancel_task(
            "by-delegate", reason="cannot continue"
        )

        assert owner_result.status is ToolResultStatus.ERROR
        assert "not found" in owner_result.error
        assert delegate_result.status is ToolResultStatus.OK
        owner_task = await manager.task_store._get_unscoped("by-owner")
        delegate_task = await manager.task_store._get_unscoped("by-delegate")
        assert owner_task.status.state is TaskState.SUBMITTED
        assert "cancellation_receipt" not in owner_task.metadata
        assert delegate_task.metadata["cancellation_receipt"] == {
            "actor_agent_id": "did:test:recipient",
            "reason": "cannot continue",
            "status_before": "submitted",
        }
        assert delegate_task.history[-1].parts[0].text == (
            "Task canceled by did:test:recipient: cannot continue"
        )
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_cancel_task_binds_atomic_transition_to_routed_recipient(tmp_path):
    manager = await create_task_manager(str(tmp_path / "routed-recipient.db"))
    try:
        await manager.create_task(
            _params("recipient-bound"),
            agent_name="did:test:actual-recipient",
            creator_agent_id="did:test:creator",
        )

        with pytest.raises(TaskCancellationAuthorizationError):
            await manager.cancel_task(
                "recipient-bound",
                agent_name="did:test:creator",
                recipient_agent_id="did:test:wrong-recipient",
            )

        assert (
            await manager.task_store._get_unscoped("recipient-bound")
        ).status.state is TaskState.SUBMITTED
        canceled = await manager.cancel_task(
            "recipient-bound",
            agent_name="did:test:creator",
            recipient_agent_id="did:test:actual-recipient",
        )
        assert canceled.status.state is TaskState.CANCELED
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_same_actor_cancel_retry_returns_existing_receipt_once(tmp_path):
    manager = await create_task_manager(str(tmp_path / "idempotent-cancel.db"))
    try:
        await manager.create_task(
            _params("idempotent"),
            agent_name="did:test:recipient",
            creator_agent_id="did:test:creator",
        )
        first = await manager.cancel_task(
            "idempotent",
            reason="withdrawn",
            agent_name="did:test:creator",
            recipient_agent_id="did:test:recipient",
        )
        history_count = len(first.history or [])

        retry = await manager.cancel_task(
            "idempotent",
            reason="withdrawn",
            agent_name="did:test:creator",
            recipient_agent_id="did:test:recipient",
        )

        assert retry.metadata["cancellation_receipt"] == first.metadata[
            "cancellation_receipt"
        ]
        assert len(retry.history or []) == history_count
        with pytest.raises(ValueError, match="Invalid state transition"):
            await manager.cancel_task(
                "idempotent",
                reason="different actor",
                agent_name="did:test:recipient",
                recipient_agent_id="did:test:recipient",
            )
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_cancel_retry_reconciles_terminal_projection_after_interruption(
    tmp_path,
):
    """Caller cancellation cannot split projection ownership across a retry."""

    manager = await create_task_manager(str(tmp_path / "cancel-reconcile.db"))
    try:
        await manager.create_task(
            _params("cancel-reconcile"),
            agent_name="did:test:recipient",
            creator_agent_id="did:test:creator",
        )
        terminal_events = asyncio.Queue()
        manager._subscribers["cancel-reconcile"] = [terminal_events]
        completions = []
        manager._on_task_complete = lambda task: completions.append(task.id)
        project = manager._project_status_transition
        projection_started = asyncio.Event()
        release_projection = asyncio.Event()

        async def slow_projection(*args, **kwargs):
            projection_started.set()
            await release_projection.wait()
            await project(*args, **kwargs)

        manager._project_status_transition = slow_projection
        cancellation = asyncio.create_task(
            manager.cancel_task(
                "cancel-reconcile",
                reason="withdrawn",
                agent_name="did:test:creator",
                recipient_agent_id="did:test:recipient",
            )
        )
        await projection_started.wait()
        cancellation.cancel()
        await asyncio.sleep(0)
        assert not cancellation.done()
        release_projection.set()
        with pytest.raises(asyncio.CancelledError):
            await cancellation
        assert (
            await manager.task_store._get_unscoped("cancel-reconcile")
        ).status.state is TaskState.CANCELED

        manager._project_status_transition = project
        session_before = await manager.session_service.get_session(
            "session-cancel-reconcile"
        )
        session_events_before = len(
            [
                event
                for event in session_before.events
                if event.get("event_type") == "status_update"
            ]
        )
        memory_before = await manager.task_store._backend.fetch_one(
            "SELECT COUNT(*) FROM a2a_memory WHERE session_id = ?",
            ("session-cancel-reconcile",),
        )
        retry = await manager.cancel_task(
            "cancel-reconcile",
            reason="withdrawn",
            agent_name="did:test:creator",
            recipient_agent_id="did:test:recipient",
        )

        event = terminal_events.get_nowait()
        assert retry.status.state is TaskState.CANCELED
        assert event["event"] == "status"
        assert event["final"] is True
        assert terminal_events.empty()
        assert completions == ["cancel-reconcile"]
        session_after = await manager.session_service.get_session(
            "session-cancel-reconcile"
        )
        assert len(
            [
                event
                for event in session_after.events
                if event.get("event_type") == "status_update"
            ]
        ) == session_events_before
        assert await manager.task_store._backend.fetch_one(
            "SELECT COUNT(*) FROM a2a_memory WHERE session_id = ?",
            ("session-cancel-reconcile",),
        ) == memory_before
    finally:
        await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("with_payload", [False, True])
async def test_cancel_readback_is_atomic_with_authorized_transition(
    tmp_path,
    with_payload,
):
    """A failed public read cannot split a committed cancel from projections."""

    manager = await create_task_manager(str(tmp_path / f"cancel-read-{with_payload}.db"))
    try:
        await manager.create_task(
            _params("cancel-readback"),
            agent_name="did:test:recipient",
            creator_agent_id="did:test:creator",
        )
        canonical_get = manager.task_store._get_unscoped
        current = await canonical_get("cancel-readback")
        task_payload = (
            current.model_copy(
                update={"status": TaskStatus(state=TaskState.CANCELED)}
            )
            if with_payload
            else None
        )
        canceled_callbacks: list[str] = []
        manager._on_task_cancelled = lambda task: canceled_callbacks.append(task.id)
        manager.task_store._get_unscoped = AsyncMock(
            side_effect=RuntimeError("injected post-update public read failure")
        )

        canceled = await manager.cancel_task(
            "cancel-readback",
            reason="withdrawn",
            agent_name="did:test:creator",
            recipient_agent_id="did:test:recipient",
            task_payload=task_payload,
        )

        assert canceled.status.state is TaskState.CANCELED
        assert canceled_callbacks == ["cancel-readback"]
        assert (
            await canonical_get("cancel-readback")
        ).status.state is TaskState.CANCELED
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_ambiguous_cancel_commit_reconciles_projections_before_rethrow(
    tmp_path,
):
    """A lost COMMIT acknowledgement cannot strand canonical cancellation."""

    manager = await create_task_manager(str(tmp_path / "ambiguous-commit.db"))
    try:
        await manager.create_task(
            _params("ambiguous-commit"),
            agent_name="did:test:recipient",
            creator_agent_id="did:test:creator",
        )
        terminal_events = asyncio.Queue()
        manager._subscribers["ambiguous-commit"] = [terminal_events]
        canceled_callbacks: list[str] = []
        completions: list[str] = []
        manager._on_task_cancelled = (
            lambda task: canceled_callbacks.append(task.id)
        )
        manager._on_task_complete = lambda task: completions.append(task.id)
        canonical_cancel = manager.task_store.cancel_if_authorized

        async def commit_then_lose_ack(*args, **kwargs):
            committed = await canonical_cancel(*args, **kwargs)
            assert committed is not None
            raise RuntimeError("lost PostgreSQL COMMIT acknowledgement")

        manager.task_store.cancel_if_authorized = commit_then_lose_ack

        with pytest.raises(RuntimeError, match="COMMIT acknowledgement"):
            await manager.cancel_task(
                "ambiguous-commit",
                reason="withdrawn",
                agent_name="did:test:creator",
                recipient_agent_id="did:test:recipient",
            )

        persisted = await manager.get_task_for_creator(
            "ambiguous-commit", "did:test:creator"
        )
        assert persisted.status.state is TaskState.CANCELED
        assert canceled_callbacks == ["ambiguous-commit"]
        assert completions == ["ambiguous-commit"]
        event = terminal_events.get_nowait()
        assert event["event"] == "status"
        assert event["final"] is True
        assert terminal_events.empty()

        manager.task_store.cancel_if_authorized = canonical_cancel
        retry = await manager.cancel_task(
            "ambiguous-commit",
            reason="withdrawn",
            agent_name="did:test:creator",
            recipient_agent_id="did:test:recipient",
        )

        assert retry.status.state is TaskState.CANCELED
        assert canceled_callbacks == ["ambiguous-commit"]
        assert completions == ["ambiguous-commit"]
        assert terminal_events.empty()
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_ambiguous_cancel_does_not_claim_an_older_same_actor_receipt(
    tmp_path,
):
    """Only this attempt's durable token can prove an ambiguous commit."""

    manager = await create_task_manager(str(tmp_path / "old-receipt.db"))
    try:
        await manager.create_task(
            _params("old-receipt"),
            agent_name="did:test:recipient",
            creator_agent_id="did:test:creator",
        )
        await manager.cancel_task(
            "old-receipt",
            agent_name="did:test:creator",
        )
        canceled_callbacks: list[str] = []
        completions: list[str] = []
        manager._on_task_cancelled = (
            lambda task: canceled_callbacks.append(task.id)
        )
        manager._on_task_complete = lambda task: completions.append(task.id)

        async def fail_before_commit(*_args, **_kwargs):
            raise RuntimeError("transport failed before this attempt committed")

        manager.task_store.cancel_if_authorized = fail_before_commit
        with pytest.raises(RuntimeError, match="before this attempt committed"):
            await manager.cancel_task(
                "old-receipt",
                agent_name="did:test:creator",
            )

        assert canceled_callbacks == []
        assert completions == []
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_create_task_never_publishes_a_stale_submitted_snapshot(tmp_path):
    manager = await create_task_manager(str(tmp_path / "create-readback.db"))
    try:
        task_id = "create-race"
        notifications = asyncio.Queue()
        manager._subscribers[task_id] = [notifications]

        async def cancel_during_projection(_session_id):
            canceled = await manager.task_store.cancel_if_authorized(
                task_id,
                actor_agent_id="did:test:creator",
                reason="raced admission",
            )
            assert canceled is not None
            return None

        manager.session_service.get_session = AsyncMock(
            side_effect=cancel_during_projection
        )
        created = await manager.create_task(
            _params(task_id),
            agent_name="did:test:recipient",
            creator_agent_id="did:test:creator",
        )

        assert created.status.state is TaskState.CANCELED
        assert notifications.empty()
    finally:
        await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "read_error",
    [RuntimeError("read outage"), asyncio.CancelledError()],
)
async def test_create_task_returns_accepted_snapshot_when_final_readback_fails(
    tmp_path,
    read_error,
):
    manager = await create_task_manager(str(tmp_path / "create-readback-failure.db"))
    try:
        accepted: list[str] = []
        manager._on_task_submitted = lambda task: accepted.append(task.id)
        manager._notify_status_update = AsyncMock()
        original_get = manager.task_store._get_unscoped
        manager.task_store._get_unscoped = AsyncMock(side_effect=read_error)

        created = await manager.create_task(
            _params("accepted-before-readback-failure"),
            agent_name="did:test:recipient",
            creator_agent_id="did:test:creator",
        )

        assert created.id == "accepted-before-readback-failure"
        assert created.status.state is TaskState.SUBMITTED
        assert accepted == ["accepted-before-readback-failure"]
        manager._notify_status_update.assert_not_awaited()
        manager.task_store._get_unscoped = original_get
        persisted = await manager.get_task_for_creator(
            created.id, "did:test:creator"
        )
        assert persisted is not None
        assert persisted.status.state is TaskState.SUBMITTED
    finally:
        await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("with_payload", [False, True])
async def test_cancel_commit_does_not_depend_on_a_post_commit_read(
    tmp_path,
    with_payload,
):
    manager = await create_task_manager(str(tmp_path / "cancel-readback-failure.db"))
    try:
        submitted = await manager.create_task(
            _params("cancel-without-readback"),
            agent_name="did:test:recipient",
            creator_agent_id="did:test:creator",
        )
        completions = []
        cancellations = []
        manager._on_task_complete = lambda task: completions.append(task.id)
        manager._on_task_cancelled = lambda task: cancellations.append(task.id)
        task_payload = None
        if with_payload:
            task_payload = submitted.model_copy(deep=True)
            task_payload.status = TaskStatus(state=TaskState.CANCELED)
            task_payload.artifacts = [
                Artifact(name="partial", parts=[TextPart(text="preserved")])
            ]

        original_get = manager.task_store._get_unscoped
        manager.task_store._get_unscoped = AsyncMock(
            side_effect=RuntimeError("post-commit reads unavailable")
        )
        canceled = await manager.cancel_task(
            submitted.id,
            reason="withdrawn",
            agent_name="did:test:creator",
            recipient_agent_id="did:test:recipient",
            task_payload=task_payload,
        )

        assert canceled.status.state is TaskState.CANCELED
        assert cancellations == [submitted.id]
        assert completions == [submitted.id]
        manager.task_store._get_unscoped = original_get
        persisted = await manager.get_task_for_creator(
            submitted.id, "did:test:creator"
        )
        assert persisted.status.state is TaskState.CANCELED
        if with_payload:
            assert persisted.artifacts[0].name == "partial"
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_artifact_append_cannot_mutate_an_authoritatively_canceled_task(tmp_path):
    manager = await create_task_manager(str(tmp_path / "artifact-after-cancel.db"))
    try:
        await manager.create_task(
            _params("artifact-after-cancel"),
            agent_name="did:test:recipient",
            creator_agent_id="did:test:creator",
        )
        await manager.cancel_task(
            "artifact-after-cancel",
            reason="work withdrawn",
            agent_name="did:test:creator",
        )

        with pytest.raises(ValueError, match="terminal task"):
            await manager.add_artifact(
                "artifact-after-cancel",
                Artifact(name="late", parts=[TextPart(text="must not land")]),
                recipient_agent_id="did:test:recipient",
            )

        canceled = await manager.task_store._get_unscoped("artifact-after-cancel")
        assert canceled.status.state is TaskState.CANCELED
        assert not canceled.artifacts
    finally:
        await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "caller,metadata",
    [
        ("did:test:peer", {}),
        ("did:test:child", {"parent_did": "did:test:creator"}),
        (
            "did:test:causal-agent",
            {"causation_chain": [{"agent_id": "did:test:causal-agent"}]},
        ),
    ],
)
async def test_cancel_task_unauthorized_peer_lineage_and_causation_is_refused(
    tmp_path, caller, metadata
):
    manager = await create_task_manager(str(tmp_path / f"{caller.rsplit(':', 1)[-1]}.db"))
    try:
        await manager.create_task(
            _params("protected", metadata=metadata),
            agent_name="did:test:recipient",
            creator_agent_id="did:test:creator",
        )

        result = await _feature(manager, caller).cancel_task("protected")

        assert result.status is ToolResultStatus.ERROR
        assert result.error == "Task protected not found"
        unchanged = await manager.task_store._get_unscoped("protected")
        assert unchanged.status.state is TaskState.SUBMITTED
        assert "cancellation_receipt" not in (unchanged.metadata or {})
    finally:
        await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("actor", "expected_recipient", "terminal"),
    [
        ("did:test:peer", None, False),
        ("did:test:creator", "did:test:wrong-recipient", False),
        ("did:test:creator", None, True),
    ],
)
async def test_cancel_predicate_rejects_before_loading_unavailable_payload(
    tmp_path,
    actor,
    expected_recipient,
    terminal,
):
    """A predicate miss must not lock and deserialize the victim payload."""

    manager = await create_task_manager(
        str(tmp_path / f"early-cancel-rejection-{terminal}.db")
    )
    try:
        await manager.create_task(
            _params("protected-payload"),
            agent_name="did:test:recipient",
            creator_agent_id="did:test:creator",
        )
        status_assignment = ", status = 'completed'" if terminal else ""
        await manager.task_store.backend.execute(
            f"""
            UPDATE a2a_tasks
            SET artifacts = ?{status_assignment}
            WHERE id = ?
            """,
            ("payload-must-not-be-decoded", "protected-payload"),
        )

        refused = await manager.task_store.cancel_if_authorized(
            "protected-payload",
            actor_agent_id=actor,
            expected_recipient_agent_id=expected_recipient,
        )

        assert refused is None
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_cancel_task_does_not_trust_stale_or_spoofed_sender_metadata(tmp_path):
    """Only the verified creator parameter may mint creator authority."""

    manager = await create_task_manager(str(tmp_path / "spoof.db"))
    try:
        await manager.create_task(
            _params(
                "spoofed",
                metadata={
                    "sender": "did:test:revoked",
                    "sender_verified": True,
                    "signature": {"stale": "untrusted task metadata"},
                },
            ),
            agent_name="did:test:recipient",
        )

        result = await _feature(manager, "did:test:revoked").cancel_task("spoofed")

        assert result.status is ToolResultStatus.ERROR
        assert (await manager.task_store._get_unscoped("spoofed")).status.state is TaskState.SUBMITTED
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_task_creation_strips_sender_authored_cancellation_receipt(tmp_path):
    """Only the atomic transition may publish or persist its receipt."""

    manager = await create_task_manager(str(tmp_path / "forged-receipt.db"))
    try:
        submitted = []
        manager._on_task_submitted = submitted.append
        created = await manager.create_task(
            _params(
                "forged-receipt",
                metadata={
                    "sender": "did:test:creator",
                    "cancellation_receipt": {
                        "actor_agent_id": "did:test:creator",
                        "reason": "forged",
                        "status_before": "working",
                    },
                },
            ),
            agent_name="did:test:recipient",
            creator_agent_id="did:test:creator",
        )

        assert "cancellation_receipt" not in (created.metadata or {})
        assert len(submitted) == 1
        assert "cancellation_receipt" not in (submitted[0].metadata or {})
        persisted = await manager.task_store._get_unscoped("forged-receipt")
        assert persisted.status.state is TaskState.SUBMITTED
        assert "cancellation_receipt" not in (persisted.metadata or {})
        assert persisted.metadata["sender"] == "did:test:creator"
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_cancellation_suppresses_an_already_queued_submission_wake(tmp_path):
    """A durable cancellation must cancel the matching cognition delivery."""

    manager = await create_task_manager(str(tmp_path / "queued-wake.db"))
    release_delivery = asyncio.Event()
    enqueue_started = asyncio.Event()
    postcheck_complete = asyncio.Event()
    cognition_executed = asyncio.Event()

    class Handle:
        def __init__(self):
            async def queued_delivery():
                await release_delivery.wait()
                cognition_executed.set()
                return SimpleNamespace(
                    status=SimpleNamespace(value="ok"),
                    error=None,
                )

            self.task = asyncio.create_task(queued_delivery())

        async def wait(self):
            postcheck_complete.set()
            return await self.task

    class Dispatcher:
        handle = None

        async def enqueue_signal(self, _signal):
            self.handle = Handle()
            enqueue_started.set()
            return self.handle

    class Recipient(EventManagerMixin):
        did = "did:test:recipient"

        def __init__(self):
            self.dispatcher = Dispatcher()
            self.task_manager = manager
            self.background_tasks = set()

        def _track_background_task(self, coroutine, *, name):
            task = asyncio.create_task(coroutine, name=name)
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
            return task

    recipient = Recipient()
    manager._on_task_submitted = recipient._on_task_submitted
    manager._on_task_cancelled = recipient._on_task_cancelled
    try:
        await manager.create_task(
            _params("queued-wake"),
            agent_name=recipient.did,
            creator_agent_id="did:test:creator",
        )
        await asyncio.wait_for(enqueue_started.wait(), timeout=1)
        await asyncio.wait_for(postcheck_complete.wait(), timeout=1)

        await manager.cancel_task(
            "queued-wake",
            reason="withdrawn before execution",
            agent_name="did:test:creator",
        )
        await asyncio.sleep(0)

        assert recipient.dispatcher.handle.task.cancelled()
        assert not cognition_executed.is_set()
    finally:
        release_delivery.set()
        if recipient.background_tasks:
            await asyncio.gather(
                *tuple(recipient.background_tasks),
                return_exceptions=True,
            )
        await manager.close()


@pytest.mark.asyncio
async def test_recipient_decline_does_not_cancel_its_current_dispatch(tmp_path):
    """The signal-driven decline must finish projection and return normally."""

    manager = await create_task_manager(str(tmp_path / "self-decline.db"))

    class Recipient(EventManagerMixin):
        did = "did:test:recipient"

        def __init__(self):
            self._a2a_submitted_signal_handles = {}

    recipient = Recipient()
    manager._on_task_cancelled = recipient._on_task_cancelled
    manager._on_task_cancellation_started = (
        recipient._on_task_cancellation_started
    )
    manager._project_status_transition = AsyncMock()
    try:
        created = await manager.create_task(
            _params("self-decline"),
            agent_name=recipient.did,
            creator_agent_id="did:test:creator",
        )

        async def decline_inside_dispatch():
            current = asyncio.current_task()
            recipient._a2a_submitted_signal_handles[created.id] = (
                SimpleNamespace(task=current)
            )
            result = await manager.cancel_task(
                created.id,
                reason="recipient cannot continue",
                agent_name=recipient.did,
            )
            # This await is where the old callback injected CancelledError.
            await asyncio.sleep(0)
            return result

        dispatch = asyncio.create_task(decline_inside_dispatch())
        result = await asyncio.wait_for(dispatch, timeout=1)

        assert result.status.state is TaskState.CANCELED
        assert dispatch.cancelled() is False
        manager._project_status_transition.assert_awaited_once()
        assert created.id not in recipient._a2a_submitted_signal_handles
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_post_registration_gate_does_not_cancel_working_dispatch(tmp_path):
    """A legitimate SUBMITTED→WORKING transition is not cancellation."""

    manager = await create_task_manager(str(tmp_path / "working-wake.db"))
    release_delivery = asyncio.Event()
    postcheck_complete = asyncio.Event()

    class Handle:
        def __init__(self):
            async def active_delivery():
                await release_delivery.wait()
                return SimpleNamespace(
                    status=SimpleNamespace(value="ok"),
                    error=None,
                )

            self.task = asyncio.create_task(active_delivery())

        async def wait(self):
            postcheck_complete.set()
            return await self.task

    class Dispatcher:
        handle = None

        async def enqueue_signal(self, _signal):
            self.handle = Handle()
            return self.handle

    class Recipient(EventManagerMixin):
        did = "did:test:recipient"

        def __init__(self):
            self.dispatcher = Dispatcher()
            self.task_manager = manager
            self.background_tasks = set()

        def _track_background_task(self, coroutine, *, name):
            task = asyncio.create_task(coroutine, name=name)
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
            return task

    recipient = Recipient()
    try:
        created = await manager.create_task(
            _params("working-wake"),
            agent_name=recipient.did,
            creator_agent_id="did:test:creator",
        )
        await manager.update_status(
            created.id,
            TaskState.WORKING,
            agent_name=recipient.did,
            recipient_agent_id=recipient.did,
        )

        recipient._on_task_submitted(created)
        await asyncio.wait_for(postcheck_complete.wait(), timeout=1)
        active = next(iter(recipient.background_tasks))
        assert not recipient.dispatcher.handle.task.cancelled()
        assert not active.done()

        release_delivery.set()
        await asyncio.wait_for(active, timeout=1)
        assert not active.cancelled()
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_execution_worker_rejects_cross_worker_durable_cancellation(
    tmp_path,
):
    """A stale process-local wake cannot execute after another worker cancels."""

    from kestrel_sdk.signals import Status
    from kestrel_sovereign.signals import (
        OrderedLockManager,
        SignalDispatcher,
        SignalLogStore,
        SourceRegistry,
    )
    from kestrel_sovereign.signals.sources.a2a_task_submitted import (
        build_a2a_task_submitted_registration,
        build_signal_for_submitted_task,
    )
    from kestrel_sovereign.storage.db import SQLiteBackend

    shared_path = str(tmp_path / "shared-task-authority.db")
    execution_manager = await create_task_manager(shared_path)
    cancellation_manager = await create_task_manager(shared_path)
    signal_backend = SQLiteBackend(str(tmp_path / "signal-log.db"))
    await signal_backend.connect()
    signal_store = SignalLogStore(signal_backend)
    await signal_store.initialize()

    class ExecutionWorker(EventManagerMixin):
        did = "did:test:recipient"

        def __init__(self):
            self.task_manager = execution_manager
            self.process_input_calls = []
            self.background_tasks = []

        async def process_input(self, prompt):
            self.process_input_calls.append(prompt)
            return "should not execute"

        def _track_background_task(self, coroutine, *, name):
            task = asyncio.create_task(coroutine, name=name)
            self.background_tasks.append(task)
            return task

    worker = ExecutionWorker()
    registry = SourceRegistry()
    registry.register(build_a2a_task_submitted_registration())
    dispatcher = SignalDispatcher(
        agent=worker,
        registry=registry,
        lock_manager=OrderedLockManager(),
        store=signal_store,
    )
    try:
        task = await execution_manager.create_task(
            _params(
                "cross-worker-canceled",
                metadata={
                    "sender": "did:test:creator",
                    "sender_verified": True,
                },
            ),
            agent_name=worker.did,
            creator_agent_id="did:test:creator",
        )
        stale_wake = build_signal_for_submitted_task(
            task,
            target_agent=worker.did,
            sender="did:test:creator",
        )

        await cancellation_manager.cancel_task(
            task.id,
            reason="withdrawn on another worker",
            agent_name="did:test:creator",
        )
        result = await dispatcher.dispatch_signal(stale_wake)

        assert result.status is Status.DROPPED_VALIDATION
        assert "already 'canceled'" in (result.error or "")
        assert worker.process_input_calls == []
    finally:
        pending = [task for task in worker.background_tasks if not task.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await execution_manager.close()
        await cancellation_manager.close()
        await signal_backend.close()


@pytest.mark.asyncio
async def test_execution_worker_stops_when_cross_worker_cancels_mid_turn(
    tmp_path,
):
    """Durable cancellation remains live authority after initial validation."""

    from kestrel_sdk.signals import Status
    from kestrel_sovereign.signals import (
        OrderedLockManager,
        SignalDispatcher,
        SignalLogStore,
        SourceRegistry,
    )
    from kestrel_sovereign.signals.sources.a2a_task_submitted import (
        build_a2a_task_submitted_registration,
        build_signal_for_submitted_task,
    )
    from kestrel_sovereign.storage.db import SQLiteBackend

    shared_path = str(tmp_path / "shared-live-task-authority.db")
    execution_manager = await create_task_manager(shared_path)
    cancellation_manager = await create_task_manager(shared_path)
    signal_backend = SQLiteBackend(str(tmp_path / "live-signal-log.db"))
    await signal_backend.connect()
    signal_store = SignalLogStore(signal_backend)
    await signal_store.initialize()
    cognition_started = asyncio.Event()
    cognition_stopped = asyncio.Event()

    class ExecutionWorker(EventManagerMixin):
        did = "did:test:recipient"

        def __init__(self):
            self.task_manager = execution_manager
            self.background_tasks = []

        async def process_input(self, _prompt):
            cognition_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cognition_stopped.set()

        def _track_background_task(self, coroutine, *, name):
            task = asyncio.create_task(coroutine, name=name)
            self.background_tasks.append(task)
            return task

    worker = ExecutionWorker()
    registry = SourceRegistry()
    registry.register(build_a2a_task_submitted_registration())
    dispatcher = SignalDispatcher(
        agent=worker,
        registry=registry,
        lock_manager=OrderedLockManager(),
        store=signal_store,
    )
    try:
        task = await execution_manager.create_task(
            _params(
                "cross-worker-live-cancel",
                metadata={
                    "sender": "did:test:creator",
                    "sender_verified": True,
                },
            ),
            agent_name=worker.did,
            creator_agent_id="did:test:creator",
        )
        wake = build_signal_for_submitted_task(
            task,
            target_agent=worker.did,
            sender="did:test:creator",
        )
        delivery = asyncio.create_task(dispatcher.dispatch_signal(wake))
        await asyncio.wait_for(cognition_started.wait(), timeout=1)

        await cancellation_manager.cancel_task(
            task.id,
            reason="withdrawn during execution on another worker",
            agent_name="did:test:creator",
        )
        result = await asyncio.wait_for(delivery, timeout=2)

        assert result.status is Status.DROPPED_VALIDATION
        assert "canceled while" in (result.error or "")
        assert cognition_stopped.is_set()
    finally:
        await execution_manager.close()
        await cancellation_manager.close()
        await signal_backend.close()


@pytest.mark.asyncio
async def test_recipient_decline_finishes_under_live_cancellation_monitor(
    tmp_path,
):
    """A recipient's in-turn decline is not mistaken for a remote Stop."""

    from kestrel_sdk.signals import Status
    from kestrel_sovereign.signals import (
        OrderedLockManager,
        SignalDispatcher,
        SignalLogStore,
        SourceRegistry,
    )
    from kestrel_sovereign.signals.sources.a2a_task_submitted import (
        build_a2a_task_submitted_registration,
        build_signal_for_submitted_task,
    )
    from kestrel_sovereign.storage.db import SQLiteBackend

    manager = await create_task_manager(str(tmp_path / "self-decline-live.db"))
    signal_backend = SQLiteBackend(str(tmp_path / "self-decline-signal-log.db"))
    await signal_backend.connect()
    signal_store = SignalLogStore(signal_backend)
    await signal_store.initialize()

    class Recipient(EventManagerMixin):
        did = "did:test:recipient"

        def __init__(self):
            self.task_manager = manager
            self.task_id = ""
            self.background_tasks = []
            self.decline_finished = False

        async def process_input(self, _prompt):
            result = await manager.cancel_task(
                self.task_id,
                reason="recipient cannot continue",
                agent_name=self.did,
            )
            await asyncio.sleep(0.1)
            self.decline_finished = True
            return result.status.state.value

        def _track_background_task(self, coroutine, *, name):
            task = asyncio.create_task(coroutine, name=name)
            self.background_tasks.append(task)
            return task

    recipient = Recipient()
    manager._on_task_cancelled = recipient._on_task_cancelled
    manager._on_task_cancellation_started = (
        recipient._on_task_cancellation_started
    )
    registry = SourceRegistry()
    registry.register(build_a2a_task_submitted_registration())
    dispatcher = SignalDispatcher(
        agent=recipient,
        registry=registry,
        lock_manager=OrderedLockManager(),
        store=signal_store,
    )
    try:
        task = await manager.create_task(
            _params(
                "self-decline-live",
                metadata={
                    "sender": "did:test:creator",
                    "sender_verified": True,
                },
            ),
            agent_name=recipient.did,
            creator_agent_id="did:test:creator",
        )
        recipient.task_id = task.id
        wake = build_signal_for_submitted_task(
            task,
            target_agent=recipient.did,
            sender="did:test:creator",
        )

        result = await dispatcher.dispatch_signal(wake)

        assert result.status is Status.OK
        assert recipient.decline_finished is True
        assert not vars(recipient).get("_a2a_self_declining_task_ids", set())
    finally:
        await manager.close()
        await signal_backend.close()


@pytest.mark.asyncio
async def test_provisional_self_decline_does_not_exempt_creator_winning_receipt():
    """The receipt actor, not an in-flight local marker, owns the exemption."""

    task_id = "creator-won-self-decline-race"
    canceled = SimpleNamespace(
        state="canceled",
        actor_agent_id="did:test:creator",
    )

    class Recipient(EventManagerMixin):
        did = "did:test:recipient"

    recipient = Recipient()
    recipient.task_manager = SimpleNamespace(
        get_task_cancellation_snapshot=AsyncMock(return_value=canceled)
    )
    recipient._a2a_self_declining_task_ids = {task_id}
    signal = SimpleNamespace(
        source="a2a.task_submitted",
        payload={"task_id": task_id},
    )

    withdrawal = await recipient.monitor_cognition_signal_execution(signal)

    assert "was canceled while" in withdrawal
    assert recipient._a2a_self_declining_task_ids == set()


@pytest.mark.asyncio
async def test_live_cancellation_monitor_uses_lightweight_snapshot():
    """The high-frequency monitor must not deserialize the complete task row."""

    task_id = "lightweight-cancellation-monitor"
    full_task = SimpleNamespace(
        status=SimpleNamespace(state=TaskState.CANCELED),
        metadata={
            "cancellation_receipt": {
                "actor_agent_id": "did:test:creator",
            }
        },
    )
    task_manager = SimpleNamespace(
        get_task=AsyncMock(return_value=full_task),
        get_task_cancellation_snapshot=AsyncMock(
            return_value=SimpleNamespace(
                state="canceled",
                actor_agent_id="did:test:creator",
            )
        ),
    )

    class Recipient(EventManagerMixin):
        did = "did:test:recipient"

    recipient = Recipient()
    recipient.task_manager = task_manager
    signal = SimpleNamespace(
        source="a2a.task_submitted",
        payload={"task_id": task_id},
    )

    withdrawal = await recipient.monitor_cognition_signal_execution(signal)

    assert "was canceled while" in withdrawal
    task_manager.get_task_cancellation_snapshot.assert_awaited_once_with(
        task_id,
        recipient_agent_id=recipient.did,
    )
    task_manager.get_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_live_cancellation_monitor_backs_off_durable_reads(monkeypatch):
    """Long cognition turns must not poll shared storage at a fixed rate."""

    snapshots = [
        SimpleNamespace(state="submitted", actor_agent_id=None),
        SimpleNamespace(state="submitted", actor_agent_id=None),
        SimpleNamespace(state="submitted", actor_agent_id=None),
        SimpleNamespace(state="canceled", actor_agent_id="did:test:creator"),
    ]
    task_manager = SimpleNamespace(
        get_task_cancellation_snapshot=AsyncMock(side_effect=snapshots),
    )
    slept = []

    async def record_sleep(delay):
        slept.append(delay)

    monkeypatch.setattr(
        "kestrel_sovereign.agent.event_manager.asyncio.sleep",
        record_sleep,
    )

    class Recipient(EventManagerMixin):
        did = "did:test:recipient"

    recipient = Recipient()
    recipient.task_manager = task_manager
    signal = SimpleNamespace(
        source="a2a.task_submitted",
        payload={"task_id": "backing-off-monitor"},
    )

    withdrawal = await recipient.monitor_cognition_signal_execution(signal)

    assert "was canceled while" in withdrawal
    assert slept == [0.1, 0.2, 0.4]


@pytest.mark.asyncio
async def test_cancellation_snapshot_selects_only_authority_columns():
    """Polling must not fetch message, history, artifacts, or metadata."""

    backend = SimpleNamespace(
        fetch_one=AsyncMock(
            return_value=("canceled", "did:test:creator")
        )
    )
    store = TaskStore(backend)

    snapshot = await store.get_cancellation_snapshot(
        "task-1",
        recipient_agent_id="did:test:recipient",
    )

    query, params = backend.fetch_one.await_args.args
    assert "recipient_agent_id = ?" in query
    assert params == ("task-1", "did:test:recipient")
    assert snapshot.state == "canceled"
    assert snapshot.actor_agent_id == "did:test:creator"


@pytest.mark.asyncio
async def test_refused_cancellation_rolls_back_local_execution_exemption(
    tmp_path,
):
    """A failed authority predicate cannot leave a later wake exempt."""

    manager = await create_task_manager(str(tmp_path / "intent-rollback.db"))
    local_intents = set()

    def mark_intent(task_id, _actor):
        local_intents.add(task_id)

        def rollback():
            local_intents.discard(task_id)

        return rollback

    manager._on_task_cancellation_started = mark_intent
    try:
        await manager.create_task(
            _params("refused-intent"),
            agent_name="did:test:recipient",
            creator_agent_id="did:test:creator",
        )

        with pytest.raises(TaskCancellationAuthorizationError):
            await manager.cancel_task(
                "refused-intent",
                agent_name="did:test:stranger",
            )

        assert local_intents == set()
        assert (
            await manager.task_store._get_unscoped("refused-intent")
        ).status.state is TaskState.SUBMITTED
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_idempotent_recipient_decline_keeps_live_execution_exemption(tmp_path):
    """A same-actor receipt retry remains an authorized in-turn decline."""

    from kestrel_sdk.signals import Signal, SignalMode
    from kestrel_sovereign.signals.context import (
        reset_current_signal,
        set_current_signal,
    )

    manager = await create_task_manager(str(tmp_path / "idempotent-decline.db"))

    class Recipient(EventManagerMixin):
        did = "did:test:recipient"

    recipient = Recipient()
    manager._on_task_cancellation_started = (
        recipient._on_task_cancellation_started
    )
    manager._on_task_cancelled = recipient._on_task_cancelled
    task = await manager.create_task(
        _params("idempotent-decline"),
        agent_name=recipient.did,
        creator_agent_id="did:test:creator",
    )
    signal = Signal(
        source="a2a.task_submitted",
        kind="task_submitted",
        mode=SignalMode.COGNITION,
        payload={"task_id": task.id},
        target_agent=recipient.did,
    )
    token = set_current_signal(signal)
    try:
        await manager.cancel_task(task.id, agent_name=recipient.did)
        # Model the monitor consuming the first attempt's exemption before the
        # tool retries after a lost response.
        recipient._a2a_self_declining_task_ids.discard(task.id)

        retried = await manager.cancel_task(task.id, agent_name=recipient.did)

        assert retried.status.state is TaskState.CANCELED
        assert task.id in recipient._a2a_self_declining_task_ids
    finally:
        reset_current_signal(token)
        await manager.close()


@pytest.mark.asyncio
async def test_legacy_metadata_cannot_supply_a_cancellation_receipt(tmp_path):
    """Rows without durable receipt columns cannot retain old metadata claims."""

    database = tmp_path / "legacy-forged-receipt.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE a2a_tasks (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                user_id TEXT,
                task_type TEXT NOT NULL,
                status TEXT DEFAULT 'submitted',
                message TEXT,
                artifacts TEXT DEFAULT '[]',
                history TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            INSERT INTO a2a_tasks (id, task_type, status, metadata)
            VALUES (?, ?, ?, ?)
            """,
            (
                "legacy-forged-receipt",
                "generic",
                "canceled",
                '{"cancellation_receipt":{"actor_agent_id":"did:test:evil",'
                '"reason":"forged","status_before":"working"}}',
            ),
        )

    manager = await create_task_manager(str(database))
    try:
        legacy = await manager.task_store._get_unscoped("legacy-forged-receipt")
        assert "cancellation_receipt" not in (legacy.metadata or {})
        with pytest.raises(
            TaskCancellationAuthorizationError,
            match="not authorized or task was not found",
        ):
            await manager.cancel_task(
                "legacy-forged-receipt",
                agent_name="did:test:evil",
            )
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_task_cannot_be_created_already_canceled(tmp_path):
    """Creation cannot bypass the authorized cancellation transition."""

    manager = await create_task_manager(str(tmp_path / "born-canceled.db"))
    try:
        with pytest.raises(ValueError, match="authorized transition"):
            await manager.task_store.save(
                Task(
                    id="born-canceled",
                    status=TaskStatus(state=TaskState.CANCELED),
                ),
                creator_agent_id="did:test:creator",
                recipient_agent_id="did:test:recipient",
            )

        assert await manager.task_store._get_unscoped("born-canceled") is None
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_lifecycle_save_strips_forged_cancellation_receipt(tmp_path):
    """Ordinary lifecycle persistence cannot mint a cancellation receipt."""

    manager = await create_task_manager(str(tmp_path / "forged-save-receipt.db"))
    try:
        task = await manager.create_task(
            _params("forged-save-receipt", metadata={"payload": "retained"}),
            agent_name="did:test:recipient",
            creator_agent_id="did:test:creator",
        )
        task.metadata["cancellation_receipt"] = {
            "actor_agent_id": "did:test:creator",
            "reason": "forged",
            "status_before": "working",
        }

        assert await manager.task_store.save_recipient_lifecycle(
            task,
            recipient_agent_id="did:test:recipient",
            expected_state=TaskState.SUBMITTED,
        ) is True

        persisted = await manager.task_store._get_unscoped("forged-save-receipt")
        assert persisted.metadata == {"payload": "retained"}
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_task_save_cannot_reassign_cancellation_authority(tmp_path):
    """A stale/replayed lifecycle write cannot mint a new delegate."""

    manager = await create_task_manager(str(tmp_path / "immutable-authority.db"))
    try:
        task = await manager.create_task(
            _params("immutable"),
            agent_name="did:test:recipient",
            creator_agent_id="did:test:creator",
        )
        with pytest.raises(ValueError, match="already exists"):
            await manager.task_store.save(
                task,
                creator_agent_id="did:test:stale-creator",
                recipient_agent_id="did:test:revoked-delegate",
            )

        stale = await _feature(manager, "did:test:stale-creator").cancel_task(
            "immutable"
        )
        revoked = await _feature(manager, "did:test:revoked-delegate").cancel_task(
            "immutable"
        )

        assert stale.status is ToolResultStatus.ERROR
        assert revoked.status is ToolResultStatus.ERROR
        assert (await manager.task_store._get_unscoped("immutable")).status.state is TaskState.SUBMITTED
        assert (
            await manager.cancel_task(
                "immutable",
                agent_name="did:test:creator",
                recipient_agent_id="did:test:recipient",
            )
        ).status.state is TaskState.CANCELED
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_cancel_task_terminal_state_is_unchanged(tmp_path):
    manager = await create_task_manager(str(tmp_path / "terminal.db"))
    try:
        task = await manager.create_task(
            _params("finished"),
            agent_name="did:test:recipient",
            creator_agent_id="did:test:creator",
        )
        await manager.update_status(
            task.id,
            TaskState.WORKING,
            agent_name="did:test:recipient",
            recipient_agent_id="did:test:recipient",
        )
        await manager.update_status(
            task.id,
            TaskState.COMPLETED,
            agent_name="did:test:recipient",
            recipient_agent_id="did:test:recipient",
        )

        result = await _feature(manager, "did:test:creator").cancel_task("finished")

        assert result.status is ToolResultStatus.ERROR
        unchanged = await manager.task_store._get_unscoped("finished")
        assert unchanged.status.state is TaskState.COMPLETED
        assert "cancellation_receipt" not in (unchanged.metadata or {})
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_foreign_terminal_and_absent_cancel_are_indistinguishable(tmp_path):
    manager = await create_task_manager(str(tmp_path / "terminal-privacy.db"))
    try:
        task = await manager.create_task(
            _params("foreign-finished"),
            agent_name="did:test:recipient",
            creator_agent_id="did:test:creator",
        )
        await manager.update_status(
            task.id,
            TaskState.WORKING,
            agent_name="did:test:recipient",
            recipient_agent_id="did:test:recipient",
        )
        await manager.update_status(
            task.id,
            TaskState.COMPLETED,
            agent_name="did:test:recipient",
            recipient_agent_id="did:test:recipient",
        )

        for task_id in ("foreign-finished", "absent-task"):
            with pytest.raises(
                TaskCancellationAuthorizationError,
                match="not authorized or task was not found",
            ):
                await manager.cancel_task(
                    task_id,
                    agent_name="did:test:foreign",
                )
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_cancel_task_authorization_and_transition_are_one_winner(tmp_path):
    """Creator and recipient racing cannot both write a cancellation receipt."""

    manager = await create_task_manager(str(tmp_path / "race.db"))
    try:
        await manager.create_task(
            _params("race"),
            agent_name="did:test:recipient",
            creator_agent_id="did:test:creator",
        )
        results = await asyncio.gather(
            _feature(manager, "did:test:creator").cancel_task("race", reason="creator"),
            _feature(manager, "did:test:recipient").cancel_task(
                "race", reason="recipient"
            ),
        )

        assert sorted(result.status.value for result in results) == ["error", "ok"]
        task = await manager.task_store._get_unscoped("race")
        assert task.status.state is TaskState.CANCELED
        receipt = task.metadata["cancellation_receipt"]
        assert (receipt["actor_agent_id"], receipt["reason"]) in {
            ("did:test:creator", "creator"),
            ("did:test:recipient", "recipient"),
        }
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_cancel_task_manager_requires_concrete_actor(tmp_path):
    manager = await create_task_manager(str(tmp_path / "missing-actor.db"))
    try:
        await manager.create_task(_params("owned"), agent_name="did:test:owner")
        with pytest.raises(TaskCancellationAuthorizationError, match="concrete"):
            await manager.cancel_task("owned", reason="anonymous")
        assert (await manager.task_store._get_unscoped("owned")).status.state is TaskState.SUBMITTED
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_respond_canceled_cannot_bypass_task_authority(tmp_path):
    manager = await create_task_manager(str(tmp_path / "respond-bypass.db"))
    try:
        await manager.create_task(
            _params("respond-protected"),
            agent_name="did:test:recipient",
            creator_agent_id="did:test:creator",
        )

        result = await _feature(manager, "did:test:unrelated").respond_to_a2a_task(
            "respond-protected",
            content="I decline someone else's task",
            state="canceled",
        )

        assert result.status is ToolResultStatus.ERROR
        assert "not found" in result.error
        assert (
            await manager.task_store._get_unscoped("respond-protected")
        ).status.state is TaskState.SUBMITTED
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_create_task_rejects_duplicate_id_without_mutating_owner_or_payload(
    tmp_path,
):
    manager = await create_task_manager(str(tmp_path / "duplicate.db"))
    try:
        await manager.create_task(
            _params("same-id", metadata={"payload": "original"}),
            agent_name="did:test:recipient-a",
            creator_agent_id="did:test:creator-a",
        )
        session_before = await manager.session_service.get_session(
            "session-same-id"
        )

        with pytest.raises(ValueError, match="already exists"):
            await manager.create_task(
                _params("same-id", metadata={"payload": "replacement"}),
                agent_name="did:test:recipient-b",
                creator_agent_id="did:test:creator-b",
            )

        original = await manager.task_store._get_unscoped("same-id")
        assert original.metadata["payload"] == "original"
        assert original.history[0].parts[0].text == "Do the work"
        session_after = await manager.session_service.get_session("session-same-id")
        assert session_after.events == session_before.events
        assert (
            await manager.cancel_task(
                "same-id",
                agent_name="did:test:creator-a",
                recipient_agent_id="did:test:recipient-a",
            )
        ).status.state is TaskState.CANCELED

        with pytest.raises(ValueError, match="already exists"):
            await manager.create_task(
                _params("same-id", metadata={"payload": "after-cancel"}),
                agent_name="did:test:recipient-b",
                creator_agent_id="did:test:creator-b",
            )

        canceled = await manager.task_store._get_unscoped("same-id")
        assert canceled.status.state is TaskState.CANCELED
        assert canceled.metadata["payload"] == "original"
    finally:
        await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("projection", ["session", "observability"])
async def test_create_task_projection_failure_does_not_report_durable_commit_as_failed(
    tmp_path,
    projection,
):
    manager = await create_task_manager(str(tmp_path / f"{projection}.db"))
    try:
        if projection == "session":
            manager.session_service.append_event = AsyncMock(
                side_effect=RuntimeError("session projection unavailable")
            )
        else:
            manager.observability_store.log_tool_call = AsyncMock(
                side_effect=RuntimeError("observability unavailable")
            )

        created = await manager.create_task(
            _params(f"accepted-{projection}"),
            agent_name="did:test:recipient",
            creator_agent_id="did:test:creator",
        )

        assert created.id == f"accepted-{projection}"
        result = await _feature(
            manager,
            "did:test:recipient",
        ).respond_to_a2a_task(created.id, "completed despite projection outage")
        assert result.status is ToolResultStatus.OK
        assert (await manager.task_store._get_unscoped(created.id)).status.state is TaskState.COMPLETED
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_async_internal_skill_task_is_owned_by_host_did(tmp_path):
    host_did = "did:test:host"
    manager = await create_task_manager(
        str(tmp_path / "internal-skill.db"), host_agent_id=host_did
    )

    class BlockingHandler:
        name = "feature-handler"

        async def handle_task(self, task):
            await asyncio.Event().wait()

        def get_skill_for_command(self, command):
            return None

    manager.register_agent(
        AgentCard(
            name="model_agent",
            url="/agents/model_agent",
            version="1.0.0",
            capabilities=AgentCapabilities(),
            skills=[
                AgentSkill(
                    id="slow_skill",
                    name="slow_skill",
                    description="Slow skill",
                )
            ],
        ),
        BlockingHandler(),
    )
    try:
        task = await manager.execute_skill(
            "model_agent",
            "slow_skill",
            {},
            sync=False,
        )

        result = await _feature(manager, host_did).cancel_task(task.id)

        assert result.status is ToolResultStatus.OK
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_stale_worker_save_cannot_overwrite_authorized_cancellation(tmp_path):
    manager = await create_task_manager(str(tmp_path / "stale-worker.db"))
    try:
        await manager.create_task(
            _params("stale-worker"),
            agent_name="did:test:recipient",
            creator_agent_id="did:test:creator",
        )
        stale_worker_copy = await manager.task_store._get_unscoped("stale-worker")
        await manager.cancel_task(
            "stale-worker",
            reason="withdrawn",
            agent_name="did:test:creator",
        )

        stale_worker_copy.status = TaskStatus(state=TaskState.COMPLETED)
        await manager.task_store.save_recipient_lifecycle(
            stale_worker_copy,
            recipient_agent_id="did:test:recipient",
            expected_state=TaskState.SUBMITTED,
        )

        persisted = await manager.task_store._get_unscoped("stale-worker")
        assert persisted.status.state is TaskState.CANCELED
        assert persisted.metadata["cancellation_receipt"]["reason"] == "withdrawn"
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_lifecycle_save_without_authority_cannot_create_task(tmp_path):
    manager = await create_task_manager(str(tmp_path / "update-only.db"))
    try:
        with pytest.raises(ValueError, match="save_recipient_lifecycle"):
            await manager.task_store.save(
                Task(
                    id="authority-less",
                    status=TaskStatus(state=TaskState.SUBMITTED),
                )
            )

        assert await manager.task_store._get_unscoped("authority-less") is None
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_generic_status_and_store_writes_cannot_cancel_task(tmp_path):
    manager = await create_task_manager(str(tmp_path / "canonical-cancel.db"))
    try:
        task = await manager.create_task(
            _params("canonical-cancel"),
            agent_name="did:test:recipient",
            creator_agent_id="did:test:creator",
        )

        with pytest.raises(
            TaskCancellationAuthorizationError, match="use cancel_task"
        ):
            await manager.update_status(
                task.id,
                TaskState.CANCELED,
                recipient_agent_id="did:test:recipient",
            )

        task.status = TaskStatus(state=TaskState.CANCELED)
        with pytest.raises(ValueError, match="cancel_if_authorized"):
            await manager.task_store.save_recipient_lifecycle(
                task,
                recipient_agent_id="did:test:recipient",
                expected_state=TaskState.SUBMITTED,
            )
        with pytest.raises(ValueError, match="cancel_if_authorized"):
            await manager.task_store.update_status(
                task.id,
                TaskStatus(state=TaskState.CANCELED),
                recipient_agent_id="did:test:recipient",
                expected_state=TaskState.SUBMITTED,
            )

        assert (await manager.task_store._get_unscoped(task.id)).status.state is TaskState.SUBMITTED
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_shutdown_cancellation_uses_authorized_host_identity(tmp_path):
    host_did = "did:test:host"
    manager = await create_task_manager(
        str(tmp_path / "shutdown-cancel.db"), host_agent_id=host_did
    )

    class BlockingHandler:
        name = "blocking"

        async def handle_task(self, task):
            await asyncio.Event().wait()

        def get_skill_for_command(self, command):
            return None

    manager.register_agent(
        AgentCard(
            name="shutdown-agent",
            url="/agents/shutdown-agent",
            version="1.0.0",
            capabilities=AgentCapabilities(),
            skills=[
                AgentSkill(
                    id="shutdown-skill",
                    name="shutdown-skill",
                    description="wait",
                )
            ],
        ),
        BlockingHandler(),
    )
    try:
        task = await manager.execute_skill(
            "shutdown-agent", "shutdown-skill", {}, sync=False
        )
        await manager.drain_execution_tasks(cancel=True)

        persisted = await manager.task_store._get_unscoped(task.id)
        assert persisted.status.state is TaskState.CANCELED
        assert persisted.metadata["cancellation_receipt"] == {
            "actor_agent_id": host_did,
            "reason": "Task canceled during shutdown",
            "status_before": "submitted",
        }
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_shutdown_cancels_worker_when_receipt_persistence_fails(tmp_path):
    manager = await create_task_manager(
        str(tmp_path / "shutdown-persistence-failure.db"),
        host_agent_id="did:test:host",
    )
    worker_started = asyncio.Event()
    worker_stopped = asyncio.Event()

    async def worker():
        worker_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            worker_stopped.set()

    task = manager._track_execution_task(
        worker(),
        "shutdown-task",
        "did:test:host",
    )
    await worker_started.wait()
    manager.cancel_task = AsyncMock(side_effect=RuntimeError("database unavailable"))

    await asyncio.wait_for(
        manager.drain_execution_tasks(cancel=True),
        timeout=0.2,
    )

    assert task.done()
    assert worker_stopped.is_set()
    assert manager._execution_tasks == set()
    await manager.close()


@pytest.mark.asyncio
async def test_async_handler_cancellation_emits_one_terminal_notification(tmp_path):
    host_did = "did:test:host"
    manager = await create_task_manager(
        str(tmp_path / "handler-cancel.db"), host_agent_id=host_did
    )

    class CancelingHandler:
        name = "canceling"

        async def handle_task(self, task):
            task.artifacts = [
                Artifact(name="partial", parts=[TextPart(text="partial output")])
            ]
            task.metadata["handler_state"] = "declined_after_partial_work"
            task.history.append(
                Message(role="agent", parts=[TextPart(text="Partial work retained")])
            )
            task.status = TaskStatus(
                state=TaskState.CANCELED,
                message=Message(
                    role="agent",
                    parts=[TextPart(text="Handler declined")],
                ),
            )
            return task

        def get_skill_for_command(self, command):
            return None

    manager.register_agent(
        AgentCard(
            name="canceling-agent",
            url="/agents/canceling-agent",
            version="1.0.0",
            capabilities=AgentCapabilities(),
            skills=[
                AgentSkill(
                    id="canceling-skill",
                    name="canceling-skill",
                    description="cancel",
                )
            ],
        ),
        CancelingHandler(),
    )
    completion = MagicMock()
    manager._on_task_complete = completion
    try:
        task = await manager.execute_skill(
            "canceling-agent", "canceling-skill", {}, sync=False
        )
        await manager.drain_execution_tasks()

        persisted = await manager.task_store._get_unscoped(task.id)
        assert persisted.status.state is TaskState.CANCELED
        assert persisted.artifacts[0].name == "partial"
        assert persisted.metadata["handler_state"] == "declined_after_partial_work"
        assert [part.text for message in persisted.history for part in message.parts] == [
            "Execute canceling-skill on canceling-agent",
            "Partial work retained",
            "Task canceled by did:test:host: Handler declined",
        ]
        completion.assert_called_once()
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_sync_handler_cancellation_runs_post_hook_with_durable_payload(tmp_path):
    host_did = "did:test:host"
    manager = await create_task_manager(
        str(tmp_path / "sync-handler-cancel.db"), host_agent_id=host_did
    )

    class CancelingHandler:
        name = "canceling"

        async def handle_task(self, task):
            task.artifacts = [
                Artifact(name="partial", parts=[TextPart(text="partial output")])
            ]
            task.metadata["handler_state"] = "canceled"
            task.status = TaskStatus(
                state=TaskState.CANCELED,
                message=Message(
                    role="agent",
                    parts=[TextPart(text="Handler declined")],
                ),
            )
            return task

        def get_skill_for_command(self, command):
            return None

    hooks = SimpleNamespace(
        execute_hooks=AsyncMock(
            return_value=HookOutput(permission_decision=PermissionDecision.ALLOW)
        ),
        execute_hooks_parallel=AsyncMock(),
    )
    manager.hooks_manager = hooks
    manager.register_agent(
        AgentCard(
            name="canceling-agent",
            url="/agents/canceling-agent",
            version="1.0.0",
            capabilities=AgentCapabilities(),
            skills=[
                AgentSkill(
                    id="canceling-skill",
                    name="canceling-skill",
                    description="cancel",
                )
            ],
        ),
        CancelingHandler(),
    )
    try:
        result = await manager.execute_skill(
            "canceling-agent", "canceling-skill", {}, sync=True
        )

        assert result.status.state is TaskState.CANCELED
        assert result.artifacts[0].name == "partial"
        assert result.metadata["handler_state"] == "canceled"
        hooks.execute_hooks_parallel.assert_awaited_once()
        assert hooks.execute_hooks_parallel.await_args.args[0] is HookEvent.POST_TOOL_USE
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_handler_cancellation_merges_concurrently_committed_payload(tmp_path):
    """A stale handler snapshot cannot erase data committed while it ran."""
    host_did = "did:test:host"
    manager = await create_task_manager(
        str(tmp_path / "handler-cancel-merge.db"), host_agent_id=host_did
    )
    entered = asyncio.Event()
    release = asyncio.Event()

    class CancelingHandler:
        name = "canceling"

        async def handle_task(self, task):
            entered.set()
            await release.wait()
            task.artifacts = [
                Artifact(name="handler", parts=[TextPart(text="partial")])
            ]
            task.metadata["handler_state"] = "declined"
            task.history.append(
                Message(role="agent", parts=[TextPart(text="handler history")])
            )
            task.status = TaskStatus(
                state=TaskState.CANCELED,
                message=Message(
                    role="agent",
                    parts=[TextPart(text="Handler declined")],
                ),
            )
            return task

        def get_skill_for_command(self, command):
            return None

    manager.register_agent(
        AgentCard(
            name="canceling-agent",
            url="/agents/canceling-agent",
            version="1.0.0",
            capabilities=AgentCapabilities(),
            skills=[
                AgentSkill(
                    id="canceling-skill",
                    name="canceling-skill",
                    description="cancel",
                )
            ],
        ),
        CancelingHandler(),
    )
    try:
        execution = asyncio.create_task(
            manager.execute_skill(
                "canceling-agent", "canceling-skill", {}, sync=True
            )
        )
        await entered.wait()
        pending = (await manager.get_pending_tasks(
            recipient_agent_id=host_did,
        ))[0]
        concurrent = await manager.task_store._get_unscoped(pending.id)
        concurrent.artifacts = [
            Artifact(name="concurrent", parts=[TextPart(text="keep")])
        ]
        concurrent.metadata["concurrent_state"] = "committed"
        concurrent.history.append(
            Message(role="agent", parts=[TextPart(text="concurrent history")])
        )
        assert await manager.task_store.save_recipient_lifecycle(
            concurrent,
            recipient_agent_id=host_did,
            expected_state=TaskState.SUBMITTED,
        ) is True

        release.set()
        result = await execution

        assert result.status.state is TaskState.CANCELED
        assert {artifact.name for artifact in result.artifacts} == {
            "concurrent",
            "handler",
        }
        assert result.metadata["concurrent_state"] == "committed"
        assert result.metadata["handler_state"] == "declined"
        history_text = [
            part.text for message in result.history for part in message.parts
        ]
        assert "concurrent history" in history_text
        assert "handler history" in history_text
    finally:
        release.set()
        await manager.close()


@pytest.mark.asyncio
async def test_sync_handler_returns_concurrent_winning_cancellation(tmp_path):
    """A valid external cancellation wins without making sync execution fail."""
    host_did = "did:test:host"
    manager = await create_task_manager(
        str(tmp_path / "sync-cancel-winner.db"), host_agent_id=host_did
    )
    entered = asyncio.Event()
    release = asyncio.Event()

    class CancelingHandler:
        name = "canceling"

        async def handle_task(self, task):
            entered.set()
            await release.wait()
            task.status = TaskStatus(
                state=TaskState.CANCELED,
                message=Message(
                    role="agent",
                    parts=[TextPart(text="Handler also canceled")],
                ),
            )
            return task

        def get_skill_for_command(self, command):
            return None

    manager.register_agent(
        AgentCard(
            name="canceling-agent",
            url="/agents/canceling-agent",
            version="1.0.0",
            capabilities=AgentCapabilities(),
            skills=[
                AgentSkill(
                    id="canceling-skill",
                    name="canceling-skill",
                    description="cancel",
                )
            ],
        ),
        CancelingHandler(),
    )
    try:
        execution = asyncio.create_task(
            manager.execute_skill(
                "canceling-agent", "canceling-skill", {}, sync=True
            )
        )
        await entered.wait()
        pending = (await manager.get_pending_tasks(
            recipient_agent_id=host_did,
        ))[0]
        await manager.cancel_task(
            pending.id,
            reason="external winner",
            agent_name=host_did,
        )

        release.set()
        result = await execution

        assert result.status.state is TaskState.CANCELED
        assert result.metadata["cancellation_receipt"]["reason"] == (
            "external winner"
        )
    finally:
        release.set()
        await manager.close()


@pytest.mark.asyncio
async def test_async_canceling_handler_preserves_concurrent_completed_winner(tmp_path):
    """A stale handler cancellation cannot rewrite completed state as failed."""
    host_did = "did:test:host"
    manager = await create_task_manager(
        str(tmp_path / "async-completed-winner.db"), host_agent_id=host_did
    )
    entered = asyncio.Event()
    release = asyncio.Event()

    class CancelingHandler:
        name = "canceling"

        async def handle_task(self, task):
            entered.set()
            await release.wait()
            task.status = TaskStatus(
                state=TaskState.CANCELED,
                message=Message(
                    role="agent",
                    parts=[TextPart(text="stale cancellation")],
                ),
            )
            return task

        def get_skill_for_command(self, command):
            return None

    manager.register_agent(
        AgentCard(
            name="canceling-agent",
            url="/agents/canceling-agent",
            version="1.0.0",
            capabilities=AgentCapabilities(),
            skills=[
                AgentSkill(
                    id="canceling-skill",
                    name="canceling-skill",
                    description="cancel",
                )
            ],
        ),
        CancelingHandler(),
    )
    try:
        await manager.execute_skill(
            "canceling-agent", "canceling-skill", {}, sync=False
        )
        await entered.wait()
        pending = (await manager.get_pending_tasks(
            recipient_agent_id=host_did,
        ))[0]
        winning = await manager.task_store._get_unscoped(pending.id)
        winning.status = TaskStatus(
            state=TaskState.COMPLETED,
            message=Message(
                role="agent",
                parts=[TextPart(text="concurrent completion")],
            ),
        )
        assert await manager.task_store.save_recipient_lifecycle(
            winning,
            recipient_agent_id=host_did,
            expected_state=TaskState.SUBMITTED,
        ) is True

        release.set()
        await manager.drain_execution_tasks()

        persisted = await manager.task_store._get_unscoped(pending.id)
        assert persisted.status.state is TaskState.COMPLETED
        assert persisted.status.message.parts[0].text == "concurrent completion"
    finally:
        release.set()
        await manager.close()


@pytest.mark.asyncio
async def test_creator_routes_cancellation_to_durable_recipient(monkeypatch):
    from kestrel_sovereign.a2a import outbound_store

    actor = SimpleNamespace(did="did:test:creator", identity=None, features={})
    peers = PeersFeature(actor)
    peers._db = object()
    peers._outbound_route_store_ready = True
    peers._own_name = "creator"
    requester = PeerRequester(actor.did, object())
    peer = PeerIdentity(
        agent_id="did:test:recipient",
        slug="recipient",
        routing_key="recipient-route",
        name="Recipient",
    )
    router = SimpleNamespace(
        cancel_a2a_task=AsyncMock(
            return_value={
                "id": "outbound-task",
                "status": "canceled",
                "cancellation_receipt": {
                    "status_before": "working",
                    "reason": "durable original reason",
                },
            }
        )
    )
    monkeypatch.setattr(
        outbound_store,
        "get_outbound_task",
        AsyncMock(
            return_value=SimpleNamespace(
                recipient="Recipient",
                recipient_agent_id="did:test:recipient",
                route_state=outbound_store.ROUTE_STATE_ROUTABLE,
            )
        ),
    )
    terminal_stamp = AsyncMock(return_value=1)
    monkeypatch.setattr(
        outbound_store,
        "update_outbound_terminal_state",
        terminal_stamp,
    )
    peers._resolve_retained_automatic_peer = AsyncMock(
        return_value=(router, requester, peer)
    )
    peers._maybe_sign_outbound = MagicMock()
    actor.features["PeersFeature"] = peers

    local_manager = MagicMock()
    # A shared PostgreSQL TaskStore can see the recipient's task row.  The
    # durable sender-owned route must win before that local visibility is used.
    local_manager.cancel_task = AsyncMock()
    local_manager.is_task_recipient = AsyncMock(return_value=False)
    feature = TaskFeature(actor)
    feature.set_task_manager(local_manager)

    result = await feature.cancel_task("outbound-task", reason="withdrawn")

    assert result.status is ToolResultStatus.OK
    assert result.data["reason"] == "durable original reason"
    payload = router.cancel_a2a_task.await_args.args[3]
    assert payload["metadata"] == {
        "sender": "creator",
        "a2a_verb": "cancel_task",
    }
    assert payload["reason"] == "withdrawn"
    peers._maybe_sign_outbound.assert_called_once_with(
        payload,
        task_id="outbound-task",
        sess_id="a2a-cancel:outbound-task",
        message="withdrawn",
    )
    router.cancel_a2a_task.assert_awaited_once_with(
        requester, peer, "outbound-task", payload
    )
    local_manager.cancel_task.assert_not_awaited()
    terminal_stamp.assert_awaited_once_with(
        peers._db,
        agent_id=actor.did,
        task_id="outbound-task",
        terminal_state="canceled",
    )


@pytest.mark.asyncio
async def test_local_router_uses_process_local_cancel_capability_without_http():
    local_cancel = AsyncMock(
        return_value={
            "id": "local-task",
            "status": "canceled",
            "cancellation_receipt": {"status_before": "submitted"},
        }
    )
    client_factory = MagicMock()
    router = LocalHostPeerDirectory(
        "http://local-host",
        client_factory=client_factory,
        local_cancel=local_cancel,
    )
    router._directory_entries = AsyncMock(
        return_value=[
            PeerIdentity(
                agent_id="did:test:recipient",
                slug="recipient",
                routing_key="recipient",
            )
        ]
    )
    requester = PeerRequester("did:test:creator", object())
    peer = PeerIdentity(
        agent_id="did:test:recipient",
        slug="recipient",
        routing_key="recipient",
    )
    payload = {
        "reason": "pre-ceremony withdrawal",
        "metadata": {"a2a_verb": "cancel_task"},
    }

    result = await router.cancel_a2a_task(
        requester, peer, "local-task", payload
    )

    assert result["status"] == "canceled"
    local_cancel.assert_awaited_once_with(
        requester, peer, "local-task", payload
    )
    client_factory.assert_not_called()


@pytest.mark.asyncio
async def test_manager_attests_pre_ceremony_local_cancellation():
    manager = AgentManager()
    sender = SimpleNamespace(
        did="did:test:creator",
        agent_id="did:test:creator",
        identity=None,
    )
    current = SimpleNamespace(id="local-task")
    canceled = SimpleNamespace(
        id="local-task",
        status=SimpleNamespace(state=TaskState.CANCELED),
        metadata={
            "cancellation_receipt": {
                "actor_agent_id": sender.did,
                "status_before": "submitted",
            }
        },
    )
    task_manager = SimpleNamespace(
        get_task_for_creator=AsyncMock(return_value=current),
        cancel_task=AsyncMock(return_value=canceled),
    )
    recipient = SimpleNamespace(
        did="did:test:recipient",
        agent_id="did:test:recipient",
        task_manager=task_manager,
    )
    manager._register_agent("creator", sender)
    manager._register_agent("recipient", recipient)
    sender_requester = PeerRequester(sender.did, object())
    recipient_requester = PeerRequester(recipient.did, object())
    sender_router = SimpleNamespace()
    recipient_router = SimpleNamespace(
        authorize_inbound_sender=AsyncMock(return_value=True)
    )
    manager.install_a2a_hosted_policy(
        sender,
        resolver=object(),
        authorizer=object(),
        router=sender_router,
        requester=sender_requester,
    )
    manager.install_a2a_hosted_policy(
        recipient,
        resolver=object(),
        authorizer=object(),
        router=recipient_router,
        requester=recipient_requester,
    )
    peer = PeerIdentity(
        agent_id=recipient.did,
        slug="recipient",
        routing_key="recipient",
    )

    result = await manager.cancel_host_attested_local_a2a_task(
        sender=sender,
        requester=sender_requester,
        peer=peer,
        task_id="local-task",
        payload={"reason": "pre-ceremony withdrawal"},
    )

    assert result["status"] == "canceled"
    recipient_router.authorize_inbound_sender.assert_awaited_once_with(
        recipient_requester,
        sender.did,
    )
    task_manager.cancel_task.assert_awaited_once_with(
        "local-task",
        reason="pre-ceremony withdrawal",
        agent_name=sender.did,
        recipient_agent_id=recipient.did,
    )


@pytest.mark.asyncio
async def test_manager_attested_local_cancellation_transitions_live_task(tmp_path):
    task_manager = await create_task_manager(str(tmp_path / "local-live.db"))
    manager = AgentManager()
    sender = SimpleNamespace(
        did="did:test:creator",
        agent_id="did:test:creator",
        identity=None,
    )
    recipient = SimpleNamespace(
        did="did:test:recipient",
        agent_id="did:test:recipient",
        task_manager=task_manager,
    )
    manager._register_agent("creator", sender)
    manager._register_agent("recipient", recipient)
    sender_requester = PeerRequester(sender.did, object())
    recipient_requester = PeerRequester(recipient.did, object())
    manager.install_a2a_hosted_policy(
        sender,
        resolver=object(),
        authorizer=object(),
        router=SimpleNamespace(),
        requester=sender_requester,
    )
    manager.install_a2a_hosted_policy(
        recipient,
        resolver=object(),
        authorizer=object(),
        router=SimpleNamespace(
            authorize_inbound_sender=AsyncMock(return_value=True)
        ),
        requester=recipient_requester,
    )
    try:
        task = await task_manager.create_task(
            _params("live-local-cancel"),
            agent_name=recipient.did,
            creator_agent_id=sender.did,
        )

        result = await manager.cancel_host_attested_local_a2a_task(
            sender=sender,
            requester=sender_requester,
            peer=PeerIdentity(
                agent_id=recipient.did,
                slug="recipient",
                routing_key="recipient",
            ),
            task_id=task.id,
            payload={"reason": "creator withdrew live task"},
        )

        assert result["status"] == "canceled"
        persisted = await task_manager.task_store._get_unscoped(task.id)
        assert persisted.status.state is TaskState.CANCELED
        assert persisted.metadata["cancellation_receipt"]["actor_agent_id"] == (
            sender.did
        )
    finally:
        await task_manager.close()


@pytest.mark.asyncio
async def test_manager_local_cancel_rejects_forged_requester_handle():
    manager = AgentManager()
    sender = SimpleNamespace(
        did="did:test:creator",
        agent_id="did:test:creator",
        identity=None,
    )
    recipient = SimpleNamespace(
        did="did:test:recipient",
        agent_id="did:test:recipient",
        task_manager=SimpleNamespace(),
    )
    manager._register_agent("creator", sender)
    manager._register_agent("recipient", recipient)
    trusted = PeerRequester(sender.did, object())
    manager.install_a2a_hosted_policy(
        sender,
        resolver=object(),
        authorizer=object(),
        router=SimpleNamespace(),
        requester=trusted,
    )
    manager.install_a2a_hosted_policy(
        recipient,
        resolver=object(),
        authorizer=object(),
        router=SimpleNamespace(authorize_inbound_sender=AsyncMock(return_value=True)),
        requester=PeerRequester(recipient.did, object()),
    )

    with pytest.raises(PeerAccessDeniedError):
        await manager.cancel_host_attested_local_a2a_task(
            sender=sender,
            requester=PeerRequester(sender.did, object()),
            peer=PeerIdentity(
                agent_id=recipient.did,
                slug="recipient",
                routing_key="recipient",
            ),
            task_id="local-task",
            payload={"reason": "forged handle"},
        )


@pytest.mark.asyncio
async def test_respond_canceled_uses_outbound_route_before_shared_task_row():
    """The response alias must preserve sender-side routing and peer scope."""

    routed_result = MagicMock(status=ToolResultStatus.OK)
    route_feature = SimpleNamespace(
        cancel_outbound_task=AsyncMock(return_value=routed_result)
    )
    actor = SimpleNamespace(
        did="did:test:creator",
        features={"PeersFeature": route_feature},
    )
    visible_recipient_row = Task(
        id="outbound-task",
        status=TaskStatus(state=TaskState.WORKING),
    )
    local_manager = MagicMock(
        get_task_for_recipient=AsyncMock(return_value=visible_recipient_row),
        is_task_recipient=AsyncMock(return_value=False),
        cancel_task=AsyncMock(),
    )
    feature = TaskFeature(actor)
    feature.set_task_manager(local_manager)

    result = await feature.respond_to_a2a_task(
        "outbound-task",
        content="withdrawn through response alias",
        state="canceled",
    )

    assert result is routed_result
    route_feature.cancel_outbound_task.assert_awaited_once_with(
        "outbound-task",
        reason="withdrawn through response alias",
        local_recipient_match=False,
    )
    local_manager.cancel_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_respond_canceled_routes_outbound_when_local_task_row_is_absent():
    """Per-agent SQLite keeps sender audit and recipient rows in different DBs."""

    routed_result = MagicMock(status=ToolResultStatus.OK)
    route_feature = SimpleNamespace(
        cancel_outbound_task=AsyncMock(return_value=routed_result)
    )
    actor = SimpleNamespace(
        did="did:test:creator",
        features={"PeersFeature": route_feature},
    )
    local_manager = MagicMock(
        get_task=AsyncMock(return_value=None),
        is_task_recipient=AsyncMock(return_value=False),
        cancel_task=AsyncMock(),
    )
    feature = TaskFeature(actor)
    feature.set_task_manager(local_manager)

    result = await feature.respond_to_a2a_task(
        "outbound-only-task",
        content="withdrawn from sender-side storage",
        state="canceled",
    )

    assert result is routed_result
    route_feature.cancel_outbound_task.assert_awaited_once_with(
        "outbound-only-task",
        reason="withdrawn from sender-side storage",
        local_recipient_match=False,
    )
    local_manager.get_task.assert_not_awaited()
    local_manager.cancel_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_database_fence_blocks_legacy_writer_resurrecting_canceled_task(
    tmp_path,
):
    """An unconditional pre-upgrade UPDATE cannot undo committed cancellation."""

    manager = await create_task_manager(str(tmp_path / "legacy-writer-fence.db"))
    try:
        await manager.create_task(
            _params("terminal-cancel"),
            agent_name="did:test:recipient",
            creator_agent_id="did:test:creator",
        )
        await manager.cancel_task(
            "terminal-cancel",
            agent_name="did:test:creator",
        )

        with pytest.raises(Exception, match="terminal A2A task cannot be replaced"):
            await manager.task_store.backend.execute(
                "UPDATE a2a_tasks SET status = 'completed' WHERE id = ?",
                ("terminal-cancel",),
            )

        assert (
            await manager.get_task_for_creator(
                "terminal-cancel", "did:test:creator"
            )
        ).status.state is TaskState.CANCELED
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_database_fence_blocks_legacy_replace_of_canceled_task(tmp_path):
    """SQLite REPLACE cannot delete and recreate a canceled authority row."""

    manager = await create_task_manager(str(tmp_path / "legacy-replace-fence.db"))
    try:
        await manager.create_task(
            _params("terminal-replace"),
            agent_name="did:test:recipient",
            creator_agent_id="did:test:creator",
        )
        await manager.cancel_task(
            "terminal-replace",
            agent_name="did:test:creator",
        )

        with pytest.raises(Exception, match="canceled A2A task is terminal"):
            await manager.task_store.backend.execute(
                """
                INSERT OR REPLACE INTO a2a_tasks
                    (id, task_type, status, creator_agent_id, recipient_agent_id)
                VALUES (?, ?, 'completed', ?, ?)
                """,
                (
                    "terminal-replace",
                    "generic",
                    "did:test:replacement-creator",
                    "did:test:replacement-recipient",
                ),
            )

        canceled = await manager.get_task_for_creator(
            "terminal-replace",
            "did:test:creator",
        )
        assert canceled is not None
        assert canceled.status.state is TaskState.CANCELED
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_database_fence_blocks_all_mutation_of_canceled_task(tmp_path):
    """A mixed-version writer cannot alter payload while retaining CANCELED."""

    manager = await create_task_manager(str(tmp_path / "canceled-write-fence.db"))
    try:
        await manager.create_task(
            _params("immutable-cancel"),
            agent_name="did:test:recipient",
            creator_agent_id="did:test:creator",
        )
        await manager.cancel_task(
            "immutable-cancel",
            agent_name="did:test:creator",
        )

        with pytest.raises(Exception, match="terminal A2A task cannot be replaced"):
            await manager.task_store.backend.execute(
                "UPDATE a2a_tasks SET metadata = ? WHERE id = ?",
                ('{"legacy":"overwrite"}', "immutable-cancel"),
            )

        assert "legacy" not in (
            (
                await manager.get_task_for_creator(
                    "immutable-cancel", "did:test:creator"
                )
            ).metadata
            or {}
        )
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_database_fence_rejects_legacy_live_insert_without_authority(
    tmp_path,
):
    """Late old-code inserts cannot recreate uncancellable live work."""

    manager = await create_task_manager(str(tmp_path / "live-authority-fence.db"))
    try:
        with pytest.raises(Exception, match="requires durable authority"):
            await manager.task_store.backend.execute(
                """
                INSERT INTO a2a_tasks (id, task_type, status)
                VALUES (?, ?, 'submitted')
                """,
                ("late-legacy-live", "generic"),
            )

        assert (
            await manager.get_task_for_creator(
                "late-legacy-live", "did:test:creator"
            )
            is None
        )
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_database_fence_rejects_legacy_terminal_replace_without_authority(
    tmp_path,
):
    """A late pre-authority worker cannot erase principals on completion."""

    manager = await create_task_manager(str(tmp_path / "terminal-authority-fence.db"))
    try:
        await manager.create_task(
            _params("late-legacy-terminal"),
            agent_name="did:test:recipient",
            creator_agent_id="did:test:creator",
        )

        with pytest.raises(Exception, match="requires durable authority"):
            await manager.task_store.backend.execute(
                """
                INSERT OR REPLACE INTO a2a_tasks
                    (id, session_id, user_id, task_type, status, message,
                     artifacts, history, metadata, updated_at)
                VALUES (?, ?, ?, ?, 'completed', ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    "late-legacy-terminal",
                    None,
                    None,
                    "generic",
                    None,
                    "[]",
                    "[]",
                    "{}",
                ),
            )

        persisted = await manager.task_store._get_unscoped("late-legacy-terminal")
        assert persisted is not None
        assert persisted.status.state is TaskState.SUBMITTED
        assert await manager.get_task_for_creator(
            "late-legacy-terminal",
            "did:test:creator",
        ) is not None
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_postgres_cancel_locks_only_an_authorized_principal_row():
    """A foreign task ID cannot be used as a cross-principal lock primitive."""

    @asynccontextmanager
    async def transaction():
        yield

    backend = SimpleNamespace(
        backend_type="postgres",
        transaction=transaction,
        fetch_one=AsyncMock(return_value=None),
        execute=AsyncMock(return_value=0),
    )
    store = TaskStore(backend)

    result = await store.cancel_if_authorized(
        "foreign-task",
        actor_agent_id="did:test:actor",
        expected_recipient_agent_id="did:test:expected-recipient",
    )

    assert result is None
    query, values = backend.fetch_one.await_args.args
    normalized = " ".join(query.split())
    assert "(creator_agent_id = ? OR recipient_agent_id = ?)" in normalized
    assert "AND recipient_agent_id = ?" in normalized
    assert normalized.endswith("FOR UPDATE")
    assert values == (
        "foreign-task",
        "did:test:actor",
        "did:test:actor",
        "did:test:expected-recipient",
    )
    backend.execute.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_database_fence_blocks_legacy_writer_on_available_backends(db_backend):
    store = TaskStore(db_backend)
    await store.initialize()
    task_id = f"terminal-cancel-{uuid4().hex}"
    try:
        await store.create(
            Task(id=task_id, status=TaskStatus(state=TaskState.SUBMITTED)),
            creator_agent_id="did:test:creator",
            recipient_agent_id="did:test:recipient",
        )
        canceled = await store.cancel_if_authorized(
            task_id,
            actor_agent_id="did:test:creator",
        )
        assert canceled is not None

        with pytest.raises(Exception, match="terminal A2A task cannot be replaced"):
            await db_backend.execute(
                "UPDATE a2a_tasks SET status = 'completed' WHERE id = ?",
                (task_id,),
            )

        assert (await store._get_unscoped(task_id)).status.state is TaskState.CANCELED
    finally:
        await db_backend.execute("DELETE FROM a2a_tasks WHERE id = ?", (task_id,))


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("with_payload", [False, True])
async def test_cancel_readback_failure_rolls_back_transition_on_available_backends(
    db_backend,
    monkeypatch,
    with_payload,
):
    store = TaskStore(db_backend)
    await store.initialize()
    task_id = f"cancel-readback-rollback-{uuid4().hex}"
    try:
        submitted = Task(
            id=task_id,
            status=TaskStatus(state=TaskState.SUBMITTED),
        )
        await store.create(
            submitted,
            creator_agent_id="did:test:creator",
            recipient_agent_id="did:test:recipient",
        )
        payload = (
            submitted.model_copy(
                update={"status": TaskStatus(state=TaskState.CANCELED)}
            )
            if with_payload
            else None
        )
        fetch_one = db_backend.fetch_one

        async def fail_canonical_read(sql, params=()):
            if "SELECT * FROM a2a_tasks" in sql:
                raise RuntimeError("injected in-transaction readback failure")
            return await fetch_one(sql, params)

        monkeypatch.setattr(db_backend, "fetch_one", fail_canonical_read)
        with pytest.raises(Exception, match="in-transaction readback failure"):
            await store.cancel_if_authorized(
                task_id,
                actor_agent_id="did:test:creator",
                task_payload=payload,
            )
        monkeypatch.setattr(db_backend, "fetch_one", fetch_one)

        persisted = await store._get_unscoped(task_id)
        assert persisted is not None
        assert persisted.status.state is TaskState.SUBMITTED
    finally:
        await db_backend.execute("DELETE FROM a2a_tasks WHERE id = ?", (task_id,))


@pytest.mark.asyncio
async def test_postgres_initialization_installs_terminal_lifecycle_trigger():
    @asynccontextmanager
    async def transaction():
        yield

    backend = SimpleNamespace(
        backend_type="postgres",
        execute_script=AsyncMock(),
        execute=AsyncMock(return_value=0),
        fetch_one=AsyncMock(return_value=(False,)),
        transaction=transaction,
    )

    await TaskStore(backend).initialize()

    scripts = "\n".join(
        call.args[0] for call in backend.execute_script.await_args_list
    )
    assert (
        "CREATE OR REPLACE FUNCTION a2a_tasks_enforce_authority_fence_v4"
        in scripts
    )
    assert "OLD.status IN ('completed', 'failed', 'canceled')" in scripts
    assert "terminal A2A task cannot be replaced" in scripts
    assert "IF TG_OP = 'INSERT'" in scripts
    assert "A2A task requires durable authority" in scripts
    assert "live A2A task requires durable authority" in scripts
    assert "CREATE TRIGGER a2a_tasks_authority_fence_v4" in scripts
    assert "EXECUTE FUNCTION a2a_tasks_enforce_authority_fence_v4()" in scripts
    statements = "\n".join(
        call.args[0] for call in backend.execute.await_args_list
    )
    assert "ADD COLUMN IF NOT EXISTS terminal_operation_id TEXT" in statements


@pytest.mark.asyncio
async def test_postgres_cancellation_schema_reprobes_under_advisory_lock():
    events: list[str] = []
    schema_probes: list[str] = []

    @asynccontextmanager
    async def transaction():
        events.append("transaction-enter")
        try:
            yield
        finally:
            events.append("transaction-exit")

    async def execute(query, _params=()):
        normalized = " ".join(query.split())
        if "pg_advisory_xact_lock" in normalized:
            events.append("advisory-lock")
        elif "CREATE INDEX" in normalized:
            events.append("index-ddl")
        return 0

    async def fetch_one(query, _params=()):
        events.append("schema-reprobe")
        schema_probes.append(query)
        return (False,)

    async def execute_script(script):
        if "a2a_tasks_enforce_authority_fence" in script:
            events.append("fence-ddl")

    backend = SimpleNamespace(
        backend_type="postgres",
        execute_script=AsyncMock(side_effect=execute_script),
        execute=AsyncMock(side_effect=execute),
        fetch_one=AsyncMock(side_effect=fetch_one),
        transaction=transaction,
    )

    await TaskStore(backend).initialize()

    lock_index = events.index("advisory-lock")
    probe_index = events.index("schema-reprobe")
    fence_index = events.index("fence-ddl")
    assert events.index("transaction-enter") < lock_index < probe_index
    assert probe_index < fence_index < events.index("transaction-exit")
    assert "terminal_operation_id" in schema_probes[0]
    assert "COUNT(*) = 7" in schema_probes[0]


@pytest.mark.asyncio
async def test_postgres_cancellation_schema_waiter_skips_completed_ddl():
    @asynccontextmanager
    async def transaction():
        yield

    backend = SimpleNamespace(
        backend_type="postgres",
        execute_script=AsyncMock(),
        execute=AsyncMock(return_value=0),
        fetch_one=AsyncMock(return_value=(True,)),
        transaction=transaction,
    )

    await TaskStore(backend).initialize()

    scripts = "\n".join(
        call.args[0] for call in backend.execute_script.await_args_list
    )
    assert "a2a_tasks_enforce_authority_fence" not in scripts
    statements = "\n".join(
        call.args[0] for call in backend.execute.await_args_list
    )
    assert "pg_advisory_xact_lock" in statements
    assert "CREATE INDEX" not in statements


@pytest.mark.asyncio
async def test_postgres_v3_terminal_fence_is_not_authority_schema_ready():
    """A v3 terminal fence must not suppress the all-insert authority upgrade."""

    @asynccontextmanager
    async def transaction():
        yield

    async def fetch_one(query, _params=()):
        # Model a database with every older object but without the v4 function
        # and trigger that fence authority-less terminal inserts.
        has_terminal_function = "a2a_tasks_enforce_authority_fence_v4" in query
        has_terminal_trigger = "a2a_tasks_authority_fence_v4" in query
        has_terminal_binding = "procedure.oid = trigger.tgfoid" in query
        return (
            not (
                has_terminal_function
                and has_terminal_trigger
                and has_terminal_binding
            ),
        )

    backend = SimpleNamespace(
        backend_type="postgres",
        execute_script=AsyncMock(),
        execute=AsyncMock(return_value=0),
        fetch_one=AsyncMock(side_effect=fetch_one),
        transaction=transaction,
    )

    await TaskStore(backend).initialize()

    scripts = "\n".join(
        call.args[0] for call in backend.execute_script.await_args_list
    )
    assert "a2a_tasks_enforce_authority_fence_v4" in scripts
    assert "a2a_tasks_authority_fence_v4" in scripts


@pytest.mark.asyncio
async def test_outbound_cancel_reports_lifecycle_conflict_not_transport(monkeypatch):
    from kestrel_sovereign.a2a import outbound_store

    actor = SimpleNamespace(did="did:test:creator", identity=None)
    peers = PeersFeature(actor)
    peers._db = object()
    peers._outbound_route_store_ready = True
    peers._own_name = "creator"
    monkeypatch.setattr(
        outbound_store,
        "get_outbound_task",
        AsyncMock(
            return_value=SimpleNamespace(
                recipient="Recipient",
                recipient_agent_id="did:test:recipient",
                route_state=outbound_store.ROUTE_STATE_ROUTABLE,
            )
        ),
    )
    router = SimpleNamespace(
        cancel_a2a_task=AsyncMock(
            side_effect=PeerTaskConflictError("already terminal")
        )
    )
    peers._resolve_retained_automatic_peer = AsyncMock(
        return_value=(
            router,
            PeerRequester(actor.did, object()),
            PeerIdentity(
                agent_id="did:test:recipient",
                slug="recipient",
                routing_key="recipient-route",
            ),
        )
    )
    peers._maybe_sign_outbound = MagicMock()

    result = await peers.cancel_outbound_task("outbound-task")

    assert result.status is ToolResultStatus.ERROR
    assert result.data["error_type"] == "lifecycle_conflict"
    assert "terminal state" in result.error


@pytest.mark.asyncio
async def test_outbound_cancel_without_route_store_cannot_fall_back_to_shared_row():
    actor = SimpleNamespace(did="did:test:creator", identity=None, features={})
    peers = PeersFeature(actor)
    peers._db = None
    peers._outbound_route_store_ready = False
    peers._own_name = "creator"
    actor.features["PeersFeature"] = peers
    shared_manager = MagicMock(
        is_task_recipient=AsyncMock(return_value=False),
        cancel_task=AsyncMock(
            return_value=Task(
                id="shared-recipient-row",
                status=TaskStatus(state=TaskState.CANCELED),
                metadata={
                    "cancellation_receipt": {
                        "actor_agent_id": actor.did,
                        "reason": None,
                        "status_before": "working",
                    }
                },
            )
        ),
    )
    tasks = TaskFeature(actor)
    tasks.set_task_manager(shared_manager)

    result = await tasks.cancel_task(
        "shared-recipient-row",
    )

    assert result.status is ToolResultStatus.ERROR
    assert "route store is unavailable" in result.error
    shared_manager.cancel_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_creator_cancel_without_peers_feature_fails_closed():
    actor = SimpleNamespace(did="did:test:creator", features={})
    manager = MagicMock(
        is_task_recipient=AsyncMock(return_value=False),
        cancel_task=AsyncMock(),
    )
    feature = TaskFeature(actor)
    feature.set_task_manager(manager)

    result = await feature.cancel_task("remote-task")

    assert result.status is ToolResultStatus.ERROR
    assert "not found" in result.error
    manager.cancel_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancel_without_peers_allows_exact_recipient_local_task():
    actor = SimpleNamespace(did="did:test:recipient", features={})
    canceled = Task(
        id="inbound-task",
        status=TaskStatus(state=TaskState.CANCELED),
        metadata={
            "cancellation_receipt": {
                "status_before": "working",
                "reason": "declined",
            }
        },
    )
    manager = MagicMock(
        is_task_recipient=AsyncMock(return_value=True),
        cancel_task=AsyncMock(return_value=canceled),
    )
    feature = TaskFeature(actor)
    feature.set_task_manager(manager)

    result = await feature.cancel_task("inbound-task", reason="declined")

    assert result.status is ToolResultStatus.OK
    manager.cancel_task.assert_awaited_once_with(
        "inbound-task",
        reason="declined",
        agent_name=actor.did,
        recipient_agent_id=actor.did,
    )


@pytest.mark.asyncio
async def test_cancel_rejects_inbound_outbound_task_id_collision(monkeypatch):
    from kestrel_sovereign.a2a import outbound_store

    actor = SimpleNamespace(did="did:test:agent", identity=None, features={})
    peers = PeersFeature(actor)
    peers._db = object()
    peers._outbound_route_store_ready = True
    peers._own_name = "agent"
    monkeypatch.setattr(
        outbound_store,
        "get_outbound_task",
        AsyncMock(
            return_value=SimpleNamespace(
                recipient="Other",
                recipient_agent_id="did:test:other",
                route_state=outbound_store.ROUTE_STATE_ROUTABLE,
            )
        ),
    )
    peers._resolve_retained_automatic_peer = AsyncMock()
    actor.features["PeersFeature"] = peers
    local_manager = MagicMock(
        is_task_recipient=AsyncMock(return_value=True),
        cancel_task=AsyncMock(),
    )
    feature = TaskFeature(actor)
    feature.set_task_manager(local_manager)

    result = await feature.cancel_task("colliding-task")

    assert result.status is ToolResultStatus.ERROR
    assert result.data["error_type"] == "ambiguous_direction"
    peers._resolve_retained_automatic_peer.assert_not_awaited()
    local_manager.cancel_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_outbound_route_cannot_fall_through_shared_task_row(monkeypatch):
    from kestrel_sovereign.a2a import outbound_store

    actor = SimpleNamespace(did="did:test:creator", identity=None, features={})
    peers = PeersFeature(actor)
    peers._db = object()
    peers._outbound_route_store_ready = True
    peers._own_name = "creator"
    monkeypatch.setattr(
        outbound_store,
        "get_outbound_task",
        AsyncMock(return_value=None),
    )
    actor.features["PeersFeature"] = peers
    local_manager = MagicMock(
        is_task_recipient=AsyncMock(return_value=False),
        cancel_task=AsyncMock(),
    )
    feature = TaskFeature(actor)
    feature.set_task_manager(local_manager)

    result = await feature.cancel_task("delivered-without-route")

    assert result.status is ToolResultStatus.ERROR
    assert "unambiguous" in result.error
    local_manager.cancel_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_outbound_route_allows_exact_inbound_recipient(monkeypatch):
    from kestrel_sovereign.a2a import outbound_store

    actor = SimpleNamespace(did="did:test:recipient", identity=None, features={})
    peers = PeersFeature(actor)
    peers._db = object()
    peers._outbound_route_store_ready = True
    peers._own_name = "recipient"
    monkeypatch.setattr(
        outbound_store,
        "get_outbound_task",
        AsyncMock(return_value=None),
    )
    actor.features["PeersFeature"] = peers
    canceled = Task(
        id="inbound-task",
        status=TaskStatus(state=TaskState.CANCELED),
        metadata={
            "cancellation_receipt": {
                "status_before": "working",
            }
        },
    )
    local_manager = MagicMock(
        is_task_recipient=AsyncMock(return_value=True),
        cancel_task=AsyncMock(return_value=canceled),
    )
    feature = TaskFeature(actor)
    feature.set_task_manager(local_manager)

    result = await feature.cancel_task("inbound-task")

    assert result.status is ToolResultStatus.OK
    local_manager.cancel_task.assert_awaited_once_with(
        "inbound-task",
        reason=None,
        agent_name=actor.did,
        recipient_agent_id=actor.did,
    )


@pytest.mark.asyncio
async def test_outbound_cancel_accepts_audit_stamp_won_by_terminal_sse(monkeypatch):
    from kestrel_sovereign.a2a import outbound_store

    actor = SimpleNamespace(did="did:test:creator", identity=None)
    peers = PeersFeature(actor)
    peers._db = object()
    peers._outbound_route_store_ready = True
    peers._own_name = "creator"
    initial = SimpleNamespace(
        recipient="Recipient",
        recipient_agent_id="did:test:recipient",
        route_state=outbound_store.ROUTE_STATE_ROUTABLE,
        terminal_state=None,
    )
    canceled = SimpleNamespace(**{**vars(initial), "terminal_state": "canceled"})
    route_lookup = AsyncMock(side_effect=[initial, canceled])
    monkeypatch.setattr(outbound_store, "get_outbound_task", route_lookup)
    terminal_stamp = AsyncMock(return_value=0)
    monkeypatch.setattr(
        outbound_store,
        "update_outbound_terminal_state",
        terminal_stamp,
    )
    peers._resolve_retained_automatic_peer = AsyncMock(
        return_value=(
            SimpleNamespace(
                cancel_a2a_task=AsyncMock(
                    return_value={
                        "id": "outbound-task",
                        "status": "canceled",
                        "cancellation_receipt": {"status_before": "working"},
                    }
                )
            ),
            PeerRequester(actor.did, object()),
            PeerIdentity(
                agent_id="did:test:recipient",
                slug="recipient",
                routing_key="recipient-route",
            ),
        )
    )
    peers._maybe_sign_outbound = MagicMock()

    result = await peers.cancel_outbound_task("outbound-task")

    assert result is not None
    assert result.status is ToolResultStatus.OK
    assert route_lookup.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        {},
        {"id": "other", "status": "canceled", "cancellation_receipt": {}},
        {"id": "outbound-task", "status": "working", "cancellation_receipt": {}},
        {
            "id": "outbound-task",
            "status": "canceled",
            "cancellation_receipt": "not-an-object",
        },
    ],
)
async def test_outbound_cancellation_rejects_malformed_peer_receipts(
    monkeypatch,
    response,
):
    from kestrel_sovereign.a2a import outbound_store

    actor = SimpleNamespace(did="did:test:creator", identity=None)
    peers = PeersFeature(actor)
    peers._db = object()
    peers._outbound_route_store_ready = True
    peers._own_name = "creator"
    monkeypatch.setattr(
        outbound_store,
        "get_outbound_task",
        AsyncMock(
            return_value=SimpleNamespace(
                recipient="Recipient",
                recipient_agent_id="did:test:recipient",
                route_state=outbound_store.ROUTE_STATE_ROUTABLE,
            )
        ),
    )
    stamp = AsyncMock(return_value=1)
    monkeypatch.setattr(outbound_store, "update_outbound_terminal_state", stamp)
    router = SimpleNamespace(cancel_a2a_task=AsyncMock(return_value=response))
    peers._resolve_retained_automatic_peer = AsyncMock(
        return_value=(
            router,
            PeerRequester(actor.did, object()),
            PeerIdentity(
                agent_id="did:test:recipient",
                slug="recipient",
                routing_key="recipient-route",
            ),
        )
    )
    peers._maybe_sign_outbound = MagicMock()

    result = await peers.cancel_outbound_task("outbound-task")

    assert result is not None
    assert result.status is ToolResultStatus.ERROR
    stamp.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        OSError("backend unavailable"),
        OutboundTaskRouteAmbiguousError("duplicate route"),
    ],
)
async def test_outbound_route_lookup_failure_never_falls_back_local(
    monkeypatch,
    failure,
):
    from kestrel_sovereign.a2a import outbound_store

    actor = SimpleNamespace(did="did:test:creator", identity=None, features={})
    peers = PeersFeature(actor)
    peers._db = object()
    peers._outbound_route_store_ready = True
    peers._own_name = "creator"
    monkeypatch.setattr(
        outbound_store,
        "get_outbound_task",
        AsyncMock(side_effect=failure),
    )
    actor.features["PeersFeature"] = peers
    local_manager = MagicMock(cancel_task=AsyncMock())
    feature = TaskFeature(actor)
    feature.set_task_manager(local_manager)

    result = await feature.cancel_task("outbound-task")

    assert result.status is ToolResultStatus.ERROR
    local_manager.cancel_task.assert_not_awaited()
