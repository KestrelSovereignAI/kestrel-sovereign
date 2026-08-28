"""Authority and atomicity regressions for A2A task cancellation (#3134)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sdk.hooks.base import HookEvent, HookOutput, PermissionDecision
from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.a2a.task_manager import (
    TaskCancellationAuthorizationError,
    create_task_manager,
)
from kestrel_sovereign.a2a.outbound_store import OutboundTaskRouteAmbiguousError
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
    PeerIdentity,
    PeerRequester,
    PeerTaskConflictError,
)
from kestrel_sovereign.features.peers.feature import PeersFeature


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
async def test_cancel_task_creator_and_execution_delegate_are_authorized(tmp_path):
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

        assert owner_result.status is ToolResultStatus.OK
        assert delegate_result.status is ToolResultStatus.OK
        owner_task = await manager.get_task("by-owner")
        delegate_task = await manager.get_task("by-delegate")
        assert owner_task.metadata["cancellation_receipt"] == {
            "actor_agent_id": "did:test:creator",
            "reason": "assignment withdrawn",
            "status_before": "submitted",
        }
        assert delegate_task.metadata["cancellation_receipt"] == {
            "actor_agent_id": "did:test:recipient",
            "reason": "cannot continue",
            "status_before": "submitted",
        }
        assert owner_task.history[-1].parts[0].text == (
            "Task canceled by did:test:creator: assignment withdrawn"
        )
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
            await manager.get_task("recipient-bound")
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
        feature_retry = await _feature(manager, "did:test:creator").cancel_task(
            "idempotent",
            reason="a newer reason that was never persisted",
        )
        assert feature_retry.status is ToolResultStatus.OK
        assert feature_retry.data["reason"] == "withdrawn"
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
            )

        canceled = await manager.get_task("artifact-after-cancel")
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
async def test_cancel_task_refuses_peer_lineage_and_causation(
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
        assert "not authorized" in result.error
        unchanged = await manager.get_task("protected")
        assert unchanged.status.state is TaskState.SUBMITTED
        assert "cancellation_receipt" not in (unchanged.metadata or {})
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
        assert (await manager.get_task("spoofed")).status.state is TaskState.SUBMITTED
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
        persisted = await manager.get_task("forged-receipt")
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
            await manager.get_task("refused-intent")
        ).status.state is TaskState.SUBMITTED
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_legacy_metadata_cannot_supply_a_cancellation_receipt(tmp_path):
    """Rows without durable receipt columns cannot retain old metadata claims."""

    manager = await create_task_manager(str(tmp_path / "legacy-forged-receipt.db"))
    try:
        await manager.task_store._backend.execute(
            """
            INSERT INTO a2a_tasks (
                id, task_type, status, metadata,
                creator_agent_id, recipient_agent_id,
                canceled_by, cancel_reason, cancel_previous_status
            ) VALUES (?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL)
            """,
            (
                "legacy-forged-receipt",
                "generic",
                "canceled",
                '{"cancellation_receipt":{"actor_agent_id":"did:test:evil",'
                '"reason":"forged","status_before":"working"}}',
            ),
        )

        legacy = await manager.get_task("legacy-forged-receipt")
        assert "cancellation_receipt" not in (legacy.metadata or {})
        with pytest.raises(ValueError, match="Invalid state transition"):
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

        assert await manager.get_task("born-canceled") is None
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

        assert await manager.task_store.save(task) is True

        persisted = await manager.get_task("forged-save-receipt")
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
        assert (await manager.get_task("immutable")).status.state is TaskState.SUBMITTED
        assert (
            await _feature(manager, "did:test:creator").cancel_task("immutable")
        ).status is ToolResultStatus.OK
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
        )
        await manager.update_status(
            task.id,
            TaskState.COMPLETED,
            agent_name="did:test:recipient",
        )

        result = await _feature(manager, "did:test:creator").cancel_task("finished")

        assert result.status is ToolResultStatus.ERROR
        unchanged = await manager.get_task("finished")
        assert unchanged.status.state is TaskState.COMPLETED
        assert "cancellation_receipt" not in (unchanged.metadata or {})
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
        task = await manager.get_task("race")
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
        assert (await manager.get_task("owned")).status.state is TaskState.SUBMITTED
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
        assert "not authorized" in result.error
        assert (
            await manager.get_task("respond-protected")
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

        original = await manager.get_task("same-id")
        assert original.metadata["payload"] == "original"
        assert original.history[0].parts[0].text == "Do the work"
        session_after = await manager.session_service.get_session("session-same-id")
        assert session_after.events == session_before.events
        assert (
            await _feature(manager, "did:test:creator-a").cancel_task("same-id")
        ).status is ToolResultStatus.OK
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
        assert (await manager.get_task(created.id)).status.state is TaskState.COMPLETED
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
        stale_worker_copy = await manager.get_task("stale-worker")
        await manager.cancel_task(
            "stale-worker",
            reason="withdrawn",
            agent_name="did:test:creator",
        )

        stale_worker_copy.status = TaskStatus(state=TaskState.COMPLETED)
        await manager.task_store.save(stale_worker_copy)

        persisted = await manager.get_task("stale-worker")
        assert persisted.status.state is TaskState.CANCELED
        assert persisted.metadata["cancellation_receipt"]["reason"] == "withdrawn"
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_lifecycle_save_without_authority_cannot_create_task(tmp_path):
    manager = await create_task_manager(str(tmp_path / "update-only.db"))
    try:
        inserted = await manager.task_store.save(
            Task(
                id="authority-less",
                status=TaskStatus(state=TaskState.SUBMITTED),
            )
        )

        assert inserted is False
        assert await manager.get_task("authority-less") is None
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
            await manager.update_status(task.id, TaskState.CANCELED)

        task.status = TaskStatus(state=TaskState.CANCELED)
        with pytest.raises(ValueError, match="cancel_if_authorized"):
            await manager.task_store.save(task)
        with pytest.raises(ValueError, match="cancel_if_authorized"):
            await manager.task_store.update_status(
                task.id, TaskStatus(state=TaskState.CANCELED)
            )

        assert (await manager.get_task(task.id)).status.state is TaskState.SUBMITTED
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

        persisted = await manager.get_task(task.id)
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

        persisted = await manager.get_task(task.id)
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
        pending = (await manager.get_pending_tasks())[0]
        concurrent = await manager.get_task(pending.id)
        concurrent.artifacts = [
            Artifact(name="concurrent", parts=[TextPart(text="keep")])
        ]
        concurrent.metadata["concurrent_state"] = "committed"
        concurrent.history.append(
            Message(role="agent", parts=[TextPart(text="concurrent history")])
        )
        assert await manager.task_store.save(concurrent) is True

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
        pending = (await manager.get_pending_tasks())[0]
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
        pending = (await manager.get_pending_tasks())[0]
        winning = await manager.get_task(pending.id)
        winning.status = TaskStatus(
            state=TaskState.COMPLETED,
            message=Message(
                role="agent",
                parts=[TextPart(text="concurrent completion")],
            ),
        )
        assert await manager.task_store.save(winning) is True

        release.set()
        await manager.drain_execution_tasks()

        persisted = await manager.get_task(pending.id)
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
        get_task=AsyncMock(return_value=visible_recipient_row),
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
