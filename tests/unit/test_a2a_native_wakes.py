"""Durable native provenance drives immediate and restarted cognition wakes."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from kestrel_sovereign._vendor.a2a.helpers.proto_helpers import new_text_message
from kestrel_sovereign._vendor.a2a.types import Role, SendMessageRequest, TaskState
from kestrel_sdk.signals import Trust

from kestrel_sovereign.a2a.extensions import CAUSATION_URI, QUESTION_CORRELATION_URI
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.a2a.stores.unified.task_store import (
    TaskMutationAuthorizationError,
    TaskStore,
)
from kestrel_sovereign.agent.event_manager import EventManagerMixin
from kestrel_sovereign.signals.durable import DurableSignalEvent
from kestrel_sovereign.signals.rehydration import DurableCognitionWithdrawal
from kestrel_sovereign.signals.sources.a2a import build_signal_for_completed_task
from kestrel_sovereign.signals.sources.a2a_task_submitted import (
    build_signal_for_submitted_task,
    submitted_input_identity,
)
from tests.utils.native_a2a import manager_for


@pytest_asyncio.fixture
async def native_manager(native_a2a_backend_factory):
    backend = native_a2a_backend_factory()
    await backend.connect()
    manager = manager_for(
        backend, NativeRecipientScope("did:kestrel:recipient", "tenant")
    )
    try:
        await manager.initialize()
        yield manager
    finally:
        await manager.close()


def request(metadata=None):
    return SendMessageRequest(
        tenant="tenant",
        metadata=metadata or {},
        message=new_text_message(
            "Real request", role=Role.ROLE_USER, context_id="host-chat-session"
        ),
    )


@pytest.mark.parametrize(
    "claimed_sender,verified", [("did:kestrel:forged", True), ("", False)]
)
async def test_immediate_and_restarted_wakes_use_committed_principals_not_metadata(
    native_manager,
    native_a2a_backend_factory,
    claimed_sender,
    verified,
):
    callbacks = []
    native_manager._on_task_submitted = callbacks.append
    task = await native_manager.create_task(
        request(
            {
                "sender": claimed_sender,
                "sender_verified": verified,
                "creator_agent_id": "did:kestrel:forged",
                "recipient_agent_id": "did:kestrel:elsewhere",
                "tenant": "forged-tenant",
                QUESTION_CORRELATION_URI: {
                    "version": 1,
                    "kind": "question",
                    "question_id": "question-one",
                },
            }
        ),
        creator_agent_id="did:kestrel:verified-sender",
    )
    first = build_signal_for_submitted_task(callbacks[0], native_manager.did)
    backend = native_a2a_backend_factory()
    await backend.connect()
    restarted = manager_for(backend, native_manager.task_store.scope)
    try:
        await restarted.initialize()
        candidate = await restarted.get_task_snapshot(
            task.id, recipient_agent_id=restarted.did
        )
        recovered = build_signal_for_submitted_task(candidate, restarted.did)
        for signal in (first, recovered):
            assert (
                signal.payload["sender"]
                == signal.caller
                == "did:kestrel:verified-sender"
            )
            assert signal.origin_trust == Trust.TRUSTED
            assert signal.payload[
                "session_id"
            ] == native_manager.session_id_for_context(task.context_id)
            assert signal.payload["session_id"] != "host-chat-session"
            assert signal.payload["request_content"] == "Real request"
            assert (
                signal.payload["question_id"] == "question-one"
                and signal.payload["reply_expected"]
            )
        assert (
            first.dedupe_key
            == recovered.dedupe_key
            == submitted_input_identity(task.id, 0)
        )
        candidates = await restarted.list_cognition_wake_candidates(
            recipient_agent_id=restarted.did,
            live_changed_since=datetime.now(UTC) - timedelta(days=1),
            terminal_changed_since=datetime.now(UTC) - timedelta(days=1),
        )
        assert candidates[0].ownership == candidate.ownership
        assert (
            await restarted.get_task_snapshot(
                task.id, recipient_agent_id="did:kestrel:elsewhere"
            )
            is None
        )
        foreign_tenant = TaskStore(
            backend,
            scope=replace(restarted.task_store.scope, tenant="another-tenant"),
            owns_backend=False,
        )
        assert (
            await foreign_tenant.get_task_snapshot(
                task.id, recipient_agent_id=restarted.did
            )
            is None
        )
    finally:
        await restarted.close()


@pytest.mark.parametrize(
    "metadata",
    [
        {CAUSATION_URI: {"version": 2, "chain": []}},
        {CAUSATION_URI: {"version": 1, "chain": [{"source": "forged"}]}},
        {
            QUESTION_CORRELATION_URI: {
                "version": 1,
                "kind": "question",
                "question_id": "",
            }
        },
    ],
)
async def test_invalid_declared_extensions_fail_before_context_and_task_admission(
    native_manager, metadata
):
    from kestrel_sovereign._vendor.a2a.utils.errors import InvalidParamsError

    with pytest.raises(InvalidParamsError):
        await native_manager.create_task(
            request(metadata), creator_agent_id="did:kestrel:sender"
        )
    assert (
        await native_manager.task_store.backend.fetch_all("SELECT id FROM a2a_tasks")
        == []
    )
    assert (
        await native_manager.task_store.backend.fetch_all(
            "SELECT context_id FROM a2a_contexts"
        )
        == []
    )


async def test_unproven_or_misrouted_snapshots_cannot_mint_trusted_wakes(
    native_manager,
):
    task = await native_manager.create_task(
        request(), creator_agent_id="did:kestrel:sender"
    )
    candidate = await native_manager.get_task_snapshot(
        task.id, recipient_agent_id=native_manager.did
    )
    with pytest.raises(TypeError, match="durable task provenance"):
        build_signal_for_submitted_task(task, native_manager.did)
    with pytest.raises(TaskMutationAuthorizationError):
        build_signal_for_submitted_task(candidate, "did:kestrel:elsewhere")
    for ownership in [
        replace(candidate.ownership, creator_agent_id=""),
        replace(candidate.ownership, context_id="different"),
    ]:
        with pytest.raises(TaskMutationAuthorizationError):
            build_signal_for_submitted_task(
                replace(candidate, ownership=ownership), native_manager.did
            )


async def test_native_rejection_reaches_completion_signal_and_actual_notification_callback(
    native_manager,
):
    class Recipient(EventManagerMixin):
        did = native_manager.did
        dispatcher = None

        def __init__(self):
            self._pending_task_notifications = []

    recipient = Recipient()
    native_manager._on_task_complete = recipient._on_background_task_complete
    task = await native_manager.create_task(
        request({"sender": "forged"}), creator_agent_id="did:kestrel:sender"
    )
    await native_manager.update_status(
        task.id,
        TaskState.TASK_STATE_REJECTED,
        recipient_agent_id=native_manager.did,
    )
    candidate = await native_manager.get_task_snapshot(
        task.id, recipient_agent_id=native_manager.did
    )
    signal = build_signal_for_completed_task(candidate, native_manager.did)
    assert signal.payload["task_state"] == "rejected"
    assert signal.caller == "did:kestrel:sender"
    assert len(recipient._pending_task_notifications) == 1
    assert "rejected" in recipient._pending_task_notifications[0]
    assert "forged" not in recipient._pending_task_notifications[0]
    assert task.id in recipient._pending_task_notifications[0]


async def test_privacy_elided_rehydration_reads_native_ownership_and_withdraws_stale_submission(
    native_manager,
):
    task = await native_manager.create_task(
        request({"sender": "forged", "sender_verified": True}),
        creator_agent_id="did:kestrel:sender",
    )
    candidate = await native_manager.get_task_snapshot(
        task.id, recipient_agent_id=native_manager.did
    )
    original = build_signal_for_submitted_task(candidate, native_manager.did)
    projected = replace(original, payload={}, caller=None)
    now = datetime.now(UTC)
    event = DurableSignalEvent(
        event_id=original.id,
        source_event_id=submitted_input_identity(task.id, 0),
        agent_id=native_manager.did,
        target_agent=native_manager.did,
        source=original.source,
        kind=original.kind,
        mode=original.mode.value,
        payload=None,
        session_id=None,
        caller_identity=None,
        visibility=original.visibility.value,
        urgency=original.urgency.value,
        dedupe_key=task.id,
        causation_chain=[],
        arrived_at=now,
        committed_at=now,
        retention_until=now + timedelta(days=1),
    )
    recipient = EventManagerMixin()
    recipient.did = native_manager.did
    recipient.task_manager = native_manager
    recovered = await recipient.rehydrate_durable_cognition_signal(
        event, dispatch_signal=projected
    )
    assert recovered.id == projected.id and recovered.arrived_at == projected.arrived_at
    assert recovered.caller == recovered.payload["sender"] == "did:kestrel:sender"
    assert recovered.payload["session_id"] != "host-chat-session"
    assert (
        await recipient.rehydrate_durable_cognition_signal(
            replace(event, target_agent="did:kestrel:elsewhere"),
            dispatch_signal=projected,
        )
        is None
    )
    await native_manager.update_status(
        task.id,
        TaskState.TASK_STATE_REJECTED,
        recipient_agent_id=native_manager.did,
    )
    assert isinstance(
        await recipient.rehydrate_durable_cognition_signal(
            event, dispatch_signal=projected
        ),
        DurableCognitionWithdrawal,
    )
    completed_event = replace(
        event, source="a2a.task_complete", source_event_id=task.id
    )
    completed = await recipient.rehydrate_durable_cognition_signal(
        completed_event, dispatch_signal=projected
    )
    assert (
        completed.payload["task_state"] == "rejected"
        and completed.caller == "did:kestrel:sender"
    )


@pytest.mark.parametrize(
    "intent,verb,expected_reply",
    [("notification", "message", False), ("assignment", "task", True)],
)
async def test_native_wake_uses_declared_intent_and_advisory_hint_without_draft_metadata(
    native_manager, intent, verb, expected_reply
):
    from kestrel_sovereign.a2a.extensions import (
        delivery_intent_metadata,
        skill_hint_metadata,
    )

    task = await native_manager.create_task(
        request(
            {
                "skill": "forged.old.skill",
                "a2a_verb": "forged-old-verb",
                "reply_expected": not expected_reply,
                **delivery_intent_metadata(intent),
                **skill_hint_metadata("published-advisory-skill"),
            }
        ),
        creator_agent_id="did:kestrel:verified-sender",
    )
    snapshot = await native_manager.get_task_snapshot(
        task.id, recipient_agent_id=native_manager.did
    )
    signal = build_signal_for_submitted_task(snapshot, native_manager.did)
    assert signal.payload["a2a_verb"] == verb
    assert signal.payload["skill_id"] == "published-advisory-skill"
    assert signal.payload["reply_expected"] is expected_reply


async def test_undeclared_draft_skill_cannot_become_native_wake_hint(native_manager):
    task = await native_manager.create_task(
        request({"skill": "operator.shell"}),
        creator_agent_id="did:kestrel:verified-sender",
    )
    snapshot = await native_manager.get_task_snapshot(
        task.id, recipient_agent_id=native_manager.did
    )
    assert (
        build_signal_for_submitted_task(snapshot, native_manager.did).payload[
            "skill_id"
        ]
        == ""
    )
