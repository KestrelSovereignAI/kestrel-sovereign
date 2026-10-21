"""Actual sender-ledger persistence, immutable routing and uncertain submission."""

from dataclasses import replace
from uuid import uuid4

import pytest
import pytest_asyncio
from a2a.types import (
    AgentInterface,
    Message,
    Part,
    Role,
    SendMessageRequest,
    Task,
    TaskState,
    TaskStatus,
)

from kestrel_sovereign.a2a.native_schema import NativeA2AMigrationRequired
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.a2a.stores.unified.outbound_store import (
    DispatchState,
    OutboundDispatchConflict,
    OutboundStore,
)


@pytest_asyncio.fixture
async def outbound(native_a2a_backend_factory):
    backend = native_a2a_backend_factory()
    await backend.connect()
    store = OutboundStore(backend, sender_scope=NativeRecipientScope("did:sender"))
    await store.initialize()
    try:
        yield store
    finally:
        await backend.close()


def request(*, context="context", tenant="tenant"):
    return SendMessageRequest(
        tenant=tenant,
        message=Message(
            message_id=uuid4().hex,
            context_id=context,
            role=Role.ROLE_USER,
            parts=[Part(text="original private message")],
        ),
    )


async def reserve(store, params, *, dispatch_id=None, recipient="did:recipient"):
    return await store.reserve(
        dispatch_id or uuid4().hex,
        params,
        recipient=NativeRecipientScope(recipient, params.tenant),
        signing_did="did:web:peer.test",
        dispatch_tool="send_a2a_task",
        verb="task",
        interface=AgentInterface(
            url="https://peer.test/a2a",
            protocol_binding="JSONRPC",
            protocol_version="1.0",
            tenant=params.tenant,
        ),
    )


def task(params, *, task_id="remote-task", state=TaskState.TASK_STATE_SUBMITTED):
    result = Task(
        id=task_id,
        context_id=params.message.context_id or "server-context",
        status=TaskStatus(state=state),
        history=[params.message],
    )
    result.history[0].task_id = result.id
    result.history[0].context_id = result.context_id
    return result


async def test_local_dispatch_identity_does_not_become_remote_task_identity(outbound):
    params = request()
    original = await reserve(outbound, params, dispatch_id="local-dispatch")
    assert original.state == DispatchState.RESERVED
    assert original.remote_task_id is None
    submitting = await outbound.begin_submission(original, operation_id="submission")
    accepted = await outbound.acknowledge(
        submitting, task(params), operation_id="receipt"
    )
    assert accepted.dispatch_id == "local-dispatch"
    assert accepted.remote_task_id == "remote-task"
    assert accepted.state == DispatchState.ACKNOWLEDGED
    assert accepted.revision == 2
    assert await outbound.get("remote-task") is None
    assert await outbound.get("local-dispatch") == accepted
    row = await outbound.backend.fetch_one(
        "SELECT request_digest, message_digest FROM a2a_outbound_dispatches"
    )
    assert all(len(value) == 64 for value in row)
    assert params.message.parts[0].text == "original private message"


async def test_unknown_acknowledgement_never_becomes_permission_to_replay(outbound):
    params = request()
    reserved = await reserve(outbound, params)
    submitting = await outbound.begin_submission(reserved, operation_id="send")
    unknown = await outbound.mark_unknown(submitting, operation_id="lost-ack")
    assert unknown.remote_task_id is None
    assert await outbound.begin_submission(unknown, operation_id="retry") is None
    assert await outbound.reconcile(unknown, [], operation_id="empty-read") is None
    assert await outbound.get(unknown.dispatch_id) == unknown
    accepted = await outbound.reconcile(
        unknown, [task(params)], operation_id="read-receipt"
    )
    assert accepted.dispatch_id == reserved.dispatch_id
    assert accepted.remote_task_id == "remote-task"
    assert accepted.state == DispatchState.ACKNOWLEDGED


@pytest.mark.parametrize("fault", ["message-id", "text", "context", "role", "metadata"])
async def test_reconciliation_requires_the_original_message(outbound, fault):
    params = request()
    reserved = await reserve(outbound, params)
    submitting = await outbound.begin_submission(reserved, operation_id="send")
    unknown = await outbound.mark_unknown(submitting, operation_id="unknown")
    candidate = task(params)
    if fault == "message-id":
        candidate.history[0].message_id = "different"
    elif fault == "text":
        candidate.history[0].parts[0].text = "different"
    elif fault == "context":
        candidate.context_id = "different"
    elif fault == "role":
        candidate.history[0].role = Role.ROLE_AGENT
    else:
        candidate.history[0].metadata["untrusted"] = True
    assert await outbound.reconcile(unknown, [candidate], operation_id="read") is None
    assert await outbound.get(unknown.dispatch_id) == unknown


async def test_multiple_remote_tasks_cannot_claim_one_original_dispatch(outbound):
    params = request()
    reserved = await reserve(outbound, params)
    submitting = await outbound.begin_submission(reserved, operation_id="send")
    unknown = await outbound.mark_unknown(submitting, operation_id="unknown")
    with pytest.raises(OutboundDispatchConflict, match="More than one"):
        await outbound.reconcile(
            unknown,
            [task(params), task(params, task_id="another")],
            operation_id="read",
        )
    assert await outbound.get(unknown.dispatch_id) == unknown


async def test_taskless_message_remains_a_message_receipt(outbound):
    params = request(context="")
    reserved = await reserve(outbound, params)
    submitting = await outbound.begin_submission(reserved, operation_id="send")
    response = Message(
        message_id="answer",
        context_id="server-context",
        role=Role.ROLE_AGENT,
        parts=[Part(text="reply")],
    )
    complete = await outbound.acknowledge(submitting, response, operation_id="receipt")
    assert complete.state == DispatchState.COMPLETE
    assert complete.remote_task_state is None
    assert complete.remote_task_id is None
    assert complete.remote_context_id == "server-context"
    assert complete.dispatch_id == reserved.dispatch_id


async def test_one_sender_cannot_read_or_mutate_another_senders_dispatch(outbound):
    params = request()
    reserved = await reserve(outbound, params)
    other = OutboundStore(
        outbound.backend, sender_scope=NativeRecipientScope("did:other")
    )
    assert await other.get(reserved.dispatch_id) is None
    with pytest.raises(OutboundDispatchConflict, match="another sender"):
        await other.begin_submission(reserved, operation_id="forged")
    assert await outbound.get(reserved.dispatch_id) == reserved
    # Each sender owns its local namespace, including an equal opaque ID.
    other_reserved = await reserve(other, params, dispatch_id=reserved.dispatch_id)
    assert other_reserved.sender_id == "did:other"
    assert await other.get(reserved.dispatch_id) == other_reserved
    assert await outbound.get(reserved.dispatch_id) == reserved


@pytest.mark.parametrize(
    "field,value",
    [
        ("recipient_id", "did:another-recipient"),
        ("tenant", "another-tenant"),
        ("signing_did", "did:web:other.test"),
        ("interface_url", "https://other.test"),
        ("protocol_binding", "HTTP+JSON"),
        ("message_id", "different"),
        ("requested_context_id", "different"),
        ("request_digest", "different"),
        ("message_digest", "different"),
        ("operation_id", "different"),
        ("state", DispatchState.UNKNOWN),
        ("remote_task_id", "forged"),
    ],
)
async def test_snapshot_fields_cannot_retarget_a_committed_dispatch(
    outbound, field, value
):
    params = request()
    reserved = await reserve(outbound, params)
    forged = replace(reserved, **{field: value})
    assert await outbound.begin_submission(forged, operation_id="retarget") is None
    assert await outbound.get(reserved.dispatch_id) == reserved


async def test_known_task_finalization_preserves_its_original_remote_identity(outbound):
    params = request()
    reserved = await reserve(outbound, params)
    submitting = await outbound.begin_submission(reserved, operation_id="send")
    accepted = await outbound.acknowledge(
        submitting, task(params), operation_id="accepted"
    )
    assert await outbound.mark_unknown(accepted, operation_id="forget-ack") is None
    with pytest.raises(OutboundDispatchConflict, match="acknowledged task"):
        await outbound.acknowledge(
            accepted, task(params, task_id="replacement"), operation_id="bad"
        )
    complete = await outbound.acknowledge(
        accepted,
        task(params, state=TaskState.TASK_STATE_COMPLETED),
        operation_id="done",
    )
    assert complete.state == DispatchState.COMPLETE
    assert complete.dispatch_id == reserved.dispatch_id
    assert complete.remote_task_id == accepted.remote_task_id
    assert complete.remote_task_state == "completed"
    assert complete.response_kind == "task"
    assert await outbound.mark_unknown(complete, operation_id="reopen") is None
    assert await outbound.begin_submission(complete, operation_id="replay") is None
    assert (
        await outbound.acknowledge(accepted, task(params), operation_id="late") is None
    )


async def test_duplicate_read_of_one_remote_task_is_one_reconciliation_witness(
    outbound,
):
    params = request(context="")
    reserved = await reserve(outbound, params)
    submitting = await outbound.begin_submission(reserved, operation_id="send")
    unknown = await outbound.mark_unknown(submitting, operation_id="lost")
    candidate = task(params)
    accepted = await outbound.reconcile(
        unknown, [candidate, candidate], operation_id="read"
    )
    assert accepted.remote_context_id == "server-context"
    assert accepted.remote_task_id == candidate.id


@pytest.mark.parametrize("field", ["context_id", "task_id"])
async def test_reconciliation_refuses_history_belonging_to_another_task(
    outbound, field
):
    params = request(context="")
    reserved = await reserve(outbound, params)
    submitting = await outbound.begin_submission(reserved, operation_id="send")
    unknown = await outbound.mark_unknown(submitting, operation_id="lost")
    candidate = task(params)
    setattr(candidate.history[0], field, "foreign")
    assert await outbound.reconcile(unknown, [candidate], operation_id="read") is None
    assert await outbound.get(unknown.dispatch_id) == unknown


async def test_two_callers_cannot_both_acquire_permission_to_submit(
    outbound, native_a2a_backend_factory
):
    import asyncio

    second_backend = native_a2a_backend_factory()
    await second_backend.connect()
    try:
        second = OutboundStore(
            second_backend, sender_scope=NativeRecipientScope("did:sender")
        )
        reserved = await reserve(outbound, request())
        outcomes = await asyncio.gather(
            outbound.begin_submission(reserved, operation_id="first"),
            second.begin_submission(reserved, operation_id="second"),
        )
        winners = [outcome for outcome in outcomes if outcome is not None]
        assert len(winners) == 1
        assert await second.get(reserved.dispatch_id) == winners[0]
    finally:
        await second_backend.close()


async def test_duplicate_local_message_does_not_create_another_dispatch(outbound):
    params = request()
    original = await reserve(outbound, params)
    with pytest.raises(OutboundDispatchConflict, match="already reserved"):
        await reserve(outbound, params)
    rows = await outbound.backend.fetch_all(
        "SELECT dispatch_id FROM a2a_outbound_dispatches"
    )
    assert rows == [(original.dispatch_id,)]


async def test_remote_id_collision_cannot_rekey_another_dispatch(outbound):
    first, second = request(context="first"), request(context="second")
    a = await outbound.begin_submission(
        await reserve(outbound, first), operation_id="a"
    )
    b = await outbound.begin_submission(
        await reserve(outbound, second), operation_id="b"
    )
    accepted = await outbound.acknowledge(a, task(first), operation_id="ack-a")
    assert await outbound.acknowledge(b, task(second), operation_id="ack-b") is None
    assert await outbound.get(a.dispatch_id) == accepted
    assert await outbound.get(b.dispatch_id) == b


async def test_revision_still_fences_stale_receipts_when_clock_and_operation_tie(
    outbound,
):
    from datetime import UTC, datetime

    tied = OutboundStore(
        outbound.backend,
        sender_scope=NativeRecipientScope("did:sender"),
        clock=lambda: datetime(2026, 10, 20, tzinfo=UTC),
    )
    params = request()
    reserved = await reserve(tied, params)
    submitting = await tied.begin_submission(reserved, operation_id="send")
    accepted = await tied.acknowledge(submitting, task(params), operation_id="receipt")
    newer = await tied.acknowledge(accepted, task(params), operation_id="receipt")
    assert newer.updated_at == accepted.updated_at
    assert newer.operation_id == accepted.operation_id
    assert newer.revision == accepted.revision + 1
    assert await tied.acknowledge(accepted, task(params), operation_id="stale") is None
    assert await tied.get(accepted.dispatch_id) == newer


async def test_draft_rows_require_explicit_migration_before_native_startup(outbound):
    await outbound.backend.execute("CREATE TABLE a2a_outbound_tasks (task_id TEXT)")
    await outbound.backend.execute(
        "INSERT INTO a2a_outbound_tasks VALUES (?)", ("historical",)
    )
    with pytest.raises(NativeA2AMigrationRequired, match="offline migration"):
        await outbound.initialize()
    assert await outbound.backend.fetch_one(
        "SELECT task_id FROM a2a_outbound_tasks"
    ) == ("historical",)


@pytest.mark.parametrize("fault", ["tenant", "role", "caller-task", "draft-version"])
async def test_bad_native_submission_binding_creates_no_reservation(outbound, fault):
    params = request()
    interface = AgentInterface(
        url="https://peer.test/a2a",
        protocol_binding="JSONRPC",
        protocol_version="1.0",
        tenant=params.tenant,
    )
    if fault == "tenant":
        interface.tenant = "another"
    elif fault == "role":
        params.message.role = Role.ROLE_AGENT
    elif fault == "caller-task":
        params.message.task_id = "caller-chosen-task"
    else:
        interface.protocol_version = "0.3"
    with pytest.raises(ValueError, match="selected native recipient interface"):
        await outbound.reserve(
            "dispatch",
            params,
            recipient=NativeRecipientScope("did:recipient", "tenant"),
            signing_did="did:web:peer.test",
            interface=interface,
            dispatch_tool="send",
            verb="task",
        )
    assert (
        await outbound.backend.fetch_all(
            "SELECT dispatch_id FROM a2a_outbound_dispatches"
        )
        == []
    )


@pytest.mark.parametrize("revision", [True, -1, 2**63])
async def test_invalid_revision_refuses_a_write(outbound, revision):
    original = await reserve(outbound, request())
    with pytest.raises(ValueError, match="SQL integer"):
        await outbound.begin_submission(
            replace(original, revision=revision), operation_id="invalid"
        )
    assert await outbound.get(original.dispatch_id) == original


@pytest.mark.parametrize("fault", ["context", "role"])
async def test_bad_peer_message_receipt_preserves_submission_uncertainty(
    outbound, fault
):
    params = request()
    submitting = await outbound.begin_submission(
        await reserve(outbound, params), operation_id="send"
    )
    response = Message(
        message_id="answer",
        context_id=params.message.context_id,
        role=Role.ROLE_AGENT,
        parts=[Part(text="answer")],
    )
    if fault == "context":
        response.context_id = "foreign"
    else:
        response.role = Role.ROLE_USER
    with pytest.raises(OutboundDispatchConflict):
        await outbound.acknowledge(submitting, response, operation_id="bad-receipt")
    assert await outbound.get(submitting.dispatch_id) == submitting


async def test_remote_task_ids_are_scoped_to_recipient_and_tenant(outbound):
    accepted = []
    for recipient, tenant in (
        ("did:first", "one"),
        ("did:second", "one"),
        ("did:first", "two"),
    ):
        params = request(tenant=tenant)
        reserved = await reserve(outbound, params, recipient=recipient)
        submitting = await outbound.begin_submission(reserved, operation_id="send")
        accepted.append(
            await outbound.acknowledge(submitting, task(params), operation_id="receipt")
        )
    assert {row.remote_task_id for row in accepted} == {"remote-task"}
    assert len({row.dispatch_id for row in accepted}) == 3
    assert [(row.recipient_id, row.tenant) for row in accepted] == [
        ("did:first", "one"),
        ("did:second", "one"),
        ("did:first", "two"),
    ]


async def test_bad_clock_configuration_cannot_write_unverifiable_audit_times(outbound):
    from datetime import UTC, datetime

    broken_clock = OutboundStore(
        outbound.backend,
        sender_scope=NativeRecipientScope("did:sender"),
        clock=lambda: datetime(2026, 10, 20, tzinfo=UTC).replace(tzinfo=None),
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        await reserve(broken_clock, request())
    assert (
        await outbound.backend.fetch_all(
            "SELECT dispatch_id FROM a2a_outbound_dispatches"
        )
        == []
    )


async def test_real_task_manager_initializes_outbound_on_its_primary_backend(
    native_a2a_backend_factory,
):
    from tests.utils.native_a2a import manager_for

    backend = native_a2a_backend_factory()
    await backend.connect()
    manager = manager_for(
        backend, NativeRecipientScope("did:sender", "local-tenant"), owns_backend=False
    )
    try:
        await manager.initialize()
        assert manager.outbound_store.backend is manager.task_store.backend is backend
        reserved = await reserve(manager.outbound_store, request())
        assert reserved.sender_id == manager.did == "did:sender"
        assert reserved.sender_tenant == "local-tenant"
        await manager.close()
        assert backend.is_connected
        assert await backend.fetch_one(
            "SELECT dispatch_id FROM a2a_outbound_dispatches WHERE sender_id = ?",
            ("did:sender",),
        ) == (reserved.dispatch_id,)
    finally:
        await manager.close()
        await backend.close()


async def test_real_manager_refuses_draft_outbound_data_before_other_schema_writes(
    native_a2a_backend_factory,
):
    from tests.utils.native_a2a import manager_for

    backend = native_a2a_backend_factory()
    await backend.connect()
    manager = manager_for(
        backend, NativeRecipientScope("did:sender"), owns_backend=False
    )
    try:
        await backend.execute("CREATE TABLE a2a_outbound_tasks (task_id TEXT)")
        await backend.execute("INSERT INTO a2a_outbound_tasks VALUES (?)", ("draft",))
        with pytest.raises(NativeA2AMigrationRequired):
            await manager.initialize()
        assert not await backend.table_exists("a2a_tasks")
        assert not await backend.table_exists("a2a_outbound_dispatches")
    finally:
        await manager.close()
        await backend.close()


async def test_equal_sender_ids_do_not_cross_local_tenant_namespaces(outbound):
    params = request()
    original = await reserve(outbound, params)
    other = OutboundStore(
        outbound.backend,
        sender_scope=NativeRecipientScope("did:sender", "other-local-tenant"),
    )
    assert await other.get(original.dispatch_id) is None
    with pytest.raises(OutboundDispatchConflict, match="another sender"):
        await other.begin_submission(original, operation_id="cross-tenant")
    forged = replace(original, sender_tenant="other-local-tenant")
    assert await other.begin_submission(forged, operation_id="forged-scope") is None
    separate = await reserve(other, params, dispatch_id=original.dispatch_id)
    assert separate.sender_id == original.sender_id
    assert separate.sender_tenant != original.sender_tenant
    assert await outbound.get(original.dispatch_id) == original
    assert await other.get(original.dispatch_id) == separate


async def test_native_outbound_listing_has_actor_scope_filter_and_deterministic_limit(
    outbound,
):
    from datetime import UTC, datetime

    fixed = datetime(2026, 10, 20, tzinfo=UTC)
    outbound._clock = lambda: fixed
    a = await reserve(outbound, request(), dispatch_id="a", recipient="did:peer-a")
    b = await reserve(outbound, request(), dispatch_id="b", recipient="did:peer-b")
    c = await reserve(outbound, request(), dispatch_id="c", recipient="did:peer-a")
    other_sender = OutboundStore(
        outbound.backend, sender_scope=NativeRecipientScope("did:other")
    )
    other_tenant = OutboundStore(
        outbound.backend,
        sender_scope=NativeRecipientScope("did:sender", "another-tenant"),
    )
    await reserve(other_sender, request(), dispatch_id="foreign-sender")
    await reserve(other_tenant, request(), dispatch_id="foreign-tenant")
    assert await outbound.list_recent() == [c, b, a]
    assert await outbound.list_recent(limit=2) == [c, b]
    assert await outbound.list_recent(recipient_id="did:peer-a") == [c, a]
    assert await outbound.list_recent(recipient_id="did:absent") == []
    assert outbound.sender_scope == NativeRecipientScope("did:sender")


@pytest.mark.parametrize("limit", [True, 0, -1, 1001, 1.5, "1"])
async def test_native_outbound_listing_refuses_invalid_limits(outbound, limit):
    with pytest.raises(ValueError, match="list limit"):
        await outbound.list_recent(limit=limit)


async def test_native_outbound_listing_honors_host_configured_capacity(outbound):
    small = OutboundStore(
        outbound.backend, sender_scope=outbound.sender_scope, max_list_limit=2
    )
    await reserve(small, request())
    assert len(await small.list_recent(limit=2)) == 1
    with pytest.raises(ValueError, match="between 1 and 2"):
        await small.list_recent(limit=3)


@pytest.mark.parametrize("capacity", [True, 0, -1, 2**31, "2", 1.5])
async def test_host_list_capacity_refuses_malformed_configuration(outbound, capacity):
    with pytest.raises(ValueError, match="list capacity"):
        OutboundStore(
            outbound.backend,
            sender_scope=outbound.sender_scope,
            max_list_limit=capacity,
        )


@pytest.mark.parametrize("recipient", ["", " ", False, 0])
async def test_outbound_list_refuses_nonconcrete_peer_filter(outbound, recipient):
    with pytest.raises(ValueError, match="recipient_id"):
        await outbound.list_recent(recipient_id=recipient)


async def test_primary_manager_outbound_scope_is_one_canonical_actor_contract(outbound):
    from kestrel_sovereign.a2a.task_manager import OutboundActorScopeError
    from tests.utils.native_a2a import manager_for

    manager = manager_for(outbound.backend, outbound.sender_scope, owns_backend=False)
    await manager.initialize()
    try:
        assert (
            manager.outbound_for_actor(outbound.sender_scope.recipient_id)
            is manager.outbound_store
        )
        with pytest.raises(OutboundActorScopeError, match="different local actor"):
            manager.outbound_for_actor("did:another-actor")
        manager.outbound_store = OutboundStore(
            outbound.backend,
            sender_scope=NativeRecipientScope(
                outbound.sender_scope.recipient_id, "different-tenant"
            ),
        )
        with pytest.raises(OutboundActorScopeError, match="tenant"):
            manager.outbound_for_actor(outbound.sender_scope.recipient_id)
    finally:
        await manager.close()


async def test_known_no_delivery_is_terminal_and_cannot_forget_an_ack(outbound):
    params = request()
    reserved = await reserve(outbound, params)
    submitting = await outbound.begin_submission(reserved, operation_id="intent")
    failed = await outbound.record_no_delivery(
        submitting, operation_id="no-native-dispatch"
    )
    assert failed.state is DispatchState.NOT_SENT
    assert failed.remote_task_id is None
    assert await outbound.begin_submission(failed, operation_id="replay") is None
    with pytest.raises(OutboundDispatchConflict, match="Only an unknown"):
        await outbound.reconcile(failed, [], operation_id="no-reconciliation")
    other_params = request()
    other = await reserve(outbound, other_params)
    active = await outbound.begin_submission(other, operation_id="second-intent")
    acknowledged = await outbound.acknowledge(
        active, task(other_params), operation_id="accepted"
    )
    assert (
        await outbound.record_no_delivery(acknowledged, operation_id="cannot-forget")
        is None
    )
    assert await outbound.get(other.dispatch_id) == acknowledged


async def test_primary_actor_contract_refuses_another_connection_even_with_same_namespace(
    outbound, native_a2a_backend_factory
):
    from kestrel_sovereign.a2a.task_manager import OutboundActorScopeError
    from tests.utils.native_a2a import manager_for

    secondary = native_a2a_backend_factory()
    await secondary.connect()
    manager = manager_for(outbound.backend, outbound.sender_scope, owns_backend=False)
    await manager.initialize()
    try:
        manager.outbound_store = OutboundStore(
            secondary, sender_scope=outbound.sender_scope
        )
        with pytest.raises(OutboundActorScopeError, match="primary backend"):
            manager.outbound_for_actor(outbound.sender_scope.recipient_id)
    finally:
        await manager.close()
        await secondary.close()


async def test_cancellation_intent_has_one_claim_and_keeps_original_receipt(outbound):
    from a2a.types import CancelTaskRequest

    params = request()
    original = await reserve(outbound, params)
    submitting = await outbound.begin_submission(original, operation_id="send")
    accepted = await outbound.acknowledge(submitting, task(params), operation_id="ack")
    cancel = CancelTaskRequest(id=accepted.remote_task_id, tenant=accepted.tenant)
    claimed = await outbound.begin_cancellation(accepted, cancel, operation_id="cancel")
    assert claimed.remote_task_id == accepted.remote_task_id
    assert claimed.response_digest == accepted.response_digest
    assert claimed.request_digest == accepted.request_digest
    assert claimed.cancel_request_digest != claimed.request_digest
    assert (
        await outbound.begin_cancellation(
            accepted, cancel, operation_id="racing-cancel"
        )
        is None
    )
    with pytest.raises(OutboundDispatchConflict, match="already in flight"):
        await outbound.begin_cancellation(claimed, cancel, operation_id="retry")
    unknown = await outbound.retain_cancellation_failure(
        claimed, may_have_sent=True, operation_id="lost-ack"
    )
    assert unknown.cancel_state is DispatchState.UNKNOWN
    with pytest.raises(OutboundDispatchConflict, match="already in flight"):
        await outbound.begin_cancellation(unknown, cancel, operation_id="replay")
    # Read observations may settle the known task, but cannot erase an uncertain cancel.
    observed = await outbound.acknowledge(
        unknown, task(params), operation_id="active-read"
    )
    assert observed.cancel_state is DispatchState.UNKNOWN
    finished = await outbound.acknowledge(
        observed,
        task(params, state=TaskState.TASK_STATE_COMPLETED),
        operation_id="terminal-read",
    )
    assert finished.cancel_state is DispatchState.COMPLETE
    assert finished.remote_task_state == "completed"
    with pytest.raises(OutboundDispatchConflict, match="terminal outcome"):
        await outbound.acknowledge(finished, task(params), operation_id="stale-replica")
    with pytest.raises(OutboundDispatchConflict, match="terminal outcome"):
        await outbound.acknowledge(
            finished,
            task(params, state=TaskState.TASK_STATE_CANCELED),
            operation_id="rewrite-outcome",
        )
    assert await outbound.get(original.dispatch_id) == finished


async def test_cancellation_claim_refuses_foreign_remote_task_or_tenant(outbound):
    from a2a.types import CancelTaskRequest

    params = request()
    original = await reserve(outbound, params)
    submitting = await outbound.begin_submission(original, operation_id="send")
    accepted = await outbound.acknowledge(submitting, task(params), operation_id="ack")
    for cancel in [
        CancelTaskRequest(id="foreign-task", tenant=accepted.tenant),
        CancelTaskRequest(id=accepted.remote_task_id, tenant="foreign-tenant"),
    ]:
        with pytest.raises(OutboundDispatchConflict, match="safe acknowledged route"):
            await outbound.begin_cancellation(
                accepted, cancel, operation_id="bad-cancel"
            )
    foreign = OutboundStore(
        outbound.backend, sender_scope=NativeRecipientScope("did:foreign")
    )
    with pytest.raises(OutboundDispatchConflict, match="another sender"):
        await foreign.begin_cancellation(
            accepted,
            CancelTaskRequest(id=accepted.remote_task_id, tenant=accepted.tenant),
            operation_id="foreign-cancel",
        )
    assert await outbound.get(original.dispatch_id) == accepted
