"""Follow-up messages retain one remote task owner and a fenced local lineage."""

from dataclasses import replace
from uuid import uuid4

import pytest
from a2a.types import AgentInterface, CancelTaskRequest, TaskState

from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.a2a.stores.unified.outbound_store import (
    DispatchState,
    OutboundDispatchConflict,
    OutboundStore,
)
from tests.unit.test_a2a_native_outbound_store import (
    outbound as outbound,  # noqa: PLC0414
)
from tests.unit.test_a2a_native_outbound_store import (
    request,
    reserve,
    task,
)


async def acknowledged(store):
    params = request()
    row = await reserve(store, params)
    sending = await store.begin_submission(row, operation_id=uuid4().hex)
    return await store.acknowledge(sending, task(params), operation_id=uuid4().hex)


async def follow_up(store, parent, *, params=None, **overrides):
    params = params or request(context=parent.remote_context_id, tenant=parent.tenant)
    if not params.message.task_id:
        params.message.task_id = parent.remote_task_id
    arguments = {
        "recipient": NativeRecipientScope(parent.recipient_id, parent.tenant),
        "signing_did": parent.signing_did,
        "interface": AgentInterface(
            url=parent.interface_url,
            protocol_binding=parent.protocol_binding,
            protocol_version="1.0",
            tenant=parent.tenant,
        ),
        "dispatch_tool": "send_a2a_task",
        "verb": "task",
        "parent": parent,
    }
    arguments.update(overrides)
    row = await store.reserve(uuid4().hex, params, **arguments)
    return row, params


async def test_follow_ups_keep_distinct_attempts_and_one_persistent_remote_owner(
    outbound,
):
    root = await acknowledged(outbound)
    assert root.root_dispatch_id == root.dispatch_id
    assert root.parent_dispatch_id is None
    first, params = await follow_up(outbound, root)
    assert first.remote_task_id == root.remote_task_id
    assert first.remote_context_id == root.remote_context_id
    sending = await outbound.begin_submission(first, operation_id=uuid4().hex)
    accepted = await outbound.acknowledge(
        sending, task(params), operation_id=uuid4().hex
    )
    restarted = OutboundStore(outbound.backend, sender_scope=outbound.sender_scope)
    stored = await restarted.get(accepted.dispatch_id)
    next_row, _ = await follow_up(restarted, stored)
    assert next_row.root_dispatch_id == root.dispatch_id
    assert next_row.parent_dispatch_id == accepted.dispatch_id
    assert len(await restarted.list_recent()) == 3
    assert {row.remote_task_id for row in await restarted.list_recent()} == {
        root.remote_task_id
    }


@pytest.mark.parametrize(
    "fault",
    [
        "context",
        "task",
        "recipient",
        "signing",
        "interface",
        "question",
        "forged-parent",
        "foreign-owner",
    ],
)
async def test_bad_continuation_cannot_claim_a_retained_route(outbound, fault):
    root = await acknowledged(outbound)
    params = request(context=root.remote_context_id, tenant=root.tenant)
    overrides = {}
    store = outbound
    if fault == "context":
        params.message.context_id = "other-context"
    elif fault == "task":
        params.message.task_id = "other-task"
    elif fault == "recipient":
        overrides["recipient"] = NativeRecipientScope("did:replacement", root.tenant)
    elif fault == "signing":
        overrides["signing_did"] = "did:web:replacement.test"
    elif fault == "interface":
        overrides["interface"] = AgentInterface(
            url="https://replacement.test/a2a",
            protocol_binding="JSONRPC",
            protocol_version="1.0",
            tenant=root.tenant,
        )
    elif fault == "question":
        overrides["verb"] = "question"
    elif fault == "forged-parent":
        root = replace(root, operation_id="forged-parent-snapshot")
    else:
        store = OutboundStore(
            outbound.backend, sender_scope=NativeRecipientScope("did:another-sender")
        )
    with pytest.raises(OutboundDispatchConflict):
        await follow_up(store, root, params=params, **overrides)
    assert len(await outbound.list_recent()) == 1, (
        "An untrusted continuation entered the sender ledger"
    )


async def test_only_one_follow_up_can_own_a_parent_until_proven_not_sent(outbound):
    root = await acknowledged(outbound)
    first, _ = await follow_up(outbound, root)
    with pytest.raises(OutboundDispatchConflict):
        await follow_up(outbound, root)
    sending = await outbound.begin_submission(first, operation_id=uuid4().hex)
    await outbound.record_no_delivery(sending, operation_id=uuid4().hex)
    replacement, _ = await follow_up(outbound, root)
    assert replacement.dispatch_id != first.dispatch_id
    assert replacement.root_dispatch_id == root.dispatch_id


async def test_canceling_any_ancestor_fences_a_reserved_follow_up(outbound):
    root = await acknowledged(outbound)
    first, params = await follow_up(outbound, root)
    sending = await outbound.begin_submission(first, operation_id=uuid4().hex)
    accepted = await outbound.acknowledge(
        sending, task(params), operation_id=uuid4().hex
    )
    last, _ = await follow_up(outbound, accepted)
    await outbound.begin_cancellation(
        root,
        CancelTaskRequest(id=root.remote_task_id, tenant=root.tenant),
        operation_id=uuid4().hex,
    )
    with pytest.raises(OutboundDispatchConflict, match="lineage"):
        await outbound.require_continuation(last)
    assert (await outbound.get(last.dispatch_id)).state is DispatchState.RESERVED


async def test_terminal_parent_refuses_follow_up_without_a_new_reservation(outbound):
    root = await acknowledged(outbound)
    params = request(context=root.remote_context_id, tenant=root.tenant)
    final = await outbound.acknowledge(
        root,
        task(params, state=TaskState.TASK_STATE_COMPLETED),
        operation_id=uuid4().hex,
    )
    with pytest.raises(OutboundDispatchConflict, match="lineage"):
        await follow_up(outbound, final)
    assert len(await outbound.list_recent()) == 1


async def test_follow_up_receipt_cannot_introduce_a_new_remote_task(outbound):
    root = await acknowledged(outbound)
    first, params = await follow_up(outbound, root)
    sending = await outbound.begin_submission(first, operation_id=uuid4().hex)
    with pytest.raises(OutboundDispatchConflict, match="acknowledged task"):
        await outbound.acknowledge(
            sending, task(params, task_id="replacement-task"), operation_id=uuid4().hex
        )
    assert await outbound.get(sending.dispatch_id) == sending


async def test_concurrent_follow_ups_from_independent_connections_have_one_owner(
    outbound, native_a2a_backend_factory
):
    import asyncio

    root = await acknowledged(outbound)
    second_backend = native_a2a_backend_factory()
    await second_backend.connect()
    second = OutboundStore(second_backend, sender_scope=outbound.sender_scope)

    async def attempt(store):
        try:
            return await follow_up(store, root)
        except OutboundDispatchConflict:
            return None

    try:
        results = await asyncio.gather(attempt(outbound), attempt(second))
        assert sum(result is not None for result in results) == 1, (
            "Two follow-ups claimed one parent"
        )
        assert len(await outbound.list_recent()) == 2
    finally:
        await second_backend.close()


@pytest.mark.parametrize(
    "fault", ["cycle", "root", "unrooted", "route", "task", "context"]
)
async def test_retained_lineage_is_revalidated_from_current_rows(outbound, fault):
    root = await acknowledged(outbound)
    child, _ = await follow_up(outbound, root)
    changes = {
        "cycle": ("parent_dispatch_id", child.dispatch_id),
        "route": ("interface_url", "https://replacement.test/a2a"),
        "task": ("remote_task_id", "replacement-task"),
        "context": ("remote_context_id", "replacement-context"),
    }
    if fault in {"root", "unrooted"}:
        another = await reserve(outbound, request(context="another-context"))
        field, value = "root_dispatch_id", another.dispatch_id
        if fault == "unrooted":
            await outbound.backend.execute(
                "UPDATE a2a_outbound_dispatches SET root_dispatch_id = ? WHERE sender_id = ? AND sender_tenant = ? AND dispatch_id = ?",
                (value, child.sender_id, child.sender_tenant, child.dispatch_id),
            )
            child = await outbound.get(child.dispatch_id)
    else:
        field, value = changes[fault]
    await outbound.backend.execute(
        f"UPDATE a2a_outbound_dispatches SET {field} = ? WHERE sender_id = ? AND sender_tenant = ? AND dispatch_id = ?",
        (value, root.sender_id, root.sender_tenant, root.dispatch_id),
    )
    with pytest.raises(OutboundDispatchConflict):
        await outbound.require_continuation(child)
    assert (await outbound.get(child.dispatch_id)).state is DispatchState.RESERVED


async def test_postgres_reservation_locks_its_ancestors_before_insert(
    outbound, monkeypatch
):
    if not outbound.is_postgres:
        pytest.skip("PostgreSQL row-lock contract")
    root = await acknowledged(outbound)
    actual = outbound.require_continuation
    observed = []

    async def inspect_lock(dispatch, **arguments):
        result = await actual(dispatch, **arguments)
        if arguments.get("lock"):
            rows = await outbound.backend.fetch_all(
                "SELECT held.mode FROM pg_locks held "
                "JOIN pg_class relation ON relation.oid = held.relation "
                "JOIN pg_namespace namespace ON namespace.oid = relation.relnamespace "
                "WHERE held.pid = pg_backend_pid() AND held.granted "
                "AND namespace.nspname = current_schema() "
                "AND relation.relname = 'a2a_outbound_dispatches'"
            )
            observed.append({row[0] for row in rows})
        return result

    monkeypatch.setattr(outbound, "require_continuation", inspect_lock)
    await follow_up(outbound, root)
    assert observed and all("RowShareLock" in locks for locks in observed), (
        "Continuation insertion did not retain its ancestor row locks"
    )
