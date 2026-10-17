"""Physical PostgreSQL COMMIT survives every proxy ingress/completion owner."""

import asyncio

import pytest

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import (
    bind_execution_custody,
    execution_commit_outcome,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.integration.test_execution_authority_postgres import (
    GenerationFence,
    native_pg as native_pg,
)
from tests.integration.test_execution_custody_review30_pg import lose_ack
from tests.unit.test_execution_custody_proxy_ingress import owner
from tests.unit.test_execution_custody_review29 import retire

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("route", ["deferred", "legacy", "ack", "retry"])
async def test_native_commit_receipt_reaches_terminal_owner_after_worker_removal(
    native_pg, monkeypatch, tmp_path, cancelled, route
):
    backend, observer, schema = native_pg
    db = AsyncDatabase(backend)

    async def narrower_native_effect():
        # Reader callbacks are independent resident inbound events, not
        # continuations of this fixture's outer publication admission. The
        # actual native effect owns its narrower original generation fence.
        with bind_execution_custody(GenerationFence()):
            await db.execute("INSERT INTO effects (id) VALUES (1)")

    agent, _, feature, client, event, _, release = await owner(
        monkeypatch,
        tmp_path,
        None,
        completion_failure=route in {"ack", "retry"},
        native_database=db,
        native_effect=narrower_native_effect,
    )
    commit_entered, commit_release, completed = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    lose_ack(
        monkeypatch, cancelled=cancelled, entered=commit_entered, release=commit_release
    )
    task = None
    try:
        with bind_execution_custody(GenerationFence()):
            if route == "deferred":
                await feature._traffic_gate.close()
                await client.event_handler(event)
                (task,) = tuple(feature._deferred_acknowledged_event_tasks)
                await feature._traffic_gate.reopen()
            elif route == "legacy":
                await client.event_handler(
                    {"type": "channel.inbound", "payload": event["payload"]["message"]}
                )
                (task,) = tuple(feature._event_ingress_tasks)
            else:
                data = event["payload"]
                request = (
                    feature._split_inbound_event_acknowledgement(data)[1]
                    if route == "ack"
                    else feature._inbound_event_retry_completion(data)
                )
                feature._schedule_event_ingress_completion(
                    client,
                    request,
                    kind=route,
                    expected_state="acknowledged" if route == "ack" else "retrying",
                )
                (task,) = tuple(feature._event_ack_tasks)
            task.add_done_callback(lambda done: completed.set())
            await asyncio.wait_for(commit_entered.wait(), 5)
            # Independent physical observer proves the effect before losing
            # its ACK. This is neither a mocked commit nor synthetic error.
            assert (
                await observer.fetchval(f'SELECT COUNT(*) FROM "{schema}".effects') == 1
            )
            commit_release.set()
            await asyncio.wait_for(completed.wait(), 5)
        result = await await_owned_task(task)
        assert result.error is not None, "native COMMIT uncertainty was discarded"
        assert execution_commit_outcome(result.error) == "unknown"
        assert not feature._event_ingress_tasks
        assert not feature._event_ack_tasks
        assert not feature._event_ack_clients
        assert not feature._deferred_acknowledged_event_tasks
        assert client.completions == (
            []
            if route in {"deferred", "legacy"}
            else ["telegram-polling-ack" if route == "ack" else "telegram-polling-nack"]
        )
        with pytest.raises(BaseException) as caught:
            await feature.shutdown()
        assert caught.value is result.error
        assert execution_commit_outcome(caught.value) == "unknown"
        assert client.stopped
        assert feature._client is None
    finally:
        release.set()
        commit_release.set()
        if task is not None:
            await await_owned_task(task)
        await retire(agent)
