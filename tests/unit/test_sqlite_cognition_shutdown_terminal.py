"""Real SQLite control receipts must settle their exact lease after close.

The shared dispatcher fixture's routing double is never invoked. Native
COMMITs, lost acknowledgements, delivery transitions and recovery are real.
"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import execution_commit_outcome
from kestrel_sovereign.signals import DurableConsumerRegistration
from kestrel_sovereign.signals.sources.channels import DURABLE_COGNITION_CONSUMER_ID
from tests.unit.test_durable_signal_delivery import (
    _channel_dispatcher,
    _channel_signal,
    _close,
)
from tests.utils.original_failure import hold_original_sqlite_commit_ack


@asynccontextmanager
async def owned_delivery(tmp_path):
    backend, agent, dispatcher = await _channel_dispatcher(
        tmp_path / "cognition.db", "did:agent:shutdown-terminal"
    )
    try:
        await dispatcher._stop_runtime_owner_heartbeat()
        ledger = dispatcher.durable_store
        await ledger.register_consumer(DurableConsumerRegistration(
            agent_id=agent.did, consumer_id=DURABLE_COGNITION_CONSUMER_ID,
            source="channel.message", max_attempts=3,
        ))
        await ledger.persist_signal(
            _channel_signal(agent.did, "original"), agent_id=agent.did,
            source_event_id="original", retention_days=1,
        )
        delivery = await ledger.claim_delivery(
            agent_id=agent.did, consumer_id=DURABLE_COGNITION_CONSUMER_ID,
            executor_id=dispatcher._durable_delivery_owner,
        )
        assert delivery is not None and delivery.lease_token
        yield backend, agent, dispatcher, delivery
    finally:
        await dispatcher._stop_runtime_owner_heartbeat()
        await _close(backend, agent)


async def original_commit_uncertainty(backend, dispatcher, delivery, monkeypatch, phase):
    """Cancel one actual successful native COMMIT before its original ACK."""
    await backend.execute("CREATE TABLE effects (id INTEGER PRIMARY KEY)")
    operation = None
    with hold_original_sqlite_commit_ack(
        monkeypatch, lambda *_: asyncio.current_task() is operation,
    ) as acknowledgement:
        if phase == "leased":
            body = backend.execute("INSERT INTO effects VALUES (1)")
        else:
            body = dispatcher.durable_store.release_managed_delivery_after_task(
                agent_id=dispatcher._agent.did,
                owner_id=dispatcher._durable_delivery_owner,
                consumer_id=delivery.consumer_id, delivery_id=delivery.delivery_id,
                lease_token=delivery.lease_token, error="original retry release",
            )
        operation = create_owned_task(body, retain_originals=True)
        await asyncio.wait_for(acknowledgement.entered.wait(), 5)
        operation.cancel("original COMMIT acknowledgement lost")
        outcome = await await_owned_task(operation)
    assert execution_commit_outcome(outcome.error) == "unknown"
    if phase == "leased":
        assert await backend.fetch_one("SELECT id FROM effects") == (1,)
    row = await backend.fetch_one(
        "SELECT status, lease_owner, lease_token FROM durable_signal_deliveries"
    )
    assert row == (phase, dispatcher._durable_delivery_owner, delivery.lease_token)
    return outcome.error


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["leased", "retry"])
async def test_closed_sqlite_cognition_persists_original_control_before_owner_release(
    tmp_path, monkeypatch, phase,
):
    async with owned_delivery(tmp_path) as (backend, agent, dispatcher, delivery):
        ledger = dispatcher.durable_store
        original = await original_commit_uncertainty(
            backend, dispatcher, delivery, monkeypatch, phase
        )
        dispatcher._retain_cognition_native_control(delivery, original)
        dispatcher._durable_shutdown = True
        await dispatcher._drain_retained_durable_cognition_cleanup_tasks()
        row = await backend.fetch_one(
            "SELECT status, lease_owner, lease_token, lease_expires_at, "
            "next_attempt_at, last_error, terminal_at FROM durable_signal_deliveries"
        )
        assert row[:5] == ("failed", None, None, None, None)
        assert row[5].startswith("execution_control_unresolved: ExecutionCommitOutcomeError")
        assert row[6] is not None and not dispatcher._retained_cognition_control_debt
        with pytest.raises(RuntimeError, match="shutting down"):
            await dispatcher.release_durable_delivery_after_task(
                consumer_id=delivery.consumer_id, delivery_id=delivery.delivery_id,
                lease_token=delivery.lease_token, error="forbidden ordinary work",
            )
        await dispatcher.shutdown_durable_delivery()
        assert (await backend.fetch_one(
            "SELECT stopped_at FROM durable_signal_runtime_owners"
        ))[0] is not None
        now = datetime.now(timezone.utc)
        assert await ledger.recover_abandoned_leases(
            agent_id=agent.did, recovering_owner_id="dispatcher:successor",
            stale_before=now + timedelta(seconds=1), now=now,
        ) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch", ["owner", "token"])
async def test_closed_sqlite_cognition_retains_control_on_exact_identity_mismatch(
    tmp_path, monkeypatch, mismatch,
):
    async with owned_delivery(tmp_path) as (backend, _agent, dispatcher, delivery):
        original = await original_commit_uncertainty(
            backend, dispatcher, delivery, monkeypatch, "leased"
        )
        expected = await backend.fetch_one("SELECT * FROM durable_signal_deliveries")
        original_owner = dispatcher._durable_delivery_owner
        snapshot = delivery
        if mismatch == "owner":
            dispatcher._durable_delivery_owner = "dispatcher:unrelated"
        else:
            snapshot = replace(delivery, lease_token="unrelated-token")
        dispatcher._durable_shutdown = True
        dispatcher._retain_cognition_native_control(snapshot, original)
        with pytest.raises(BaseException) as caught:
            await dispatcher._drain_retained_durable_cognition_cleanup_tasks()
        assert execution_commit_outcome(caught.value) == "unknown"
        assert dispatcher._retained_cognition_control_debt
        assert await backend.fetch_one("SELECT * FROM durable_signal_deliveries") == expected
        dispatcher._durable_delivery_owner = original_owner


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_ackable", [None, True])
async def test_original_retry_capability_does_not_admit_release_or_cursor_ack(
    tmp_path, terminal_ackable,
):
    async with owned_delivery(tmp_path) as (backend, agent, dispatcher, delivery):
        ledger = dispatcher.durable_store
        arguments = dict(
            agent_id=agent.did, owner_id=dispatcher._durable_delivery_owner,
            consumer_id=delivery.consumer_id, delivery_id=delivery.delivery_id,
            lease_token=delivery.lease_token, error="original retry release",
        )
        retry = await ledger.release_managed_delivery_after_task(**arguments)
        assert retry.status == "retry" and retry.lease_token == delivery.lease_token
        expected = await backend.fetch_one("SELECT * FROM durable_signal_deliveries")
        if terminal_ackable is not None:
            arguments.update(terminal=True, terminal_ackable=terminal_ackable)
        assert await ledger.release_managed_delivery_after_task(**arguments) is None
        assert await backend.fetch_one("SELECT * FROM durable_signal_deliveries") == expected
