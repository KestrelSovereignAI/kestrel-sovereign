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
from kestrel_sovereign.signals.sources import a2a, a2a_question_answered, a2a_task_submitted, channels
from tests.unit.test_durable_signal_delivery import (
    _channel_dispatcher,
    _channel_signal,
    _close,
)
from tests.utils.original_failure import hold_original_sqlite_commit_ack


@pytest.fixture(params=[channels, a2a, a2a_task_submitted, a2a_question_answered],
                ids=lambda source: source.SOURCE_NAME)
def cognition_source(request):
    return request.param


@pytest.fixture
def cognition_registration(cognition_source):
    builders = {
        channels: channels.build_channel_message_registration,
        a2a: a2a.build_a2a_task_complete_registration,
        a2a_task_submitted: a2a_task_submitted.build_a2a_task_submitted_registration,
        a2a_question_answered: a2a_question_answered.build_a2a_question_answered_registration,
    }
    return builders[cognition_source]()


def cognition_signal(source, agent_id):
    signal = _channel_signal(agent_id, "original")
    if source is a2a:
        payload = dict(task_id="task1", task_state="completed", result_summary="owned result")
    elif source is a2a_task_submitted:
        payload = dict(task_id="task1", session_id="context1", sender="did:peer:original",
                       input_revision=0, request_content="owned task")
    elif source is a2a_question_answered:
        payload = dict(question_id="question1", dispatch_id="dispatch1", recipient_id="did:peer:original",
                       original_question="owned question", reply_text="owned answer", state="completed",
                       request={"messageId": "question1", "role": "ROLE_USER", "parts": [{"text": "owned question"}]})
    else:
        payload = signal.payload
    return replace(signal, source=source.SOURCE_NAME, payload=payload)


@asynccontextmanager
async def owned_delivery(tmp_path, cognition_source):
    backend, agent, dispatcher = await _channel_dispatcher(
        tmp_path / "cognition.db", "did:agent:shutdown-terminal"
    )
    try:
        await dispatcher._stop_runtime_owner_heartbeat()
        ledger = dispatcher.durable_store
        await ledger.register_consumer(DurableConsumerRegistration(
            agent_id=agent.did, consumer_id=cognition_source.DURABLE_COGNITION_CONSUMER_ID,
            source=cognition_source.SOURCE_NAME, max_attempts=3,
        ))
        await ledger.persist_signal(
            cognition_signal(cognition_source, agent.did), agent_id=agent.did,
            source_event_id="original", retention_days=1,
        )
        delivery = await ledger.claim_delivery(
            agent_id=agent.did, consumer_id=cognition_source.DURABLE_COGNITION_CONSUMER_ID,
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
    tmp_path, monkeypatch, phase, cognition_source,
):
    async with owned_delivery(tmp_path, cognition_source) as (backend, agent, dispatcher, delivery):
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
    tmp_path, monkeypatch, mismatch, cognition_source,
):
    async with owned_delivery(tmp_path, cognition_source) as (backend, _agent, dispatcher, delivery):
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
    tmp_path, terminal_ackable, cognition_source,
):
    async with owned_delivery(tmp_path, cognition_source) as (backend, agent, dispatcher, delivery):
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


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["nack", "managed", "hold"])
async def test_original_sqlite_retry_ack_excludes_local_and_fresh_claims(
    tmp_path, monkeypatch, cognition_source, kind,
):
    from kestrel_sovereign.signals.durable import DurableSignalStore
    from kestrel_sovereign.storage.db.sqlite import SQLiteBackend

    async with owned_delivery(tmp_path, cognition_source) as (backend, agent, dispatcher, delivery):
        sibling = SQLiteBackend(str(tmp_path / "cognition.db"))
        await sibling.connect()
        fresh = DurableSignalStore(sibling)
        ledger = dispatcher.durable_store
        operation = competing = None
        arguments = dict(agent_id=agent.did, consumer_id=delivery.consumer_id,
                         delivery_id=delivery.delivery_id, lease_token=delivery.lease_token)
        try:
            with hold_original_sqlite_commit_ack(
                monkeypatch, lambda *_: asyncio.current_task() is operation,
            ) as acknowledgement:
                if kind == "hold":
                    release = ledger.release_delivery_for_hold(**arguments)
                elif kind == "nack":
                    release = ledger.nack_delivery(**arguments, error="original retry")
                else:
                    release = ledger.release_managed_delivery_after_task(
                        **arguments, owner_id=dispatcher._durable_delivery_owner, error="original retained retry",
                    )
                operation = create_owned_task(release, retain_originals=True)
                await asyncio.wait_for(acknowledgement.entered.wait(), 5)
                # The original backend still owns its COMMIT return gate.
                # Evaluate its actual local exclusion predicate through the
                # independent connection instead of deadlocking behind it.
                assert ledger._retry_release_receipt_pending(agent.did, delivery.consumer_id, delivery.delivery_id)
                predicate, parameters = ledger._retry_receipt_exclusion_sql()
                import aiosqlite
                async with aiosqlite.connect(tmp_path / "cognition.db") as reader:
                    async with reader.execute(
                        f"SELECT delivery_id FROM durable_signal_deliveries WHERE {predicate}", parameters,
                    ) as cursor:
                        assert await cursor.fetchone() is None
                # A sibling backend can also serialize behind the original
                # file owner's acknowledgement. Its eventual native claim
                # must still refuse the original receipt-pending capability.
                competing = create_owned_task(fresh.claim_delivery_for_event(
                    agent_id=agent.did, consumer_id=delivery.consumer_id,
                    event_id=delivery.event_id, executor_id="dispatcher:sibling",
                ), retain_originals=True)
                await asyncio.sleep(0)
                if competing.done():
                    assert competing.result() is None
                operation.cancel("original successful release lost its COMMIT acknowledgement")
                outcome = await await_owned_task(operation)
            assert execution_commit_outcome(outcome.error) == "unknown"
            competitor = await await_owned_task(competing)
            assert competitor.error is None and competitor.result is None
            assert await fresh.claim_delivery(
                agent_id=agent.did, consumer_id=delivery.consumer_id, executor_id="dispatcher:sibling",
            ) is None
            dispatcher._retain_cognition_native_control(delivery, outcome.error)
            dispatcher._durable_shutdown = True
            await dispatcher._drain_retained_durable_cognition_cleanup_tasks()
            assert await sibling.fetch_one(
                "SELECT status, lease_owner, lease_token, acknowledged_at FROM durable_signal_deliveries"
            ) == ("failed", None, None, None)
            assert not dispatcher._retained_cognition_control_debt
        finally:
            if operation is not None:
                await await_owned_task(operation)
            if competing is not None:
                await await_owned_task(competing)
            await sibling.close()
