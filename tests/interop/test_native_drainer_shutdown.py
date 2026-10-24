"""A real resident claim must finish its original COMMIT before shutdown joins it."""

import asyncio

import aiosqlite
import pytest
from kestrel_sdk.signals import (
    RedactionPolicy,
    Signal,
    SignalMode,
    SourceRegistration,
    Trust,
)

from kestrel_sovereign.signals.durable import DurableConsumerRegistration


async def read_owned_delivery(environment, consumer):
    """Inspect the same file after either public close entry point."""
    database = environment.root / "agent" / "kestrel_prime.db"
    async with (
        aiosqlite.connect(database.as_uri() + "?mode=ro", uri=True) as reader,
        reader.execute(
            "SELECT status, attempts, last_error FROM durable_signal_deliveries WHERE consumer_id=?", (consumer,),
        ) as cursor,
    ):
        return await cursor.fetchone()


@pytest.mark.timeout(120)
async def test_shutdown_still_closes_the_primary_after_early_close_publication_fails(native_go_environment, monkeypatch):
    agent = native_go_environment.recipient
    dispatcher = agent.dispatcher
    backend = dispatcher.durable_store.backend
    original = dispatcher.shutdown
    failure = RuntimeError("original early shutdown publication failed")
    calls = 0

    def publish_then_fail_once():
        nonlocal calls
        calls += 1
        original()
        if calls == 1:
            raise failure

    monkeypatch.setattr(dispatcher, "shutdown", publish_then_fail_once)
    with pytest.raises(RuntimeError) as caught:
        await agent.shutdown()
    assert caught.value is failure
    assert dispatcher._durable_shutdown
    assert not backend.is_connected, "early close failure skipped the agent resource sweep"


@pytest.mark.timeout(120)
@pytest.mark.parametrize("entrypoint", ["dispatcher", "agent"])
async def test_shutdown_joins_an_original_resident_claim_commit(native_go_environment, monkeypatch, entrypoint):
    agent = native_go_environment.recipient
    dispatcher = agent.dispatcher
    ledger = dispatcher.durable_store
    consumer, source = "native-shutdown-commit", "a2a.task_complete"
    assert source in agent.durable_rehydratable_sources
    await ledger.register_consumer(DurableConsumerRegistration(
        consumer_id=consumer, source=source, agent_id=agent.did,
    ))
    await ledger.persist_signal(
        Signal(source=source, kind="claim", mode=SignalMode.COGNITION, payload={}, target_agent=agent.did),
        agent_id=agent.did, source_event_id="original-shutdown-claim", retention_days=1,
    )
    original = aiosqlite.Connection._execute
    commit_entered, release_commit, shutdown_entered = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def hold_original_commit_ack(connection, function, *args, **kwargs):
        result = await original(connection, function, *args, **kwargs)
        task = asyncio.current_task()
        if function.__name__ == "commit" and task in set(dispatcher._durable_cognition_drainers.values()):
            # The actual SQLite worker has committed. Retain its original
            # reply here to make the shutdown/claim boundary deterministic.
            commit_entered.set()
            await release_commit.wait()
        return result

    monkeypatch.setattr(aiosqlite.Connection, "_execute", hold_original_commit_ack)
    await dispatcher.start_durable_cognition_consumer(consumer)
    await asyncio.wait_for(commit_entered.wait(), 5)
    drainer = dispatcher._durable_cognition_drainers[consumer]
    original_shutdown = dispatcher._complete_durable_shutdown

    async def observe_shutdown():
        shutdown_entered.set()
        return await original_shutdown()

    monkeypatch.setattr(dispatcher, "_complete_durable_shutdown", observe_shutdown)
    closing = asyncio.create_task(
        agent.shutdown() if entrypoint == "agent" else dispatcher.shutdown_durable_delivery()
    )
    try:
        await asyncio.wait_for(shutdown_entered.wait(), 5)
        assert drainer.cancelling() == 0, "shutdown cancelled an actual claim COMMIT before its original reply"
        assert not closing.done()
    finally:
        release_commit.set()
        await asyncio.gather(closing, return_exceptions=True)
    assert drainer.done()
    drainer.result()
    closing.result()
    assert dispatcher._outcome_log_control_error is None
    assert await read_owned_delivery(native_go_environment, consumer) == (
        "leased", 1, "cognition admission pending",
    )


@pytest.mark.timeout(120)
async def test_shutdown_cancels_only_the_drainer_that_never_crossed_its_ready_barrier(native_go_environment):
    agent = native_go_environment.recipient
    dispatcher = agent.dispatcher
    await dispatcher.register_durable_consumer(DurableConsumerRegistration(
        consumer_id="unentered-shutdown", source="a2a.task_complete", agent_id=agent.did,
    ))
    ready = agent._runtime_publication_ready
    barrier = asyncio.Event()
    agent._runtime_publication_ready = barrier
    try:
        await dispatcher.start_durable_cognition_consumer("unentered-shutdown")
    finally:
        agent._runtime_publication_ready = ready
    drainer = dispatcher._durable_cognition_drainers["unentered-shutdown"]
    await asyncio.sleep(0)
    assert drainer not in dispatcher._entered_durable_cognition_drainers
    try:
        dispatcher.shutdown()
        await asyncio.sleep(0)
        assert drainer.cancelling() == 1, "unentered wrapper was not cancelled"
        await asyncio.wait_for(dispatcher.shutdown_durable_delivery(), 5)
        assert drainer.cancelled()
        assert dispatcher._outcome_log_control_error is None
    finally:
        barrier.set()
        await asyncio.gather(drainer, return_exceptions=True)


async def prepare_owned_cognition(environment):
    agent, dispatcher = environment.recipient, environment.recipient.dispatcher
    source, consumer = "native.shutdown.cognition", "native-shutdown-body"
    prompt = environment.root / "shutdown-cognition.md"
    prompt.write_text("Owned shutdown cognition test")
    dispatcher._registry.register(SourceRegistration(
        name=source, schema=dict, default_mode=SignalMode.COGNITION,
        allowed_modes=frozenset({SignalMode.COGNITION}), prompt_template=prompt,
        trust=Trust.TRUSTED,
        log_redaction=RedactionPolicy(summarize=lambda payload: "<shutdown cognition>"),
        retention_days=1,
    ))
    await dispatcher.register_durable_consumer(DurableConsumerRegistration(
        consumer_id=consumer, source=source, agent_id=agent.did,
    ))
    ledger = dispatcher.durable_store
    signal = Signal(source=source, kind="shutdown-body", mode=SignalMode.COGNITION,
                    payload={}, caller="did:kestrel:owned-test", target_agent=agent.did)
    await ledger.persist_signal(
        signal, agent_id=agent.did, source_event_id="original-shutdown-body", retention_days=1,
        caller_identity_factory=lambda: dispatcher._protect_durable_caller_identity(signal),
    )
    return agent, dispatcher, ledger, consumer


@pytest.mark.timeout(120)
@pytest.mark.parametrize("entrypoint", ["dispatcher", "agent"])
async def test_shutdown_drains_the_original_successful_ack_commit(native_go_environment, monkeypatch, entrypoint):
    agent, dispatcher, _ledger, consumer = await prepare_owned_cognition(native_go_environment)
    original = aiosqlite.Connection._execute
    entered, release = asyncio.Event(), asyncio.Event()
    ack_owner = None

    async def hold_original_ack(connection, function, *args, **kwargs):
        nonlocal ack_owner
        result = await original(connection, function, *args, **kwargs)
        task = asyncio.current_task()
        if function.__name__ == "execute" and len(args) > 1 and "UPDATE durable_signal_deliveries" in str(args[0]):
            parameters = args[1]
            if parameters and parameters[0] == "acknowledged":
                ack_owner = task
        if function.__name__ == "commit" and ack_owner is task:
            entered.set()
            await release.wait()
        return result

    monkeypatch.setattr(aiosqlite.Connection, "_execute", hold_original_ack)
    closing = None
    try:
        await dispatcher.start_durable_cognition_consumer(consumer)
        await asyncio.wait_for(entered.wait(), 5)
        assert ack_owner is not None and not ack_owner.done()
        assert dispatcher._durable_cognition_drainers[consumer] in dispatcher._durable_admission_owners
        closing = asyncio.create_task(
            agent.shutdown() if entrypoint == "agent" else dispatcher.shutdown_durable_delivery()
        )
        await asyncio.sleep(0)
        assert ack_owner.cancelling() == 0 and not closing.done()
    finally:
        release.set()
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
    assert ack_owner.done()
    ack_owner.result()
    closing.result()
    assert dispatcher._outcome_log_control_error is None
    # Full Agent close has retired its original backend. Read the same owned
    # file through a separately joined read-only connection for the oracle.
    assert (await read_owned_delivery(native_go_environment, consumer))[:2] == ("acknowledged", 1)


@pytest.mark.timeout(120)
async def test_a_completed_cognition_keeps_its_original_ack_admission_after_close(native_go_environment, monkeypatch):
    _agent, dispatcher, _ledger, consumer = await prepare_owned_cognition(native_go_environment)
    completed = asyncio.Event()
    original = dispatcher._completed_durable_cognition_result

    def close_after_original_body(task, **kwargs):
        result = original(task, **kwargs)
        dispatcher.shutdown()
        completed.set()
        return result

    monkeypatch.setattr(dispatcher, "_completed_durable_cognition_result", close_after_original_body)
    await dispatcher.start_durable_cognition_consumer(consumer)
    await asyncio.wait_for(completed.wait(), 5)
    await asyncio.wait_for(dispatcher.shutdown_durable_delivery(), 5)
    assert (await read_owned_delivery(native_go_environment, consumer))[:2] == ("acknowledged", 1)
    assert dispatcher._outcome_log_control_error is None


@pytest.mark.timeout(120)
async def test_shutdown_does_not_claim_after_an_original_candidate_read(native_go_environment, monkeypatch):
    _agent, dispatcher, _ledger, consumer = await prepare_owned_cognition(native_go_environment)
    entered, release = asyncio.Event(), asyncio.Event()
    original_hold_read = dispatcher._agent_is_held

    async def hold_original_candidate_read():
        result = await original_hold_read()
        entered.set()
        await release.wait()
        return result

    monkeypatch.setattr(dispatcher, "_agent_is_held", hold_original_candidate_read)
    closing = None
    try:
        await dispatcher.start_durable_cognition_consumer(consumer)
        await asyncio.wait_for(entered.wait(), 5)
        dispatcher.shutdown()
        closing = asyncio.create_task(dispatcher.shutdown_durable_delivery())
        release.set()
        await asyncio.wait_for(asyncio.shield(closing), 5)
        assert await read_owned_delivery(native_go_environment, consumer) == ("pending", 0, None)
    finally:
        release.set()
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)


@pytest.mark.timeout(120)
@pytest.mark.parametrize("resist_cancellation", [False, True])
async def test_shutdown_stops_cognition_and_retains_its_exact_lease_until_safe(
    native_go_environment, monkeypatch, resist_cancellation,
):
    from kestrel_sovereign.signals import dispatcher as dispatcher_module

    agent, dispatcher, ledger, consumer = await prepare_owned_cognition(native_go_environment)
    entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original_process_input = agent.process_input

    async def paused_input(*args, **kwargs):
        entered.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                if not resist_cancellation:
                    raise
        return await original_process_input(*args, **kwargs)

    monkeypatch.setattr(agent, "process_input", paused_input)
    monkeypatch.setattr(dispatcher_module, "_DURABLE_COGNITION_CANCELLATION_GRACE", 0.01)
    closing = None
    try:
        await dispatcher.start_durable_cognition_consumer(consumer)
        await asyncio.wait_for(entered.wait(), 5)
        drainer = dispatcher._durable_cognition_drainers[consumer]
        original_lease = await ledger.backend.fetch_one(
            "SELECT lease_token, status, attempts FROM durable_signal_deliveries WHERE consumer_id=?", (consumer,),
        )
        assert original_lease[0] and original_lease[1:] == ("leased", 1)
        closing = asyncio.create_task(dispatcher.shutdown_durable_delivery())
        cancellation_notice = asyncio.create_task(cancelled.wait())
        try:
            observed, _ = await asyncio.wait({cancellation_notice}, timeout=5)
            assert cancellation_notice in observed, "shutdown never requested cancellation of the actual cognition body"
        finally:
            cancellation_notice.cancel()
            await asyncio.gather(cancellation_notice, return_exceptions=True)
        completed = await asyncio.wait_for(asyncio.shield(closing), 5)
        assert completed is not resist_cancellation
        assert drainer.done() and not drainer.cancelled()
        drainer.result()
        assert await ledger.backend.fetch_one(
            "SELECT lease_token, status, attempts FROM durable_signal_deliveries WHERE consumer_id=?", (consumer,),
        ) == original_lease
        owner = await ledger.backend.fetch_one(
            "SELECT stopped_at FROM durable_signal_runtime_owners WHERE agent_id=? AND owner_id=?",
            (agent.did, dispatcher._durable_delivery_owner),
        )
        assert (owner[0] is None) is resist_cancellation
        assert dispatcher.durable_shutdown_owner_fenced is resist_cancellation
        release.set()
        await asyncio.wait_for(dispatcher.wait_for_durable_shutdown_release(), 10)
        owner = await ledger.backend.fetch_one(
            "SELECT stopped_at FROM durable_signal_runtime_owners WHERE agent_id=? AND owner_id=?",
            (agent.did, dispatcher._durable_delivery_owner),
        )
        assert owner[0] is not None
        assert dispatcher._outcome_log_control_error is None
    finally:
        release.set()
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
            await dispatcher.wait_for_durable_shutdown_release()
