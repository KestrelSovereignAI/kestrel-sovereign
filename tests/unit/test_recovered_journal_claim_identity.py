"""Public rehydration cannot mint journal identity outside its real claim."""
import asyncio
import time
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from kestrel_sdk.signals import CausationFrame, SignalMode
from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import ExecutionCustody, bind_execution_runtime
from kestrel_sovereign.signals import DurableConsumerRegistration
from kestrel_sovereign.signals.sources.channels import DURABLE_COGNITION_CONSUMER_ID
from tests.unit.test_durable_signal_delivery import _channel_dispatcher, _channel_signal, _close
from tests.unit.test_signal_outcome_terminal_custody import _OriginalWork


@pytest.mark.asyncio
@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('substitution', ['source', 'target_agent', 'mode', 'registration',
                                        'id', 'arrived_at', 'payload', 'causation_chain'])
async def test_public_rehydration_cannot_replace_claimed_journal_identity(tmp_path, substitution, asynchronous):
    backend, agent, dispatcher = await _channel_dispatcher(tmp_path / 'original-claim.db', 'did:agent:original-claim')
    effects, hooks = [], []
    scope = ExecutionCustody(_OriginalWork())
    agent._execution_custody = scope
    task = None
    try:
        await dispatcher.register_durable_consumer(DurableConsumerRegistration(
            consumer_id=DURABLE_COGNITION_CONSUMER_ID, source='channel.message',
            agent_id=agent.did, max_attempts=0))
        canonical = _channel_signal(agent.did, 'original-claimed-event')
        persisted = await dispatcher._durable_store.persist_signal(canonical, agent_id=agent.did,
            source_event_id='original-claimed-event', retention_days=7, caller_identity=None)
        delivery = await dispatcher.claim_durable_delivery_for_event(
            consumer_id=DURABLE_COGNITION_CONSUMER_ID, event_id=persisted.event_id,
            executor_id=dispatcher._durable_delivery_owner)
        assert delivery is not None and delivery.lease_token
        changed = {'source': 'unregistered.foreign.source', 'target_agent': 'did:agent:foreign',
                   'mode': SignalMode.ACTION}.get(substitution)
        detached = substitution in ('id', 'arrived_at', 'payload', 'causation_chain')

        def rehydrate(event, *, dispatch_signal):
            hooks.append(event.event_id)
            if substitution == 'registration':
                assert dispatcher._registry.unregister(canonical.source)
                return canonical
            if detached:
                if substitution == 'id':
                    dispatch_signal.id = 'sig_foreign_audit_row'
                elif substitution == 'arrived_at':
                    dispatch_signal.arrived_at = datetime(2000, 1, 1, tzinfo=timezone.utc)
                elif substitution == 'payload':
                    dispatch_signal.payload['hook-owned'] = True
                else:
                    dispatch_signal.causation_chain.append(CausationFrame(agent_id=agent.did,
                        source='public.hook', signal_id=canonical.id, turn_id=None, depth=1,
                        emitted_at=datetime.now(timezone.utc)))
                return replace(canonical, id=dispatch_signal.id, arrived_at=dispatch_signal.arrived_at)
            return replace(canonical, **{substitution: changed})

        async def asynchronous_rehydrate(event, *, dispatch_signal):
            result = rehydrate(event, dispatch_signal=dispatch_signal)
            await asyncio.sleep(0)
            return result

        async def actual_effect(prompt):
            effects.append(True)
            if not detached:
                scope.revoke('actual effect finished before terminal journal')
            return 'actual completed effect'

        agent.rehydrate_durable_cognition_signal = asynchronous_rehydrate if asynchronous else rehydrate
        agent.durable_rehydratable_sources = frozenset({'channel.message'})
        agent.process_input = actual_effect
        incoming = dispatcher._durable_retry_seed(delivery.event)
        original_seed = (incoming.id, incoming.arrived_at, dict(incoming.payload), tuple(incoming.causation_chain))
        with bind_execution_runtime(SimpleNamespace(_execution_custody=scope)):
            task = asyncio.create_task(dispatcher._route_durable_cognition_delivery(incoming,
                dispatcher._registry.get(incoming.source), time.monotonic(),
                persisted_event_id=delivery.event_id, consumer_id=delivery.consumer_id,
                durable_admission=None, durable_created=False, use_live_signal=False,
                claimed_delivery=delivery))
        outcome = await await_owned_task(task)
        await dispatcher._drain_outcome_logs_for_signal(incoming.id)
        assert hooks == [persisted.event_id]
        rows = await backend.fetch_all('SELECT source,target_agent,mode FROM signal_log WHERE id=?', (incoming.id,))
        current, = await dispatcher.list_durable_deliveries()
        if detached:
            assert outcome.error is None and effects == [True]
            assert (incoming.id, incoming.arrived_at, incoming.payload, tuple(incoming.causation_chain)) == original_seed
            assert await backend.fetch_all('SELECT id FROM signal_log') == [(original_seed[0],)]
            assert current.status == 'acknowledged'
        else:
            assert not rows and not effects, 'rehydration substituted the claimed envelope before original journal acceptance'
            assert current.status == 'failed'
        assert current.lease_expires_at is None
        assert not dispatcher._retained_cognition_control_debt
    finally:
        if task is not None:
            await await_owned_task(task)
        await dispatcher._stop_runtime_owner_heartbeat()
        await _close(backend, agent)
