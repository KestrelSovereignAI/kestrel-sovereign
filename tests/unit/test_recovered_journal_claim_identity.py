"""Public rehydration cannot mint journal identity outside its real claim."""
import asyncio
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest
from kestrel_sdk.signals import SignalMode
from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import ExecutionCustody, bind_execution_runtime
from kestrel_sovereign.signals import DurableConsumerRegistration
from kestrel_sovereign.signals.sources.channels import DURABLE_COGNITION_CONSUMER_ID
from tests.unit.test_durable_signal_delivery import _channel_dispatcher, _channel_signal, _close
from tests.unit.test_signal_outcome_terminal_custody import _OriginalWork


@pytest.mark.asyncio
@pytest.mark.parametrize('substitution', ['source', 'target_agent', 'mode', 'registration'])
async def test_public_rehydration_cannot_replace_claimed_journal_identity(tmp_path, substitution):
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
                   'mode': SignalMode.ACTION, 'registration': None}[substitution]

        def rehydrate(event, *, dispatch_signal):
            hooks.append(event.event_id)
            if substitution == 'registration':
                assert dispatcher._registry.unregister(canonical.source)
                return canonical
            return replace(canonical, **{substitution: changed})

        async def actual_effect(prompt):
            effects.append(True)
            scope.revoke('actual effect finished before terminal journal')
            return 'actual completed effect'

        agent.rehydrate_durable_cognition_signal = rehydrate
        agent.durable_rehydratable_sources = frozenset({'channel.message'})
        agent.process_input = actual_effect
        incoming = dispatcher._durable_retry_seed(delivery.event)
        with bind_execution_runtime(SimpleNamespace(_execution_custody=scope)):
            task = asyncio.create_task(dispatcher._route_durable_cognition_delivery(incoming,
                dispatcher._registry.get(incoming.source), time.monotonic(),
                persisted_event_id=delivery.event_id, consumer_id=delivery.consumer_id,
                durable_admission=None, durable_created=False, use_live_signal=False,
                claimed_delivery=delivery))
        await await_owned_task(task)
        await dispatcher._drain_outcome_logs_for_signal(incoming.id)
        assert hooks == [persisted.event_id]
        rows = await backend.fetch_all('SELECT source,target_agent,mode FROM signal_log WHERE id=?', (incoming.id,))
        assert not rows and not effects, 'rehydration substituted the claimed envelope before original journal acceptance'
        current, = await dispatcher.list_durable_deliveries()
        assert current.status == 'failed' and current.lease_expires_at is None
        assert not dispatcher._retained_cognition_control_debt
    finally:
        if task is not None:
            await await_owned_task(task)
        await dispatcher._stop_runtime_owner_heartbeat()
        await _close(backend, agent)
