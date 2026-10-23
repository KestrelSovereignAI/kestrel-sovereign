"""Actual native post-ACK journal failure; no replacement lease or fake clock."""
import asyncio
import pytest
from types import SimpleNamespace

from tests.unit.test_durable_signal_delivery import _channel_dispatcher, _channel_signal, _close
from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.signals import DurableConsumerRegistration
from kestrel_sovereign.signals.sources.channels import DURABLE_COGNITION_CONSUMER_ID
from kestrel_sovereign.execution_custody import ExecutionCustody, bind_execution_runtime
from tests.unit.test_signal_outcome_terminal_custody import _OriginalWork
from kestrel_sovereign.privacy import get_privacy_preset


@pytest.mark.asyncio
async def test_actual_stopped_route_is_never_requeued_by_renewal_join_observation(tmp_path):
    from tests.unit.test_durable_signal_delivery import _LifecycleAgent
    backend, previous_agent, dispatcher = await _channel_dispatcher(tmp_path / 'actual-stop-renewal.db', 'did:agent:actual-stop-renewal')
    agent = _LifecycleAgent(previous_agent.did)
    dispatcher._agent = agent
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    refused = []
    class RefuseRenewalObservation(asyncio.Task):
        def remove_done_callback(self, callback):
            if not refused and getattr(callback, '__name__', '') == 'mark_complete':
                refused.append(self)
                raise RuntimeError('actual renewer join callback removal refused')
            return super().remove_done_callback(callback)
    def factory(loop, coroutine, **kwargs):
        if getattr(getattr(coroutine, 'cr_code', None), 'co_name', '') == 'renew':
            return RefuseRenewalObservation(coroutine, loop=loop, **kwargs)
        return previous_factory(loop, coroutine, **kwargs) if previous_factory else asyncio.Task(coroutine, loop=loop, **kwargs)
    handle = None
    try:
        await dispatcher.register_durable_consumer(DurableConsumerRegistration(
            consumer_id=DURABLE_COGNITION_CONSUMER_ID, source='channel.message', agent_id=agent.did, max_attempts=0))
        loop.set_task_factory(factory)
        handle = await dispatcher.enqueue_durable_cognition(_channel_signal(agent.did, 'actual-stop-renewal'),
            source_event_id='actual-stop-renewal', consumer_id=DURABLE_COGNITION_CONSUMER_ID)
        await asyncio.wait_for(agent.process_started.wait(), 5)
        assert agent.cancel_current_request(handle.signal_id)
        await await_owned_task(handle.task)
        assert refused and all(owner.done() for owner in refused)
        delivery, = await dispatcher.list_durable_deliveries()
        assert delivery.status == 'failed' and delivery.lease_expires_at is None, 'renewal observation requeued a genuinely completed operator Stop'
        assert not dispatcher._retained_cognition_control_debt
    finally:
        loop.set_task_factory(previous_factory)
        agent.allow_process_completion.set()
        if handle is not None:
            await await_owned_task(handle.task)
        await await_owned_task(asyncio.create_task(dispatcher.shutdown_durable_delivery()))
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_actual_ack_cannot_become_new_claim_debt_after_native_audit_failure(tmp_path, monkeypatch):
    backend, agent, dispatcher = await _channel_dispatcher(tmp_path / 'actual-post-ack.db', 'did:agent:actual-post-ack')
    agent.privacy_config = get_privacy_preset('normal')
    original = dispatcher._write_outcome_log
    original_ack = dispatcher.ack_durable_delivery
    original_terminal = dispatcher._apply_cognition_control_terminal
    acknowledged = asyncio.Event()
    transition_entered, release_transition = asyncio.Event(), asyncio.Event()
    transitions, terminal_attempts = [], []
    async def observe_terminal(*args, **kwargs):
        terminal_attempts.append(args[0].delivery_id)
        return await original_terminal(*args, **kwargs)
    monkeypatch.setattr(dispatcher, '_apply_cognition_control_terminal', observe_terminal)
    async def hold_actual_transition():
        async with agent._durable_persistence_gate.exclusive():
            transition_entered.set()
            await release_transition.wait()
    async def observe_actual_ack(*args, **kwargs):
        result = await original_ack(*args, **kwargs)
        if result:
            acknowledged.set()
        return result
    monkeypatch.setattr(dispatcher, 'ack_durable_delivery', observe_actual_ack)
    async def privacy_before_actual_write(*args, **kwargs):
        if acknowledged.is_set():
            transitions.append(asyncio.create_task(hold_actual_transition()))
            await transition_entered.wait()
        try:
            return await original(*args, **kwargs)
        finally:
            release_transition.set()
    monkeypatch.setattr(dispatcher, '_write_outcome_log', privacy_before_actual_write)
    handle = None
    try:
        await dispatcher.register_durable_consumer(DurableConsumerRegistration(
            consumer_id=DURABLE_COGNITION_CONSUMER_ID, source='channel.message', agent_id=agent.did, max_attempts=0))
        scope = ExecutionCustody(_OriginalWork())
        from kestrel_sovereign.kestrel_agent import KestrelAgent
        class HostedPublisher(type(agent)):
            _runtime_owner_context = KestrelAgent._runtime_owner_context
            _track_runtime_task = KestrelAgent._track_runtime_task
            _remember_shutdown_control = KestrelAgent._remember_shutdown_control
        agent.__class__ = HostedPublisher
        agent._execution_custody = scope
        agent._background_tasks = set()
        agent._runtime_publication_ready = asyncio.Event()
        agent._runtime_publication_ready.set()
        with bind_execution_runtime(SimpleNamespace(_execution_custody=scope)):
            handle = await dispatcher.enqueue_durable_cognition(_channel_signal(agent.did, 'actual-post-ack'),
                source_event_id='actual-post-ack', consumer_id=DURABLE_COGNITION_CONSUMER_ID)
            outcome = await await_owned_task(handle.task)
        assert outcome.error is not None
        assert acknowledged.is_set(), repr(outcome.error)
        status, expires = await backend.fetch_one('SELECT status, lease_expires_at FROM durable_signal_deliveries WHERE agent_id=?', (agent.did,))
        assert status == 'acknowledged' and expires is None
        assert not terminal_attempts, 'journal refusal attempted a second native terminal CAS after actual ACK'
        assert not dispatcher._retained_cognition_control_debt, 'actual successful ACK was turned into a failed terminalization CAS and new retained claim debt'
    finally:
        release_transition.set()
        for transition in transitions:
            await await_owned_task(transition)
        if handle is not None:
            await await_owned_task(handle.task)
        # Preserve any failed original-token debt for the assertion. The
        # disposable database is closed, not made reclaimable or rewritten;
        # its actual heartbeat is joined without retrying an impossible CAS.
        await dispatcher._stop_runtime_owner_heartbeat()
        await _close(backend, agent)
