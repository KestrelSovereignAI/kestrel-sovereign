"""Real dispatcher attempts retain native SQL operation identity.

The cognition body is a declared file-effect fixture on a partial native
Agent. This tests dispatcher/journal wiring, not READY boot or an LLM.
"""

from dataclasses import replace
from datetime import timedelta
from types import MethodType

import pytest
from kestrel_sdk.resources import ResourceOutcomeUnknown
from kestrel_sdk.signals import Status

from kestrel_sovereign.agent.invocation import current_invocation_provenance
from kestrel_sovereign.auth import CallerContext
from kestrel_sovereign.signals import (
    DurableConsumerRegistration, OrderedLockManager, SignalDispatcher,
    SignalLogStore, SourceRegistry,
)
from kestrel_sovereign.signals.effect_provenance import signal_effect_provenance
from kestrel_sovereign.signals.sources import a2a_task_submitted, channels
from kestrel_sovereign.tools.effect_journal import execute_tool_effect
from tests.unit.test_durable_signal_delivery import _channel_signal
from tests.unit.test_tool_effect_execution import owner
from tests.unit.test_tool_effect_invocation import governed
from tests.unit.test_a2a_native_input_fencing import private_event


pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(30)]


async def dispatcher_for(owner):
    registry = SourceRegistry()
    registry.register(channels.build_channel_message_registration())
    store = SignalLogStore(owner.tool_effect_store.database.backend)
    await store.initialize()
    dispatcher = SignalDispatcher(agent=owner, registry=registry,
        lock_manager=OrderedLockManager(), store=store)
    await dispatcher.initialize_durable_delivery()
    await dispatcher.register_durable_consumer(DurableConsumerRegistration(
        consumer_id=channels.DURABLE_COGNITION_CONSUMER_ID,
        source=channels.SOURCE_NAME, agent_id=owner.did, max_attempts=0,
        correlation_selector=f'payload.{channels.DURABLE_COGNITION_MARKER}={channels.DURABLE_COGNITION_MARKER_VALUE}'))
    return dispatcher


async def test_fresh_dispatch_attempt_cannot_replay_original_source_effect(owner, tmp_path):
    effects = tmp_path / 'original-source-effects'
    entered, provenance = [], []

    @governed
    async def cognition(agent, user_input, *, invocation_id=None,
                        invocation_provenance=None, session_id=None, **kwargs):
        entered.append(invocation_id)
        provenance.append(current_invocation_provenance())

        async def effect():
            with effects.open('a') as stream:
                stream.write('one effect\n')
            return 'committed file effect'

        await execute_tool_effect(agent, registration='owned-file', arguments={}, execute=effect)
        if len(entered) == 1:
            raise OSError('Owned cognition failed after its acknowledged effect')
        return 'completed'

    owner.process_input = MethodType(cognition, owner)
    first_signal = _channel_signal(owner.did, 'original-input')
    first = await dispatcher_for(owner)
    second = None
    try:
        result = await first.dispatch_signal(first_signal,
            _durable_delivery_consumer_id=channels.DURABLE_COGNITION_CONSUMER_ID,
            source_event_id='original-provider-event')
        assert result.status is Status.FAILED
        assert effects.read_text() == 'one effect\n'
        assert (await first.list_durable_deliveries())[0].status == 'retry'
        await first.shutdown_durable_delivery()
        second = await dispatcher_for(owner)
        retry = replace(_channel_signal(owner.did, 'original-input'), id='fresh-retry-attempt',
            arrived_at=first_signal.arrived_at + timedelta(seconds=10))
        result = await second.dispatch_signal(retry,
            _durable_delivery_consumer_id=channels.DURABLE_COGNITION_CONSUMER_ID,
            source_event_id='original-provider-event')
        assert result.status is Status.FAILED
        assert 'tool_effect_reconciliation_required' in result.error
        assert entered == [first_signal.id], 'Retry reached cognition before journal reconciliation'
        assert effects.read_text() == 'one effect\n'
        retained, = await second.list_durable_deliveries()
        assert retained.status == 'failed', 'Effect reconciliation must not schedule another cognition retry'
        assert retained.acknowledged_at is None
        fresh = _channel_signal(owner.did, 'new-input')
        result = await second.dispatch_signal(fresh,
            _durable_delivery_consumer_id=channels.DURABLE_COGNITION_CONSUMER_ID,
            source_event_id='new-provider-event')
        assert result.status is Status.OK
        assert effects.read_text() == 'one effect\none effect\n'
        assert len(entered) == 2
        assert provenance[0].operation_key != provenance[1].operation_key
    finally:
        await first.shutdown_durable_delivery()
        if second is not None:
            await second.shutdown_durable_delivery()


async def test_a2a_working_wake_reuses_input_but_continuation_is_a_new_operation(owner):
    signal = replace(_channel_signal(owner.did, 'unused'),
        source=a2a_task_submitted.SOURCE_NAME,
        dedupe_key=a2a_task_submitted.submitted_input_identity('native-task', 0),
        payload={'task_id': 'native-task', 'input_revision': 0, 'request_content': 'original task'})
    event = replace(private_event(signal), payload=signal.payload)
    submitted = signal_effect_provenance(event, agent_id=owner.did)
    working = signal_effect_provenance(replace(event, event_id='working-event',
        source_event_id=a2a_task_submitted.submitted_input_identity('native-task', 0, working_revision=9)),
        agent_id=owner.did)
    assert submitted.operation_key == working.operation_key
    assert submitted.input_digest == working.input_digest
    continued = signal_effect_provenance(replace(event, event_id='continued-event',
        source_event_id=a2a_task_submitted.submitted_input_identity('native-task', 1),
        payload={**signal.payload, 'input_revision': 1, 'request_content': 'continued task'}), agent_id=owner.did)
    assert continued.operation_key != submitted.operation_key
    assert continued.input_digest != submitted.input_digest
    # Volatile A2A recovery supplies a separately verified source envelope.
    # The payload-elided event still owns identity; no new key or durable
    # journal is needed merely to run a read-only cognition turn.
    elided = private_event(signal)
    recovered = signal_effect_provenance(elided, agent_id=owner.did, canonical_signal=signal)
    assert recovered.operation_key == submitted.operation_key
    assert recovered.input_digest == submitted.input_digest
    with pytest.raises(ValueError, match='source recipient'):
        signal_effect_provenance(elided, agent_id=owner.did,
            canonical_signal=replace(signal, source='unrelated.source'))
    with pytest.raises(ValueError, match='another'):
        signal_effect_provenance(event, agent_id='did:other')
    with pytest.raises(ValueError, match='disagrees'):
        signal_effect_provenance(replace(event, payload={**signal.payload, 'input_revision': 1}), agent_id=owner.did)


async def test_lost_effect_result_terminalizes_without_claiming_cursor_ack(owner, tmp_path):
    witness = tmp_path / 'uncertain-source-effect'
    entered = []

    @governed
    async def cognition(agent, user_input, *, invocation_id=None,
                        invocation_provenance=None, session_id=None, **kwargs):
        entered.append(invocation_id)

        async def effect():
            witness.write_text('one uncertain effect')
            raise ResourceOutcomeUnknown('private remote outcome')

        return await execute_tool_effect(agent, registration='owned-file', arguments={}, execute=effect)

    owner.process_input = MethodType(cognition, owner)
    dispatcher = await dispatcher_for(owner)
    try:
        result = await dispatcher.dispatch_signal(_channel_signal(owner.did, 'uncertain-input'),
            _durable_delivery_consumer_id=channels.DURABLE_COGNITION_CONSUMER_ID,
            source_event_id='uncertain-provider-event')
        assert result.status is Status.FAILED
        assert result.error.startswith('tool_effect_outcome_unconfirmed:')
        assert 'private remote outcome' not in result.error
        delivery, = await dispatcher.list_durable_deliveries()
        assert delivery.status == 'failed'
        assert delivery.acknowledged_at is None
        assert await owner.tool_effect_store.database.fetchone(
            'SELECT state FROM tool_effect_claims') == ('UNCONFIRMED',)
        await dispatcher.dispatch_signal(_channel_signal(owner.did, 'uncertain-input'),
            _durable_delivery_consumer_id=channels.DURABLE_COGNITION_CONSUMER_ID,
            source_event_id='uncertain-provider-event')
        assert len(entered) == 1
        assert witness.read_text() == 'one uncertain effect'
    finally:
        await dispatcher.shutdown_durable_delivery()


async def test_signal_retry_namespace_is_independent_of_an_enclosing_http_caller(owner):
    from kestrel_sovereign.tool_effect_store import ToolEffectReconciliationRequired

    entered = []

    @governed
    async def cognition(agent, user_input, *, invocation_id=None,
                        invocation_provenance=None, caller=None):
        entered.append(user_input)

        async def effect():
            return 'one admitted effect'

        return await execute_tool_effect(agent, registration='owned-effect', arguments={}, execute=effect)

    provenance = signal_effect_provenance(_channel_signal(owner.did, 'stable-event'), agent_id=owner.did)
    assert await cognition(owner, 'owned input', invocation_id='first-lifecycle',
        invocation_provenance=provenance, caller=CallerContext.authenticated('alice')) == 'one admitted effect'
    with pytest.raises(ToolEffectReconciliationRequired):
        await cognition(owner, 'same original input, new canary', invocation_id='retry-lifecycle',
            invocation_provenance=provenance, caller=CallerContext.authenticated('bob'))
    assert entered == ['owned input']
