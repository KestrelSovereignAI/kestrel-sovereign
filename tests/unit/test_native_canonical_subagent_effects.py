"""Canonical Core and adopted SDK subagent effects retain their original approval."""

import pytest
import pytest_asyncio
from kestrel_sdk.features.base import Feature as SDKFeature
from kestrel_sdk.hooks.base import Hook, HookEvent, HookOutput
from kestrel_sdk.llm.response import LLMResponse, ToolCall
from kestrel_sdk.resources import ResourceConflict, ResourceOutcomeUnknown
from kestrel_sovereign.agent.invocation import current_invocation_effect_checkpoint
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.subagent_dispatch import ensure_subagent_dispatch
from kestrel_sovereign.tool_effect_store import ToolEffectReconciliationRequired
from kestrel_sovereign.execution_custody import execution_evidence_carriers
from tests.unit.test_tool_effect_bound_operation import bound_operation as bound_operation
from tests.unit.test_tool_effect_execution import owner
from tests.unit.test_tool_effect_invocation import governed
from tests.utils.codex_inline import OwnedCallbackReader

pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(30)]


@pytest_asyncio.fixture(params=[Feature, SDKFeature], ids=['core', 'sdk'])
async def canonical_subagent(bound_operation, request):
    host, binding, witness = bound_operation
    failure = TimeoutError('canonical native leaf lost its original reply')
    checkpoints = []

    class NativeFeature(request.param):
        tool_description = 'One original file effect'

        async def initialize(self):
            pass

        @tool(name='native_leaf', description='Write one original marker')
        async def native_leaf(self, marker: str):
            checkpoints.append(current_invocation_effect_checkpoint())
            with witness.open('a') as stream:
                stream.write(marker + '\n')
            if self.lose_reply:
                raise failure
            if self.after_effect is not None:
                self.after_effect()
            if self.defer_result:
                from kestrel_sdk.tools.deferred import DeferredToolResult
                from kestrel_sdk.tools.result import ToolResult
                return DeferredToolResult(ToolResult.ok('original').to_dict(), publish=lambda: self.deferred_parts)
            if self.native_result is not None:
                return self.native_result
            return {'success': True, 'marker': marker}

    feature = ensure_subagent_dispatch(NativeFeature)(host)
    feature.lose_reply = False
    feature.after_effect = None
    feature.defer_result = False
    feature.deferred_parts = []
    feature.native_result = None
    host.features[feature.name] = feature

    reader = OwnedCallbackReader()
    await reader.start()

    class Model:
        calls = 0
        inline = False
        text_only = False
        continuations = 0
        continuation_error = None
        before_tool = None

        async def generate(self, **kwargs):
            self.calls += 1
            if self.before_tool is not None:
                self.before_tool()
            if self.text_only:
                return LLMResponse(content='done', tool_calls=None)
            if self.inline:
                await reader.dispatch(kwargs['tool_executor'], 'native_leaf', {'marker':'original'})
                if self.continuation_error is not None:
                    raise self.continuation_error
                return LLMResponse(content='done', tool_calls=None)
            return LLMResponse(content=None, tool_calls=[
                ToolCall(id='native-leaf-call', name='native_leaf', arguments={'marker': 'original'})])

        async def generate_with_messages(self, **kwargs):
            self.continuations += 1
            if self.continuation_error is not None:
                raise self.continuation_error
            return LLMResponse(content='done', tool_calls=None)

    model = host.llm_service = Model()

    @governed
    async def chat(agent, user_input, *, invocation_id=None):
        return await agent._dispatch_tool_call(
            ToolCall(id='canonical-dispatch', name=feature.tool_name, arguments={'task': 'one effect'}),
            {feature.tool_name: feature}, {feature.tool_name}, [], 0, user_input,
            session_id='canonical-session')

    async def invoke(lane):
        if lane == 'named':
            return await host.execute_named_tool(feature.tool_name, {'task': 'one effect'},
                session_id='canonical-session', operation_id='canonical-original-intent')
        return await chat(host, 'one original canonical input', invocation_id='canonical-original-intent')

    try:
        yield host, feature, witness, model, failure, checkpoints, invoke
    finally:
        await reader.stop()


@pytest.mark.parametrize('lane', ['chat', 'named'])
@pytest.mark.parametrize('change', ['feature', 'method', 'disabled', 'foreign-owner', 'registry'])
async def test_canonical_subagent_revalidates_original_permission_registration(
    canonical_subagent, native_tool_effect_agent, lane, change,
):
    host, feature, witness, model, _, _, invoke = canonical_subagent
    replacement = type(feature)(host)
    foreign = await native_tool_effect_agent(did=host.did) if change == 'foreign-owner' else None

    class ReplaceAfterPermission(Hook):
        def __init__(self):
            super().__init__('canonical-original-approval', [HookEvent.PRE_SUBAGENT_CALL])
            self.calls = 0

        async def execute(self, value):
            self.calls += 1
            if change == 'feature':
                host.features[feature.name] = replacement
            elif change == 'method':
                feature.execute_as_subagent = replacement.execute_as_subagent
            elif change == 'disabled':
                feature.enabled = False
            elif change == 'foreign-owner':
                feature.agent = foreign
            else:
                host.register_dynamic_tools('canonical-generation-change', list(host._direct_tools.values())[:1])
            return HookOutput.allow()

    policy = ReplaceAfterPermission()
    host.hooks_manager.register(policy)
    if lane == 'named':
        with pytest.raises(ResourceConflict):
            await invoke(lane)
    else:
        result = await invoke(lane)
        assert result['success'] is False
    assert policy.calls == 1
    assert model.calls == 0, 'The stale canonical dispatcher entered its model loop'
    assert not witness.exists(), 'A stale canonical feature entered its physical effect'


@pytest.mark.parametrize('lane', ['chat', 'named'])
async def test_canonical_leaf_detaches_the_post_approval_binding(canonical_subagent, monkeypatch, lane):
    host, _, witness, _, _, _, invoke = canonical_subagent
    approved = {'marker': 'approved'}

    class Rewrite(Hook):
        def __init__(self):
            super().__init__('canonical-original-binding', [HookEvent.PRE_TOOL_USE])

        async def execute(self, value):
            return HookOutput.modify(approved)

    host.hooks_manager.register(Rewrite())
    claim = host.tool_effect_store.claim

    async def mutate_after_claim(*args, **kwargs):
        receipt = await claim(*args, **kwargs)
        approved['marker'] = 'unapproved'
        return receipt

    monkeypatch.setattr(host.tool_effect_store, 'claim', mutate_after_claim)
    result = await invoke(lane)
    assert result['success'] is True
    assert approved['marker'] == 'unapproved'
    assert witness.read_text().splitlines() == ['approved'], 'The effect used a hook-owned mutable binding'
    rows = await host.tool_effect_store.database.fetchall(
        "SELECT args_redacted FROM a2a_tool_dispatches WHERE tool_name='native_leaf'")
    assert len(rows) == 1 and 'approved' in rows[0][0] and 'unapproved' not in rows[0][0]


@pytest.mark.parametrize('lane', ['chat', 'named'])
async def test_uncertain_canonical_leaf_retains_audit_checkpoint_and_replay_fence(canonical_subagent, lane):
    host, feature, witness, _, original, checkpoints, invoke = canonical_subagent
    feature.lose_reply = True
    with pytest.raises(ResourceOutcomeUnknown) as caught:
        await invoke(lane)
    assert any(error is original for error in execution_evidence_carriers(caught.value))
    assert witness.read_text().splitlines() == ['original']
    assert len(checkpoints) == 1 and checkpoints[0].completed and checkpoints[0].checkpointed
    rows = await host.tool_effect_store.database.fetchall(
        "SELECT result_status, error_class, parent_dispatch_id FROM a2a_tool_dispatches WHERE tool_name='native_leaf'")
    assert len(rows) == 1 and rows[0][:2] == ('error', 'ResourceOutcomeUnknown')
    assert rows[0][2], 'The native leaf lost the dispatch that delegated its effect'
    parent = await host.tool_effect_store.database.fetchone(
        'SELECT tool_name, result_status FROM a2a_tool_dispatches WHERE dispatch_id=?', (rows[0][2],))
    assert parent == (feature.tool_name, 'error'), 'The native leaf names an unrecorded parent dispatch'
    with pytest.raises(ToolEffectReconciliationRequired):
        await invoke(lane)
    assert witness.read_text().splitlines() == ['original']


@pytest.mark.parametrize('inline', [False, True], ids=['loop', 'owned-reader'])
@pytest.mark.parametrize('change', ['feature', 'method', 'disabled', 'registry'])
async def test_canonical_loop_keeps_the_original_registration_at_its_leaf(canonical_subagent, inline, change):
    host, feature, witness, model, _, _, invoke = canonical_subagent
    replacement = type(feature)(host)
    def change_before_leaf():
        if change == 'feature':
            host.features[feature.name] = replacement
        elif change == 'method':
            feature.execute_as_subagent = replacement.execute_as_subagent
        elif change == 'disabled':
            feature.enabled = False
        else:
            host.register_dynamic_tools('canonical-late-generation', list(host._direct_tools.values())[:1])
    model.before_tool = change_before_leaf
    model.inline = inline
    await invoke('chat')
    assert model.calls == 1
    assert not witness.exists(), 'A canonical leaf lost its original dispatcher registration'
    assert await host.tool_effect_store.database.fetchall('SELECT state FROM tool_effect_claims') == []


@pytest.mark.parametrize('inline', [False, True], ids=['loop', 'owned-reader'])
async def test_canonical_loop_does_not_move_its_leaf_into_a_foreign_native_host(
    canonical_subagent, native_tool_effect_agent, inline,
):
    host, feature, witness, model, _, _, invoke = canonical_subagent
    foreign = await native_tool_effect_agent(did=host.did)
    def change_owner():
        feature.agent = foreign
    model.before_tool = change_owner
    model.inline = inline
    await invoke('chat')
    assert not witness.exists()
    assert 'native_leaf' not in host._direct_tools, 'A foreign feature was promoted after its refusal'
    assert await foreign.tool_effect_store.database.fetchone('SELECT COUNT(*) FROM tool_effect_claims') == (0,)
    assert await foreign.tool_effect_store.database.fetchone('SELECT COUNT(*) FROM a2a_tool_dispatches') == (0,), 'Original leaf dispatch was written into another host'


@pytest.mark.parametrize('lane', ['chat', 'named'])
@pytest.mark.parametrize('target', ['leaf', 'parent'])
async def test_native_canonical_audit_refusal_keeps_the_file_effect_and_original_replay_fence(
    canonical_subagent, monkeypatch, lane, target,
):
    host, feature, witness, _, _, _, invoke = canonical_subagent
    selected = 'native_leaf' if target == 'leaf' else feature.tool_name
    await host.tool_effect_store.database.execute(
        "CREATE TRIGGER refuse_original_dispatch BEFORE INSERT ON a2a_tool_dispatches "
        "WHEN NEW.tool_name = '" + selected + "' "
        "BEGIN SELECT RAISE(ABORT, 'original native audit refused'); END")
    write = host.observability_store.log_tool_dispatch
    attempts = []
    async def record(entry, **kwargs):
        attempts.append(entry.tool_name)
        return await write(entry, **kwargs)
    monkeypatch.setattr(host.observability_store, 'log_tool_dispatch', record)
    with pytest.raises(ResourceOutcomeUnknown):
        await invoke(lane)
    assert witness.read_text().splitlines() == ['original']
    assert attempts == ['native_leaf', feature.tool_name], 'The required original dispatch audit was attempted twice'
    assert await host.tool_effect_store.database.fetchall('SELECT state FROM tool_effect_claims') == [('RETURNED_OK',)]
    with pytest.raises(ToolEffectReconciliationRequired):
        await invoke(lane)
    assert witness.read_text().splitlines() == ['original']
    assert attempts == ['native_leaf', feature.tool_name]


@pytest.mark.parametrize('inline', [False, True], ids=['loop', 'owned-reader'])
@pytest.mark.parametrize('decision', ['allow', 'deny', 'ask'])
async def test_original_leaf_host_is_retained_through_the_permission_await(
    canonical_subagent, native_tool_effect_agent, inline, decision,
):
    host, feature, witness, model, _, _, invoke = canonical_subagent
    foreign = await native_tool_effect_agent(did=host.did)
    class MoveAtPermission(Hook):
        def __init__(self):
            super().__init__('original-leaf-host', [HookEvent.PRE_TOOL_USE])
            self.calls = 0
        async def execute(self, value):
            if value.tool_name == 'native_leaf':
                self.calls += 1
                feature.agent = foreign
                if decision == 'deny':
                    return HookOutput.deny('Original leaf permission refused')
                if decision == 'ask':
                    return HookOutput.ask('original-approval', 'Original leaf needs approval')
            return HookOutput.allow()
    policy = MoveAtPermission()
    host.hooks_manager.register(policy)
    model.inline = inline
    result = await invoke('chat')
    assert policy.calls == 1
    assert not witness.exists()
    assert await foreign.tool_effect_store.database.fetchone('SELECT COUNT(*) FROM a2a_tool_dispatches') == (0,), 'Permission await redirected the original leaf audit into another host'
    assert result['success'] is False and model.continuations == 0
    assert 'native_leaf' not in host._direct_tools


@pytest.mark.parametrize('inline', [False, True], ids=['loop', 'owned-reader'])
@pytest.mark.parametrize('timing', ['permission', 'claim-ack'])
async def test_registration_refusal_stops_the_original_loop_without_continuation(
    canonical_subagent, monkeypatch, inline, timing,
):
    host, feature, witness, model, _, _, invoke = canonical_subagent
    if timing == 'permission':
        class Withdraw(Hook):
            def __init__(self):
                super().__init__('original-leaf-registration', [HookEvent.PRE_TOOL_USE])
            async def execute(self, value):
                if value.tool_name == 'native_leaf':
                    feature.enabled = False
                return HookOutput.allow()
        host.hooks_manager.register(Withdraw())
    else:
        claim = host.tool_effect_store.claim
        async def withdraw_after_original_claim(*args, **kwargs):
            receipt = await claim(*args, **kwargs)
            feature.enabled = False
            return receipt
        monkeypatch.setattr(host.tool_effect_store, 'claim', withdraw_after_original_claim)
    model.inline = inline
    result = await invoke('chat')
    assert result['success'] is False, 'A refused original registration became a successful subagent response'
    assert not witness.exists() and model.continuations == 0
    assert 'native_leaf' not in host._direct_tools
    states = await host.tool_effect_store.database.fetchall('SELECT state FROM tool_effect_claims')
    assert states == ([] if timing == 'permission' else [('NOT_EXECUTED',)])


@pytest.mark.parametrize('lane', ['chat', 'named'])
@pytest.mark.parametrize('change', ['feature', 'method', 'disabled', 'foreign-owner', 'registry'])
async def test_text_only_completion_cannot_promote_a_withdrawn_original_feature(
    canonical_subagent, native_tool_effect_agent, lane, change,
):
    host, feature, witness, model, _, _, invoke = canonical_subagent
    replacement = type(feature)(host)
    foreign = await native_tool_effect_agent(did=host.did) if change == 'foreign-owner' else None
    def withdraw_during_response():
        if change == 'feature':
            host.features[feature.name] = replacement
        elif change == 'method':
            feature.execute_as_subagent = replacement.execute_as_subagent
        elif change == 'disabled':
            feature.enabled = False
        elif change == 'foreign-owner':
            feature.agent = foreign
        else:
            host.register_dynamic_tools('original-text-completion', list(host._direct_tools.values())[:1])
    model.before_tool = withdraw_during_response
    model.text_only = True
    if lane == 'named':
        with pytest.raises(ResourceConflict):
            await invoke(lane)
    else:
        result = await invoke(lane)
        assert result['success'] is False, 'Withdrawn original feature was accepted after its text-only model response'
    assert model.calls == 1 and model.continuations == 0
    assert not witness.exists()
    assert 'native_leaf' not in host._direct_tools, 'Text-only completion remounted a withdrawn feature'
    assert await host.tool_effect_store.database.fetchall('SELECT state FROM tool_effect_claims') == []


@pytest.mark.parametrize('inline', [False, True], ids=['loop', 'owned-reader'])
async def test_a_continuation_failure_after_a_native_file_effect_cannot_be_a_normal_reply(canonical_subagent, inline):
    host, feature, witness, model, _, _, invoke = canonical_subagent
    original = RuntimeError('private-original-provider-continuation')
    model.inline = inline
    model.continuation_error = original
    with pytest.raises(ResourceOutcomeUnknown) as caught:
        await invoke('chat')
    assert any(carrier is original for carrier in execution_evidence_carriers(caught.value))
    assert 'private-original-provider-continuation' not in str(caught.value)
    assert witness.read_text().splitlines() == ['original']
    assert await host.tool_effect_store.database.fetchall('SELECT state FROM tool_effect_claims') == [('RETURNED_OK',)]
    assert await host.tool_effect_store.database.fetchone(
        'SELECT result_status FROM a2a_tool_dispatches WHERE tool_name=?', (feature.tool_name,)) == ('error',)
    with pytest.raises(ToolEffectReconciliationRequired):
        await invoke('chat')
    assert witness.read_text().splitlines() == ['original']


@pytest.mark.parametrize('inline', [False, True], ids=['loop', 'owned-reader'])
async def test_deferred_leaf_retains_original_host_through_completion(
    canonical_subagent, native_tool_effect_agent, inline,
):
    host, feature, witness, model, _, _, invoke = canonical_subagent
    foreign = await native_tool_effect_agent(did=host.did)
    model.inline = inline
    feature.defer_result = True
    feature.after_effect = lambda: setattr(feature, 'agent', foreign)
    # The original outer registration refuses the now withdrawn dispatcher.
    with pytest.raises(ResourceOutcomeUnknown) as caught:
        await invoke('named')
    assert any(isinstance(error, ResourceConflict)
        for error in execution_evidence_carriers(caught.value))
    assert witness.read_text().splitlines() == ['original']
    rows = await foreign.tool_effect_store.database.fetchone('SELECT COUNT(*) FROM a2a_tool_dispatches')
    assert rows == (0,), 'Deferred completion wrote the original leaf audit into a foreign host'


@pytest.mark.parametrize('inline', [False, True], ids=['loop', 'owned-reader'])
async def test_standalone_canonical_component_refuses_original_host_change(
    canonical_subagent, native_tool_effect_agent, inline,
):
    from tests.utils.native_tool_effects import drive_native

    host, feature, witness, model, _, _, _ = canonical_subagent
    foreign = await native_tool_effect_agent(did=host.did)
    model.inline = inline
    model.before_tool = lambda: setattr(feature, 'agent', foreign)
    with pytest.raises(ResourceConflict):
        await drive_native(host, lambda: feature.execute_as_subagent(task='one effect'))
    assert not witness.exists() and model.continuations == 0
    assert await foreign.tool_effect_store.database.fetchone(
        'SELECT COUNT(*) FROM a2a_tool_dispatches') == (0,)


@pytest.mark.parametrize('completion', ['serialize', 'deferred-delivery', 'audit'])
@pytest.mark.parametrize('route', ['codex', 'catcher'])
async def test_canonical_leaf_completion_stays_inside_original_execution_boundary(
    canonical_subagent, monkeypatch, completion, route,
):
    from kestrel_sovereign.agent.parts import part_collector
    from kestrel_sovereign.execution_custody import is_tool_execution_terminal_error
    from kestrel_sovereign.llm.codex_adapter import CodexAdapter
    from kestrel_sovereign.storage.db.interface import DatabaseError
    from kestrel_sovereign.tools.effect_journal import current_tool_effect_turn
    from tests.utils.native_tool_effects import drive_native

    host, feature, witness, _, _, _, _ = canonical_subagent
    original = ValueError('private-native-leaf-completion')
    sql_errors, attempts = [], []
    if completion == 'serialize':
        class NativeValue:
            def to_dict(self):
                raise original
        feature.native_result = {'success': True, 'payload': NativeValue()}
    elif completion == 'deferred-delivery':
        feature.defer_result = True
        feature.deferred_parts = [{'type': 'private_view', 'data': {'reference': 'leaf-owned'}}]
        emit = host._reemit_envelope_parts
        def fail_after_ack(envelope):
            if any(part.get('type') == 'private_view' for part in envelope.get('parts', [])):
                raise original
            return emit(envelope)
        monkeypatch.setattr(host, '_reemit_envelope_parts', fail_after_ack)
    else:
        await host.tool_effect_store.database.execute(
            "CREATE TRIGGER refuse_leaf_completion BEFORE INSERT ON a2a_tool_dispatches "
            "WHEN NEW.tool_name = 'native_leaf' "
            "BEGIN SELECT RAISE(ABORT, 'private-native-leaf-audit'); END")
        write = host.observability_store.log_tool_dispatch
        async def fail_native_audit(entry, **kwargs):
            attempts.append(entry.tool_name)
            try:
                return await write(entry, **kwargs)
            except DatabaseError as error:
                sql_errors.append(error)
                raise
        monkeypatch.setattr(host.observability_store, 'log_tool_dispatch', fail_native_audit)
    seen, executed, terminal = {}, [], []
    async def operation():
        runtime_tools = feature.get_tools()
        executor = feature._make_feature_inline_tool_executor(
            runtime_tools=runtime_tools, session_id='canonical-completion')
        handler = CodexAdapter()._make_tool_call_handler(
            executor, 'native-thread', frozenset({'native_leaf'}),
            executed_log=executed, terminal_errors=terminal)
        async def invoke():
            if route == 'codex':
                return await handler({'threadId': 'native-thread', 'name': 'native_leaf',
                    'arguments': {'marker': 'original'}})
            return await feature._execute_subagent_tool(
                tool_name='native_leaf', args={'marker': 'original'},
                tools_by_name={tool.name: tool for tool in runtime_tools},
                session_id='canonical-completion')
        with part_collector(host=host, session_id='canonical-completion'):
            try:
                seen['first_result'] = await invoke()
            except Exception as error:
                seen['first_error'] = error
            seen['unconfirmed'] = tuple(current_tool_effect_turn(host).unconfirmed)
            try:
                seen['second_result'] = await invoke()
            except Exception as error:
                seen['second_error'] = error
    try:
        await drive_native(host, operation)
    except ResourceOutcomeUnknown as error:
        seen['root_error'] = error
    assert witness.read_text().splitlines() == ['original'], (
        'Canonical leaf completion escaped its shared boundary and repeated the physical effect')
    source = sql_errors[0] if completion == 'audit' else original
    assert any(source is item for error in seen['unconfirmed']
        for item in execution_evidence_carriers(error)), 'Canonical leaf completion lost its original live uncertainty'
    assert 'first_error' in seen and is_tool_execution_terminal_error(seen['first_error'])
    assert 'second_error' in seen and 'second_result' not in seen
    assert 'root_error' in seen, 'Canonical leaf input completed successfully after uncertain delivery'
    assert source in execution_evidence_carriers(seen['root_error'])
    if route == 'codex':
        assert terminal == [seen['first_error'], seen['first_error']] and executed == []
        assert seen['second_error'] is seen['first_error']
    assert await host.tool_effect_store.database.fetchall(
        'SELECT state FROM tool_effect_claims') == [('RETURNED_OK',)]
    rows = await host.tool_effect_store.database.fetchall(
        "SELECT result_status,error_class FROM a2a_tool_dispatches WHERE tool_name='native_leaf'")
    assert rows == ([] if completion == 'audit' else
        [('success', None)] if completion == 'deferred-delivery' else [('error', 'ResourceOutcomeUnknown')])
    if completion == 'audit':
        assert attempts == ['native_leaf'] and len(sql_errors) == 1


@pytest.mark.parametrize('completion', ['parent-delivery', 'model-continuation'])
@pytest.mark.parametrize('failure', ['ordinary', 'conflict', 'unknown'])
@pytest.mark.parametrize('route', ['chat', 'named-inline'])
async def test_canonical_parent_completion_retains_returned_leaf_evidence(
    canonical_subagent, monkeypatch, completion, failure, route,
):
    from uuid import uuid4
    from kestrel_sovereign.agent.parts import part_collector
    from kestrel_sovereign.execution_custody import is_tool_execution_terminal_error
    from kestrel_sovereign.tools.effect_journal import current_tool_effect_turn
    from tests.utils.native_tool_effects import drive_native

    host, feature, witness, model, _, _, _ = canonical_subagent
    original = {'ordinary': ValueError, 'conflict': ResourceConflict,
        'unknown': ResourceOutcomeUnknown}[failure]('private-parent-completion')
    if completion == 'model-continuation':
        model.continuation_error = original
    else:
        emit = host._reemit_envelope_parts
        def fail_parent_delivery(envelope):
            if isinstance(envelope, dict) and isinstance(envelope.get('result'), str):
                raise original
            return emit(envelope)
        monkeypatch.setattr(host, '_reemit_envelope_parts', fail_parent_delivery)
    seen = {}
    async def operation():
        async def invoke():
            if route == 'named-inline':
                return await host.execute_named_tool(feature.tool_name, {'task': 'one effect'},
                    session_id='canonical-parent-completion', source='owned-inline', _capture={})
            return await host._dispatch_tool_call(
                ToolCall(id=uuid4().hex, name=feature.tool_name, arguments={'task': 'one effect'}),
                {feature.tool_name: feature}, {feature.tool_name}, [], 0, 'one original input',
                session_id='canonical-parent-completion')
        with part_collector(host=host, session_id='canonical-parent-completion'):
            try:
                seen['first_result'] = await invoke()
            except Exception as error:
                seen['first_error'] = error
            seen['unconfirmed'] = tuple(current_tool_effect_turn(host).unconfirmed)
            try:
                seen['second_result'] = await invoke()
            except Exception as error:
                seen['second_error'] = error
    try:
        await drive_native(host, operation)
    except ResourceOutcomeUnknown as error:
        seen['root_error'] = error
    assert witness.read_text().splitlines() == ['original'], (
        'Canonical parent completion lost its returned leaf and repeated the physical effect')
    assert any(original is item for error in seen['unconfirmed']
        for item in execution_evidence_carriers(error)), 'Canonical parent completion lost its original live uncertainty'
    assert 'first_error' in seen and is_tool_execution_terminal_error(seen['first_error'])
    assert 'second_error' in seen and 'second_result' not in seen
    assert 'root_error' in seen and original in execution_evidence_carriers(seen['root_error'])
    assert model.calls == 1, 'A model continued after its original completion became uncertain'
    assert await host.tool_effect_store.database.fetchall(
        'SELECT state FROM tool_effect_claims') == [('RETURNED_OK',)]
    rows = await host.tool_effect_store.database.fetchall(
        "SELECT result_status,error_class FROM a2a_tool_dispatches WHERE tool_name='native_leaf'")
    assert rows == [('success', None)]


def _unfinished_decision():
    from types import MappingProxyType
    from kestrel_sdk.llm.decisions import DecisionResult, NoulAnswer
    from kestrel_sovereign.turn_completion import CONTINUATION_QUESTION

    return DecisionResult(
        answers=MappingProxyType({CONTINUATION_QUESTION: NoulAnswer(p_true=1.0)}),
        vendor='openrouter', route='openrouter:api', model='owned-decision',
        thresholds=MappingProxyType({CONTINUATION_QUESTION: .5}),
        calibrated=True, input_tokens=1, duration_ms=0,
    )


@pytest.mark.parametrize('content', ['done', 'Let me check the file.'])
@pytest.mark.parametrize('continuation_check', ['regex', 'decision'])
@pytest.mark.parametrize('route', ['chat', 'named-inline'])
async def test_canonical_parent_cannot_deliver_a_swallowed_terminal_leaf_result(
    canonical_subagent, monkeypatch, route, continuation_check, content,
):
    from uuid import uuid4
    from kestrel_sovereign.agent.parts import part_collector
    from kestrel_sovereign.llm.codex_adapter import CodexAdapter
    from kestrel_sovereign.turn_completion import (
        CONTINUATION_CHECK_ENV, CONTINUATION_CHECK_MODEL_ENV,
    )
    from tests.utils.native_tool_effects import drive_native

    host, feature, witness, model, _, _, _ = canonical_subagent
    monkeypatch.setenv(CONTINUATION_CHECK_ENV, continuation_check)
    monkeypatch.setenv(CONTINUATION_CHECK_MODEL_ENV, 'openrouter:api/owned-decision')
    decisions = []
    async def decide(request, **kwargs):
        decisions.append(request)
        return _unfinished_decision()
    model.decide = decide
    original = ValueError('private-swallowed-leaf-completion')
    class NativeValue:
        def to_dict(self):
            raise original
    feature.native_result = {'success': True, 'data': NativeValue()}
    seen = {}
    async def provider(**kwargs):
        model.calls += 1
        handler = CodexAdapter()._make_tool_call_handler(
            kwargs['tool_executor'], 'swallow-thread', frozenset({'native_leaf'}),
            executed_log=[], terminal_errors=[])
        try:
            await handler({'threadId': 'swallow-thread', 'name': 'native_leaf',
                'arguments': {'marker': 'original'}})
        except Exception as error:
            seen['leaf_error'] = error
        return LLMResponse(content=content, tool_calls=None)
    monkeypatch.setattr(model, 'generate', provider)
    async def operation():
        with part_collector(host=host, session_id='swallowed-terminal'):
            try:
                if route == 'named-inline':
                    seen['parent_result'] = await host.execute_named_tool(
                        feature.tool_name, {'task': 'one effect'}, session_id='swallowed-terminal',
                        source='owned-inline', _capture={})
                else:
                    seen['parent_result'] = await host._dispatch_tool_call(
                        ToolCall(id=uuid4().hex, name=feature.tool_name, arguments={'task': 'one effect'}),
                        {feature.tool_name: feature}, {feature.tool_name}, [], 0, 'one input',
                        session_id='swallowed-terminal')
            except Exception as error:
                seen['parent_error'] = error
    try:
        await drive_native(host, operation)
    except ResourceOutcomeUnknown as error:
        seen['root_error'] = error
    assert 'leaf_error' in seen and original in execution_evidence_carriers(seen['leaf_error'])
    assert 'parent_result' not in seen, 'Canonical parent delivered business output after its terminal leaf failed'
    assert isinstance(seen.get('parent_error'), ResourceOutcomeUnknown)
    assert original in execution_evidence_carriers(seen['parent_error'])
    assert 'root_error' in seen and witness.read_text().splitlines() == ['original']
    assert decisions == [], 'A decision model ran after the original leaf completion became uncertain'
    assert model.continuations == 0, 'A repair model ran after the original leaf completion became uncertain'
    assert model.calls == 1


async def test_canonical_repair_rechecks_original_completion_after_decision(
    canonical_subagent, monkeypatch,
):
    from kestrel_sovereign.llm.codex_adapter import CodexAdapter
    from kestrel_sovereign.turn_completion import CONTINUATION_CHECK_ENV
    from tests.utils.native_tool_effects import drive_native

    host, feature, witness, model, _, _, _ = canonical_subagent
    monkeypatch.setenv(CONTINUATION_CHECK_ENV, 'decision')
    original = ValueError('private-leaf-completion-during-decision')
    class NativeValue:
        def to_dict(self):
            raise original
    feature.native_result = {'success': True, 'data': NativeValue()}
    decisions, seen = [], {}
    async def operation():
        executor = feature._make_feature_inline_tool_executor()
        async def decide(request, **kwargs):
            decisions.append(request)
            handler = CodexAdapter()._make_tool_call_handler(
                executor, 'decision-thread', frozenset({'native_leaf'}),
                executed_log=[], terminal_errors=[])
            try:
                await handler({'threadId': 'decision-thread', 'name': 'native_leaf',
                    'arguments': {'marker': 'original'}})
            except Exception as error:
                seen['leaf_error'] = error
            return _unfinished_decision()
        model.decide = decide
        try:
            seen['result'] = await feature._repair_subagent_premature_yield(
                LLMResponse(content='Let me check the file.', tool_calls=None),
                [], [item.schema.to_openai_format() for item in feature.get_tools()],
                tool_executor=executor)
        except Exception as error:
            seen['error'] = error
    try:
        await drive_native(host, operation)
    except ResourceOutcomeUnknown as error:
        seen['root_error'] = error
    assert len(decisions) == 1
    assert original in execution_evidence_carriers(seen['leaf_error'])
    assert 'root_error' in seen and witness.read_text().splitlines() == ['original']
    assert model.continuations == 0, 'A repair model ran after the decision observed uncertain completion'
    assert 'result' not in seen and isinstance(seen.get('error'), ResourceOutcomeUnknown)
    assert original in execution_evidence_carriers(seen['error'])
