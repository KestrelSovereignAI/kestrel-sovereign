"""Actual Core credentials, native MCP wire, approvals, SQL and external effects."""

import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
from uuid import uuid4

import pytest
import httpx2
from fastapi import Request
from kestrel_sdk.resources import ResourceConflict
from kestrel_sovereign.agent.parts import part_collector
from kestrel_sovereign.auth import caller_context_scope
from kestrel_sovereign.tools.effect_invocation import ToolEffectOperation
from kestrel_sovereign._async_ownership import raise_owned_outcome

from kestrel_sovereign.features.security.permissions import PermissionLevel
from tests.integration.test_mcp_apps_host import ALICE, BOB, apps_boundary, privacy_transition_probe

pytestmark = [pytest.mark.mcp_integration, pytest.mark.timeout(30)]


def wire(boundary):
    path = Path(str(boundary.witness) + '.input-wire')
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


async def pause(boundary, *, principal='key', session='original-chat'):
    response = await boundary.client.post('/api/owned-initial', headers=boundary.headers(principal),
        json={'marker': 'original-marker', 'session_id': session})
    assert response.status_code == 200, response.text
    assert 'private-form-key' not in response.text and 'server-owned-state' not in response.text
    assert 'Private city?' not in response.text
    assert response.json()['status'] == 'partial'
    part, = [p for p in response.json()['parts'] if p['type'] == 'mcp_input']
    assert set(part['data']) == {'reference', 'session_id'}
    assert part['data']['session_id'] == session
    assert boundary.markers == [] and len(wire(boundary)) == 1
    return response.json()


async def pending(boundary, *, principal='key', session='original-chat'):
    return await boundary.client.get('/api/mcp/inputs', headers=boundary.headers(principal),
        params={'session_id': session})


async def answer(boundary, row, *, principal='key', session='original-chat', responses=None, **changes):
    return await boundary.client.post('/api/mcp/inputs/' + row['request_id'] + '/responses',
        headers=boundary.headers(principal), params={'session_id': session},
        json=dict(revision=row['revision'], operation_id=uuid4().hex,
            responses={'private-form-key': {'action': 'accept', 'content': {'city': 'owned-private-answer'}}}
                if responses is None else responses, **changes))


async def open_input_app(b):
    reference = await b.publish(marker='opened-once')
    opened = await b.open(reference)
    assert opened.status_code == 200, opened.text
    view_id = opened.json()['view_id']
    tool = next(item for item in b.tools if item.tool.native_tool.name == 'mutate_marker')
    await b.permissions.set_permission('mcp:owned-apps', tool.binding.name, PermissionLevel.ALLOW)
    return view_id


@pytest.mark.parametrize(('apps_boundary', 'app_input'), [
    ({'operator_input': 'form'}, False),
    ({'operator_input': 'form', 'input_tool': 'mutate_marker'}, True),
], indirect=['apps_boundary'])
async def test_input_response_takes_privacy_mutex_before_persistence_lease(
        apps_boundary, app_input, monkeypatch):
    b = apps_boundary
    # Select the actual paused form through the same native routes as the UI.
    if app_input:
        row, _ = await pause_app(b)
    else:
        await pause(b)
        row, = (await pending(b)).json()['inputs']
    host = b.feature._input_host
    read, release = asyncio.Event(), asyncio.Event()
    original_read = host.store.read_for

    async def held_read(*args, **kwargs):
        result = await original_read(*args, **kwargs)
        read.set()
        await release.wait()
        return result

    monkeypatch.setattr(host.store, 'read_for', held_read)
    response = asyncio.create_task(answer(b, row))
    try:
        await asyncio.wait_for(read.wait(), 5)
        async with privacy_transition_probe(b.agent, monkeypatch) as (transition, inversion):
            release.set()
            continued = await asyncio.wait_for(response, 5)
            if not inversion:
                await asyncio.wait_for(transition, 5)
            assert not inversion, (
                'MCP input response retained a shared persistence lease before its privacy mutex, '
                'letting a real privacy transition form the opposite lock order')
            assert continued.status_code == 200, continued.text
            assert continued.json()['state'] == 'RESOLVED'
            assert len(wire(b)) == 2
    finally:
        release.set()
        await asyncio.gather(response, return_exceptions=True)


async def pause_app(b):
    view_id = await open_input_app(b)
    called = await b.client.post('/api/mcp/apps/views/' + view_id + '/tools/call',
        params={'session_id': 'original-chat'}, headers=b.headers(),
        json={'name': 'mutate_marker', 'arguments': {'marker': 'continued-once'}, 'operation_id': uuid4().hex})
    assert called.status_code == 202, called.text
    assert set(called.json()) == {'input_required'}
    row, = (await pending(b)).json()['inputs']
    assert called.json()['input_required'] == {'reference': row['request_id'], 'session_id': 'original-chat'}
    assert b.markers == ['opened-once'] and len(wire(b)) == 1
    return row, view_id


@pytest.mark.parametrize('apps_boundary', [{'operator_input': 'form', 'input_tool': 'mutate_marker',
    'hold_first_input': True}], indirect=True)
async def test_parallel_app_calls_recheck_pending_input_before_second_native_dispatch(apps_boundary, monkeypatch):
    b = apps_boundary
    view_id = await open_input_app(b)
    original_view = b.host._views[view_id].view
    def call(marker):
        return b.client.post('/api/mcp/apps/views/' + view_id + '/tools/call',
            params={'session_id': 'original-chat'}, headers=b.headers(),
            json={'name': 'mutate_marker', 'arguments': {'marker': marker}, 'operation_id': uuid4().hex})
    first = asyncio.create_task(call('continued-once'))
    second = None
    try:
        await asyncio.wait_for(b.input_barrier[0].wait(), 5)
        attempted = asyncio.Event()
        original_operation = b.host._operation

        @asynccontextmanager
        async def observed_operation(*args, **kwargs):
            attempted.set()
            async with original_operation(*args, **kwargs) as check:
                yield check

        monkeypatch.setattr(b.host, '_operation', observed_operation)
        second = asyncio.create_task(call('must-not-start'))
        await asyncio.wait_for(attempted.wait(), 5)
        assert not second.done() and len(original_view._calls) == 1
        # The second real HTTP request now waits at the host mutex, before
        # entering its View child. It must observe the first input checkpoint
        # when that original mutex admission resumes.
        b.input_barrier[1].set()
        response = await asyncio.wait_for(second, 5)
        assert response.status_code == 404, response.text
        assert response.json() == {'error': {'code': 'mcp_app_unavailable'}}, (
            'A concurrent App call passed the original pending-input View fence')
        assert len(wire(b)) == 1
        initial = await asyncio.wait_for(first, 5)
        assert initial.status_code == 202, initial.text
        row, = (await pending(b)).json()['inputs']
        resumed = await answer(b, row)
        assert resumed.status_code == 200, resumed.text
        assert b.markers == ['opened-once', 'continued-once'] and len(wire(b)) == 2
    finally:
        b.input_barrier[1].set()
        await asyncio.gather(first, *([second] if second is not None else []), return_exceptions=True)


@pytest.mark.parametrize('apps_boundary', [{'operator_input': 'mixed'}], indirect=True)
async def test_private_form_is_durable_and_fresh_governed_continuation_is_single_use(apps_boundary):
    b = apps_boundary
    await pause(b)
    listed = await pending(b)
    assert listed.status_code == 200, listed.text
    assert 'no-store' in listed.headers['cache-control']
    assert 'server-owned-state' not in listed.text and 'original-marker' not in listed.text
    row, = listed.json()['inputs']
    selected = await b.client.get('/api/mcp/inputs/' + row['request_id'],
        headers=b.headers(), params={'session_id': 'original-chat'})
    assert selected.status_code == 200 and selected.json() == row
    assert selected.json()['server_url'].startswith('http://127.0.0.1:')
    assert set(row['forms']) == {'private-form-key'} and row['resumable']
    assert row['forms']['private-form-key']['message'] == 'Private city?'
    initial = b.feature._input_host.store
    audit = await initial.database.fetchall('SELECT reason FROM mcp_input_decisions')
    assert audit == [('url_mode_declined',)]
    continued = await answer(b, row)
    assert continued.status_code == 200, continued.text
    assert continued.json()['state'] == 'RESOLVED'
    app_part, = [part for part in continued.json()['tool_result'].get('parts', []) if part['type'] == 'mcp_app']
    assert app_part['data']['session_id'] == 'original-chat'
    opened = await b.open(app_part['data']['reference'])
    assert opened.status_code == 200, opened.text
    assert opened.json()['result']['_meta']['private-witness'] == 'original-marker'
    recovered = await b.client.get('/api/mcp/inputs/' + row['request_id'], headers=b.headers(),
        params={'session_id': 'original-chat'})
    assert recovered.json().get('continuation', {}).get('tool_result') == continued.json()['tool_result'], (
        'A reloaded input card lost its completed audited publication')
    assert b.markers == ['original-marker']
    calls = wire(b)
    assert len(calls) == 2 and calls[0]['arguments'] == calls[1]['arguments'] == {'marker': 'original-marker'}
    assert 'requestState' not in calls[0] and calls[1]['requestState'] == 'server-owned-state'
    assert calls[1]['inputResponses'] == {
        'private-form-key': {'action': 'accept', 'content': {'city': 'owned-private-answer'}},
        'private-url-key': {'action': 'decline'}}
    assert (await answer(b, row)).status_code == 409
    assert b.markers == ['original-marker'] and len(wire(b)) == 2
    assert (await pending(b)).json() == {'inputs': []}
    dispatches = await b.backend.fetch_all('SELECT result_status,session_id FROM a2a_tool_dispatches')
    assert len(dispatches) == 2 and all(row[1] == 'original-chat' for row in dispatches)


@pytest.mark.parametrize('apps_boundary', [{'operator_input': 'form', 'input_tool': 'mutate_marker'}], indirect=True)
async def test_existing_app_receives_input_reference_then_exact_private_native_result(apps_boundary):
    b = apps_boundary
    row, view_id = await pause_app(b)
    repeated = await b.client.post('/api/mcp/apps/views/' + view_id + '/tools/call',
        params={'session_id': 'original-chat'}, headers=b.headers(),
        json={'name': 'mutate_marker', 'arguments': {'marker': 'duplicate-initial'}, 'operation_id': uuid4().hex})
    assert repeated.status_code == 404, 'An App with pending input admitted another initial native call'
    assert len(wire(b)) == 1
    resumed = await answer(b, row)
    assert resumed.status_code == 200, resumed.text
    assert resumed.json()['app_result']['_meta']['private-witness'] == 'continued-once'
    assert resumed.json()['app_result']['_meta']['io.modelcontextprotocol/serverInfo']['name'] == 'permission-boundary-witness'
    assert resumed.json()['app_result']['content'] == [{'type': 'text', 'text': 'Recorded: continued-once'}]
    assert resumed.json()['state'] == 'RESOLVED'
    assert b.markers == ['opened-once', 'continued-once'] and len(wire(b)) == 2


@pytest.mark.parametrize('apps_boundary', [{'operator_input': 'form', 'input_tool': 'mutate_marker'}], indirect=True)
async def test_closed_app_is_not_adopted_by_a_fresh_input_reader(apps_boundary):
    b = apps_boundary
    row, view_id = await pause_app(b)
    closed = await b.client.delete('/api/mcp/apps/views/' + view_id,
        params={'session_id': 'original-chat'}, headers=b.headers())
    assert closed.status_code == 200, closed.text
    current, = (await pending(b)).json()['inputs']
    assert not current['resumable']
    assert (await answer(b, row)).status_code == 409
    assert b.markers == ['opened-once'] and len(wire(b)) == 1


@pytest.mark.parametrize('apps_boundary', [{'operator_input': 'form', 'input_tool': 'mutate_marker'}], indirect=True)
async def test_input_result_retains_app_view_until_actual_outer_transport_settles(apps_boundary):
    b = apps_boundary
    row, view_id = await pause_app(b)
    original_view = b.host._views[view_id].view
    reached, cancelled, settle = asyncio.Event(), asyncio.Event(), asyncio.Event()
    cancellations = []

    async def blocked_transport(scope, receive, send):
        async def physical_send(message):
            if message['type'] == 'http.response.body' and b'private-witness' in message.get('body', b''):
                reached.set()
                try:
                    await settle.wait()
                except asyncio.CancelledError:
                    cancellations.append(True)
                    cancelled.set()
                    await settle.wait()
                    raise
            await send(message)
        await b.app(scope, receive, physical_send)

    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=blocked_transport),
            base_url='https://host.apps.test') as client:
        call = asyncio.create_task(client.post('/api/mcp/inputs/' + row['request_id'] + '/responses',
            params={'session_id': 'original-chat'}, headers=b.headers(),
            json={'revision': row['revision'], 'operation_id': uuid4().hex,
                'responses': {'private-form-key': {'action': 'accept', 'content': {'city': 'owned-private-answer'}}}}))
        drain = None
        try:
            await asyncio.wait_for(reached.wait(), 5)
            assert len(original_view._delivery_tasks) == 1, 'The original App View did not own its actual response send'
            b.host.withdraw_private_data()
            drain = asyncio.create_task(b.host.maintain())
            await asyncio.wait_for(cancelled.wait(), 5)
            b.host.withdraw_private_data()
            await asyncio.sleep(0)
            assert not call.done() and not drain.done() and cancellations == [True]
            settle.set()
            await asyncio.wait_for(drain, 5)
            result, = await asyncio.gather(call, return_exceptions=True)
            assert isinstance(result, BaseException)
            assert b.markers == ['opened-once', 'continued-once'] and len(wire(b)) == 2
            assert not b.host._views
        finally:
            settle.set()
            await asyncio.gather(call, *([drain] if drain else []), return_exceptions=True)


@pytest.mark.parametrize('apps_boundary', [{'operator_input': 'form'}], indirect=True)
async def test_wrong_principal_session_and_stale_or_invalid_answer_never_reach_wire(apps_boundary):
    b = apps_boundary
    await pause(b, principal=ALICE)
    row, = (await pending(b, principal=ALICE)).json()['inputs']
    assert (await pending(b, principal=BOB)).json() == {'inputs': []}
    assert (await pending(b, principal=ALICE, session='other')).json() == {'inputs': []}
    assert (await answer(b, row, principal=BOB)).status_code == 409
    assert (await answer(b, row, principal=ALICE, session='other')).status_code == 409
    invalid = await answer(b, row, principal=ALICE,
        responses={'private-form-key': {'action': 'accept', 'content': {'city': 5}}})
    assert invalid.status_code == 400, invalid.text
    stale = await answer(b, {**row, 'revision': row['revision'] + 1}, principal=ALICE)
    assert stale.status_code == 409, stale.text
    assert len(wire(b)) == 1 and b.markers == []
    assert (await answer(b, row, principal=ALICE)).status_code == 200
    assert b.markers == ['original-marker']


@pytest.mark.parametrize('apps_boundary', [{'operator_input': 'form'}], indirect=True)
async def test_response_rechecks_current_tool_permission_before_consuming_checkpoint(apps_boundary):
    b = apps_boundary
    await pause(b)
    row, = (await pending(b)).json()['inputs']
    await b.permissions.set_permission('mcp:owned-apps', row['tool'], PermissionLevel.DENY)
    denied = await answer(b, row)
    assert denied.status_code == 200, denied.text
    assert denied.json()['state'] == 'WAITING'
    assert len(wire(b)) == 1 and b.markers == []
    await b.permissions.set_permission('mcp:owned-apps', row['tool'], PermissionLevel.ASK)
    submitted = asyncio.create_task(answer(b, row))
    try:
        approval = await asyncio.wait_for(b.requests.get(), 3)
        assert len(wire(b)) == 1 and b.markers == []
        # The waiting HTTP operation does not occupy a conversation turn.
        assert b.agent.get_turn_bound_session_id() is None
        accepted = await b.queue.submit_decision(approval.id, approved=True, scope='once')
        assert accepted.in_memory and accepted.persisted
        continued = await submitted
        assert continued.status_code == 200, continued.text
        assert b.markers == ['original-marker']
    finally:
        if not submitted.done():
            submitted.cancel()
        await asyncio.gather(submitted, return_exceptions=True)


@pytest.mark.parametrize('apps_boundary', [{'operator_input': 'url'}], indirect=True)
async def test_url_only_request_is_audited_and_declined_without_navigation_or_human_wait(apps_boundary):
    b = apps_boundary
    response = await b.client.post('/api/owned-initial', headers=b.headers(),
        json={'marker': 'must-not-run', 'session_id': 'original-chat'})
    assert response.status_code == 200, response.text
    assert len(wire(b)) == 2 and b.markers == []
    assert (await pending(b)).json() == {'inputs': []}
    assert await b.feature._input_host.store.database.fetchall('SELECT reason FROM mcp_input_decisions') == [('url_mode_declined',)]


@pytest.mark.parametrize('apps_boundary', [{'operator_input': 'form'}], indirect=True)
async def test_new_input_host_and_native_client_recover_original_checkpoint_without_initial_replay(apps_boundary):
    from kestrel_feature_mcp.connection import bind_tool_caller

    b = apps_boundary
    await pause(b)
    row, = (await pending(b)).json()['inputs']
    registered = next(item for batch in b.feature._input_runtime._registrations.values()
        for item in batch if item.binding.name == row['tool'])
    old_connection = registered.tool.bound_caller.connection
    url = old_connection.url
    b.feature._unmount_tools('mcp:owned-apps')
    await old_connection.close()
    old_host = b.feature._input_host
    await b.feature._close_inputs()
    await b.feature._start_inputs()
    assert old_host.retired and b.feature._input_host is not old_host
    connection = b.feature._new_mcp_connection(url)
    try:
        native_tools = await connection.connect()
        caller = bind_tool_caller(connection, lambda: True, 'owned-apps-peer',
            tool_names=[tool.name for tool in native_tools])
        b.feature._mount_tools('mcp:owned-apps', 'app',
            [tool.model_dump(mode='json', by_alias=True) for tool in native_tools], caller)
        recovered, = (await pending(b)).json()['inputs']
        assert recovered['request_id'] == row['request_id'] and recovered['resumable']
        resumed = await answer(b, recovered)
        assert resumed.status_code == 200, resumed.text
        assert resumed.json()['state'] == 'RESOLVED'
        assert len(wire(b)) == 2 and b.markers == ['original-marker']
    finally:
        b.feature._unmount_tools('mcp:owned-apps')
        await connection.close()


@pytest.mark.parametrize('apps_boundary', [{'operator_input': 'form'}], indirect=True)
async def test_two_real_http_responses_cannot_execute_the_continuation_twice(apps_boundary):
    b = apps_boundary
    await pause(b)
    row, = (await pending(b)).json()['inputs']
    responses = await asyncio.gather(answer(b, row), answer(b, row))
    assert any(response.status_code == 200 and response.json()['state'] == 'RESOLVED'
        for response in responses), [response.text for response in responses]
    assert all(response.status_code in {200, 409} for response in responses), [r.text for r in responses]
    assert b.markers == ['original-marker'] and len(wire(b)) == 2
    current = await b.client.get('/api/mcp/inputs/' + row['request_id'], headers=b.headers(),
        params={'session_id': 'original-chat'})
    assert current.status_code == 200 and current.json()['state'] == 'RESOLVED'


@pytest.mark.parametrize('apps_boundary', [{'operator_input': 'form', 'input_rounds': 2}], indirect=True)
async def test_reloaded_original_card_recovers_next_native_round_without_initial_replay(apps_boundary):
    b = apps_boundary
    await pause(b)
    first, = (await pending(b)).json()['inputs']
    continued = await answer(b, first)
    assert continued.status_code == 200, continued.text
    next_id = continued.json()['next_request_id']
    assert next_id and next_id != first['request_id']
    assert b.markers == [] and len(wire(b)) == 2
    reloaded = await b.client.get('/api/mcp/inputs/' + first['request_id'], headers=b.headers(),
        params={'session_id': 'original-chat'})
    assert reloaded.status_code == 200, reloaded.text
    assert reloaded.json()['continuation']['next_request_id'] == next_id
    second, = (await pending(b)).json()['inputs']
    assert second['request_id'] == next_id
    assert (await answer(b, first)).status_code == 409
    completed = await answer(b, second)
    assert completed.status_code == 200, completed.text
    assert completed.json()['state'] == 'RESOLVED' and completed.json()['next_request_id'] is None
    assert b.markers == ['original-marker'] and len(wire(b)) == 3
    assert [call.get('requestState') for call in wire(b)] == [None, 'server-owned-state', 'server-owned-state-2']


@pytest.mark.parametrize('apps_boundary', [{'operator_input': 'form', 'input_max_payload_bytes': 4096}], indirect=True)
async def test_completed_native_effect_with_unstored_result_never_reports_invalid_input(apps_boundary):
    b = apps_boundary
    marker = 'owned-large-marker-' + 'x' * 1500
    initial = await b.client.post('/api/owned-initial', headers=b.headers(),
        json={'marker': marker, 'session_id': 'original-chat'})
    assert initial.status_code == 200 and initial.json()['status'] == 'partial', initial.text
    row, = (await pending(b)).json()['inputs']
    completed = await answer(b, row)
    assert completed.status_code == 503, 'A completed native effect was misreported as invalid operator input'
    assert completed.json()['error']['code'] == 'mcp_input_outcome_unknown_do_not_replay'
    assert b.markers == [marker] and len(wire(b)) == 2
    assert (await answer(b, row)).status_code == 409
    assert b.markers == [marker] and len(wire(b)) == 2


@pytest.mark.parametrize('apps_boundary', [{'operator_input': 'form'}], indirect=True)
async def test_pending_input_fences_the_original_parent_turn_before_another_tool_admission(apps_boundary):
    b = apps_boundary
    registered = next(item for item in b.tools if item.tool.native_tool.name == 'open_marker')

    @b.app.post('/api/owned-sequential-input')
    async def sequential(request: Request):
        with caller_context_scope(request.state.caller), part_collector(host=b.agent, session_id='original-chat'):
            async def operation():
                first = await registered.binding.invoke({'marker': 'original-marker'},
                    session_id='original-chat', operation_id=uuid4().hex, source='chat')
                try:
                    await registered.binding.invoke({'marker': 'original-marker'},
                        session_id='original-chat', operation_id=uuid4().hex, source='chat')
                except ResourceConflict:
                    pass
                return first
            outcome = await b.agent._execute_tool_operation_at_stop_boundary(operation,
                name='owned-parent-input-turn', tool_operation=ToolEffectOperation('owned-input-parent',
                    uuid4().hex, 'owned-parent', 'owned-parent', {}, 'original-chat'))
            return raise_owned_outcome(outcome, operation='owned-parent-input-turn', retain_originals=True)

    result = await b.client.post('/api/owned-sequential-input', headers=b.headers())
    assert result.status_code == 200, result.text
    assert len(wire(b)) == 1, 'A confirmed MCP checkpoint allowed a second initial operation in its original turn'
    assert b.markers == []
    assert len((await pending(b)).json()['inputs']) == 1


@pytest.mark.parametrize('apps_boundary', [{'operator_input': 'form'}], indirect=True)
@pytest.mark.parametrize('streaming', [False, True])
async def test_orchestrator_returns_control_before_followup_model_work(apps_boundary, streaming):
    from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

    b = apps_boundary
    registered = next(item for item in b.tools if item.tool.native_tool.name == 'open_marker')

    @b.app.post('/api/owned-input-orchestrator')
    async def orchestrate(request: Request):
        with caller_context_scope(request.state.caller), part_collector(host=b.agent, session_id='original-chat'):
            async def operation():
                # A typed adapter boundary drives actual Core dispatch. The
                # real unconfigured service cannot supply a follow-up model.
                response = LLMResponse(content='', tool_calls=[ToolCall(id=uuid4().hex,
                    name=registered.binding.name, arguments={'marker': 'original-marker'})])
                options = dict(feature_tools=b.agent._build_all_tools(), system_prompt='Owned input control test',
                    force_local_only=True, effective_model='owned-unconfigured-model',
                    user_message='Run the owned tool once', session_id='original-chat', max_iterations=2)
                if streaming:
                    return ''.join([chunk async for chunk in b.agent._handle_orchestrator_response_streaming(
                        response, tool_events=[], tool_results=[], **options)])
                return await b.agent._handle_orchestrator_response_impl(response, **options)
            outcome = await b.agent._execute_tool_operation_at_stop_boundary(operation,
                name='owned-input-orchestrator', tool_operation=ToolEffectOperation('owned-input-orchestrator',
                    uuid4().hex, 'owned-parent', 'owned-parent', {}, 'original-chat'))
            return {'text': raise_owned_outcome(outcome, operation='owned-input-orchestrator', retain_originals=True)}

    response = await b.client.post('/api/owned-input-orchestrator', headers=b.headers())
    assert response.status_code == 200, response.text
    assert 'waiting for operator input' in response.json()['text']
    assert 'No final result' in response.json()['text']
    assert len(wire(b)) == 1 and b.markers == []


async def test_declared_input_configuration_prepares_before_native_feature_activation(
    native_tool_effect_agent,
):
    from kestrel_feature_mcp.feature import MCPAgent
    from kestrel_sovereign.feature_registry import get_package_for_feature
    from kestrel_sovereign.security.capability_authority import HostCapabilityAccessDenied

    agent = await native_tool_effect_agent(did='did:test:owned-input-config-boot')
    package = get_package_for_feature('MCPAgent')
    assert package
    config = agent._agent_toml_path()
    assert config is not None and not config.exists()
    config.write_text(f'[features.{package.name}.config.inputs]\nenabled=true\n'
        f'[features.{package.name}.config.inputs.host]\nretention_seconds=60.0\n')
    feature = MCPAgent(agent)
    try:
        try:
            await agent._register_feature(feature)
        except HostCapabilityAccessDenied as error:
            pytest.fail('Declared MCP input settings could not prepare before native activation: ' + type(error).__name__)
        assert feature._input_host is not None and not feature._input_host.retired
        assert agent.feature_contribution_runtime.is_active(feature)
        feature._input_host.require_current()
    finally:
        await agent._unregister_feature_runtime(feature, unload=True)


@pytest.mark.parametrize('apps_boundary', [{'operator_input': 'form'}], indirect=True)
async def test_descriptive_origin_cannot_persist_input_without_actual_core_admission(apps_boundary):
    from kestrel_sovereign.mcp_input import MCPInputOrigin
    from kestrel_sovereign.security.capability_authority import HostCapabilityAccessDenied
    from kestrel_sovereign.tools.effect_journal import ToolEffectJournalUnavailable

    b = apps_boundary
    description = MCPInputOrigin('invented-operation', 'invented-session', 'invented-principal', True, lambda: None)
    try:
        await b.feature._input_host.suspend(description, connection_id='invented-connection',
            contract_digest='a' * 64, method='tools/call', payload={'private': 'not-authorized'})
    except (HostCapabilityAccessDenied, ToolEffectJournalUnavailable):
        pass
    assert await b.feature._input_host.store.database.fetchone('SELECT COUNT(*) FROM mcp_inputs') == (0,), (
        'A descriptive origin wrote a checkpoint before the host required its actual native admission')
