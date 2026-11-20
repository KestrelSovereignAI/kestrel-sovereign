"""Core custody on actual governed HTTP dispatch, not Task-wire acceptance.

These probes replace only a registered handler body. Authentication, selection,
journal, storage, permission and deferred retirement are the native Core path.
The paired Feature tests separately have to prove native Task peer traffic.
"""

import asyncio
from dataclasses import replace
import json

import pytest
import pytest_asyncio
from mcp import types
from kestrel_sdk.resources import ResourceConflict, ResourceOutcomeUnknown
from kestrel_sdk.tools.result import ToolResult

from kestrel_sovereign.mcp_tasks import MCPTaskHostPolicy
from kestrel_sovereign.mcp_task_store import MCPTaskIntegrityError
from kestrel_sovereign.security.capability_authority import HostCapabilityAccessDenied
from kestrel_sovereign.tools.effect_journal import ToolEffectJournalUnavailable, current_tool_effect_execution
from tests.integration.test_mcp_apps_host import apps_boundary as apps_boundary
from tests.shared.mcp_task_custody import attach as attach_task, binding as task_binding, submission
from tests.utils.original_failure import lost_original_database_commit_ack
from kestrel_sovereign.execution_custody import execution_evidence_carriers

pytestmark = [pytest.mark.mcp_integration, pytest.mark.timeout(30)]

CONTRACT = b'{"connection":"owned-apps","tool":{"name":"initial"}}'


@pytest_asyncio.fixture
async def task_boundary(apps_boundary, request):
    b = apps_boundary
    host = b.agent.create_mcp_task_host(b.feature, input_host=b.feature._input_host,
        configuration=getattr(request, 'param', {}), is_current=lambda: not b.feature._closing)
    await host.initialize()
    try:
        yield b, host
    finally:
        await host.close()
        # A borrowed host must never close the original effect database.
        assert await b.agent.tool_effect_store.database.fetchone('SELECT 1') == (1,)


def initial(b):
    return next(item for item in b.tools if item.tool.native_tool.name == 'open_marker')


async def invoke(b, marker='approved'):
    return await b.client.post('/api/owned-initial', headers=b.headers('key'),
        json={'marker': marker, 'session_id': 'original-chat'})


async def admit(host, selected):
    return await host.admit(selected.binding, connection_name='owned-apps',
        authorization_id='a' * 64, contract=CONTRACT)


@pytest.mark.parametrize('apps_boundary', [{'operator_input': 'form'}], indirect=True)
@pytest.mark.parametrize('task_boundary', [{'maintenance_page_records': 1}], indirect=True)
async def test_task_maintenance_visits_one_page_per_tick_and_wraps(task_boundary, monkeypatch):
    _, host = task_boundary
    reservations = [await host.store.reserve(
        task_binding(agent_did=host.agent.did, operation_id=f'maintenance-task-{index}'), submission())
        for index in range(3)]
    handles = sorted(item.record.handle for item in reservations)
    opened = []
    open_row = host.store._open

    def record_open(row):
        result = open_row(row)
        opened.append(result.handle)
        return result

    monkeypatch.setattr(host.store, '_open', record_open)
    for expected in ([handles[0]], [handles[1]], [handles[2]], [], [handles[0]]):
        opened.clear()
        await host.maintain()
        assert opened == expected


@pytest.mark.parametrize('apps_boundary', [{'operator_input': 'form'}], indirect=True)
@pytest.mark.parametrize('task_boundary', [{'maintenance_page_records': 1}], indirect=True)
async def test_task_maintenance_rejects_duplicate_authenticated_operations_across_pages(task_boundary):
    _, host = task_boundary
    store = host.store
    other = await store.reserve(task_binding(agent_did=host.agent.did), submission())
    reservation = await store.reserve(
        task_binding(agent_did=host.agent.did, operation_id='different-task'), submission())
    attached = await attach_task(store, reservation.issuance)
    mutation = await store.reserve_mutation(attached, operation_id='different-mutation', kind='cancel')

    # Seed a damaged inventory with two authentic rows, each of which is
    # internally valid. Only a scan across both can detect the reused ID.
    envelope = mutation.record.envelope
    receipts = envelope['control']['mutations']
    receipts[other.record.binding.operation_id] = receipts.pop('different-mutation')
    async with store.write_transaction():
        await store._lock()
        actual, row = await store._read_row(attached.handle)
        await store._persist(actual, row, actual.phase, envelope)

    assert await store.read_handle(attached.handle) is not None
    assert await store.read_handle(other.record.handle) == other.record
    await host.maintain()
    with pytest.raises(MCPTaskIntegrityError, match='conflicting original custody'):
        await host.maintain()


@pytest.mark.parametrize('apps_boundary', [{'operator_input': 'form'}], indirect=True)
@pytest.mark.parametrize('outcome', ['complete', 'error', 'task', 'refusal', 'unknown'])
async def test_native_handler_custody_keeps_exact_args_and_retires_only_confirmed_owners(
        task_boundary, monkeypatch, outcome):
    b, host = task_boundary
    selected = initial(b)
    actual_native = types.CallToolResult(result_type='complete', content=[types.TextContent(type='text', text='actual-result')],
        is_error=outcome == 'error').model_dump(mode='json', by_alias=True, exclude_unset=True)
    if outcome == 'task':
        from kestrel_feature_mcp.tasks_protocol import CreateTaskResult

        actual_native = CreateTaskResult.model_validate(dict(resultType='task', taskId='owned-native-task',
            status='working', createdAt='2026-11-19T00:00:00Z', lastUpdatedAt='2026-11-19T00:00:00Z',
            ttlMs=None)).model_dump(mode='json', by_alias=True, exclude_unset=True)
    captured = []

    async def handler(**arguments):
        round = await admit(host, selected)
        captured.append(round)
        frame = current_tool_effect_execution(b.agent)
        assert round.issuance.record.binding.operation_id == frame.claim.claim_id
        sealed = await host.store.open_submission(round.issuance.record)
        assert sealed.arguments == {'marker': 'approved'} and sealed.contract == CONTRACT
        assert sealed.delivery.kind == 'chat' and sealed.delivery.session_id == 'original-chat'
        assert round.issuance.record.binding.origin.kind == 'interactive'
        assert await host.store.database.fetchone('SELECT phase FROM mcp_task_records') == ('ISSUING',)
        if outcome == 'refusal':
            await round.refuse()
        else:
            round.start_wire()
            if outcome == 'unknown':
                round.unknown()
                raise ResourceOutcomeUnknown('Owned unconfirmed result')
            if outcome == 'task':
                await round.attach(actual_native)
            else:
                await round.settle_result(actual_native)
        return ToolResult.ok('Owned host component completed').to_dict()

    monkeypatch.setattr(selected.tool, 'execute', handler)
    response = await invoke(b)
    assert response.status_code == (503 if outcome == 'unknown' else 200), response.text
    record, = await host.store.list_retained()
    expected = {'task': 'ATTACHED', 'refusal': 'NOT_SENT', 'unknown': 'ISSUING'}.get(outcome, 'SETTLED')
    assert record.phase == expected
    if outcome in {'complete', 'error', 'task'}:
        assert record.envelope['native'] == actual_native
    held = len(selected.binding.registration.task_holds)
    assert held == (1 if outcome in {'task', 'unknown'} else 0), 'Owner retention outlived confirmed dispatch retirement'
    with pytest.raises((ResourceConflict, HostCapabilityAccessDenied, ToolEffectJournalUnavailable)):
        captured[0].require_wire()


@pytest.mark.parametrize('apps_boundary', [{'operator_input': 'form'}], indirect=True)
@pytest.mark.parametrize('mismatch', ['tool', 'owner', 'registration', 'agent'])
async def test_host_refuses_a_binding_different_from_the_actual_engine_selection(task_boundary, monkeypatch, mismatch):
    b, host = task_boundary
    selected = initial(b)
    foreign = {'tool': object(), 'owner': 'unselected-owner', 'registration': object(), 'agent': object()}[mismatch]
    supplied = replace(selected.binding, **{mismatch: foreign})
    refused = []

    async def handler(**arguments):
        with pytest.raises(ResourceConflict):
            await host.admit(supplied, connection_name='owned-apps', authorization_id='a' * 64, contract=CONTRACT)
        refused.append(True)
        return ToolResult.ok('Refused foreign binding').to_dict()

    monkeypatch.setattr(selected.tool, 'execute', handler)
    assert (await invoke(b)).status_code == 200
    assert refused == [True]
    assert await host.store.list_retained() == () and not selected.binding.registration.task_holds


@pytest.mark.parametrize('apps_boundary', [{'operator_input': 'form'}], indirect=True)
async def test_a_copied_child_cannot_dispatch_its_parents_task_round(task_boundary, monkeypatch):
    b, host = task_boundary
    selected = initial(b)
    admitted = []

    async def handler(**arguments):
        round = await admit(host, selected)
        async def child():
            try:
                round.require_wire()
            except (ResourceConflict, HostCapabilityAccessDenied, ToolEffectJournalUnavailable):
                admitted.append(False)
            else:
                admitted.append(True)
        await asyncio.create_task(child())
        await round.refuse()
        return ToolResult.ok('Owned child probe').to_dict()

    monkeypatch.setattr(selected.tool, 'execute', handler)
    assert (await invoke(b)).status_code == 200
    assert admitted == [False], 'A copied child reused the actual parent Task round'


@pytest.mark.parametrize('apps_boundary', [{'operator_input': 'form'}], indirect=True)
async def test_another_current_tool_cannot_be_substituted_for_the_actual_handler(task_boundary, monkeypatch):
    b, host = task_boundary
    selected = initial(b)
    other = next(item for item in b.tools if item is not selected)
    assert other.binding.is_current
    refused = []

    async def handler(**arguments):
        try:
            round = await admit(host, other)
        except ResourceConflict:
            refused.append(True)
        else:
            refused.append(False)
            await round.refuse()
        return ToolResult.ok('Observed actual selection boundary').to_dict()

    monkeypatch.setattr(selected.tool, 'execute', handler)
    assert (await invoke(b)).status_code == 200
    assert refused == [True], 'A different current mounted tool replaced the actual engine selection'
    assert await host.store.list_retained() == ()


@pytest.mark.parametrize('configuration', [dict(max_active=65, max_records=64),
    dict(max_native_claim_bytes=2_000_000), dict(max_continuations=-1), dict(max_active=True),
    dict(unknown_setting=1)])
def test_policy_refuses_invalid_or_unretainable_host_budgets(configuration):
    with pytest.raises(ValueError):
        MCPTaskHostPolicy.model_validate(configuration)


@pytest.mark.parametrize('apps_boundary', [{'operator_input': 'form'}], indirect=True)
@pytest.mark.parametrize('lost_at', ['reserve', 'settle'])
async def test_lost_native_commit_ack_keeps_custody_and_the_original_error_until_read_reconciliation(
        task_boundary, monkeypatch, lost_at):
    b, host = task_boundary
    selected = initial(b)
    armed, effects = [], []
    execute = host.store.database.execute

    async def observe(query, *args, **kwargs):
        result = await execute(query, *args, **kwargs)
        if ((lost_at == 'reserve' and query.startswith('INSERT INTO mcp_task_records'))
                or (lost_at == 'settle' and query.startswith('UPDATE mcp_task_records SET remote_digest='))):
            armed.append(True)
        return result

    async def accept():
        return bool(armed)

    async def handler(**arguments):
        round = await admit(host, selected)
        round.start_wire()
        effects.append('one-native-disposition')
        await round.settle_result(types.CallToolResult(result_type='complete', content=[]).model_dump(
            mode='json', by_alias=True, exclude_unset=True))
        return ToolResult.ok('Must not confirm a lost native commit ACK').to_dict()

    monkeypatch.setattr(host.store.database, 'execute', observe)
    monkeypatch.setattr(selected.tool, 'execute', handler)
    with lost_original_database_commit_ack(monkeypatch, host.store, accept_commit=accept) as originals:
        result, = await asyncio.gather(invoke(b), return_exceptions=True)
    assert len(originals) == 1
    assert isinstance(result, BaseException), 'Native commit control was projected as a completed HTTP tool result'
    assert originals[0] in execution_evidence_carriers(result), 'Original native lost-ACK source was erased'
    # Native control prevents even inspecting an untrusted checkpoint or
    # borrowing a new reader to project a local disposition.
    inspected = None
    try:
        inspected = await host.input_host.current_record(object(), retirement_error=result)
    except BaseException as observation:
        pytest.fail('Retirement inspection replaced original native control: ' + repr(observation))
    assert inspected is None
    record, = await host.store.list_retained()
    assert record.phase == ('ISSUING' if lost_at == 'reserve' else 'SETTLED')
    assert effects == ([] if lost_at == 'reserve' else ['one-native-disposition'])
    assert len(selected.binding.registration.task_holds) == 1
    await host.maintain()
    assert not host._unreconciled
    assert len(selected.binding.registration.task_holds) == (1 if lost_at == 'reserve' else 0)
