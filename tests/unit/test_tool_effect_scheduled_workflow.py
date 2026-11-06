"""Native scheduler occurrence, workflow task, SQL journal and file effects.

This is a component path through a real Runner, not a READY/model boot claim.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from types import SimpleNamespace

from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.scheduler.feature import SchedulerFeature
from kestrel_sovereign.features.scheduler.runner import (
    SCHEDULER_PROTOCOL_VERSION, SchedulerRunner, get_current_scheduler_execution,
)
from kestrel_sovereign.features.tasks.feature import TaskFeature
from tests.unit.test_tool_effect_execution import owner
from tests.utils.native_a2a import manager_for
from tests.utils.scheduler_ticks import tick_and_settle


pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(30)]


@pytest_asyncio.fixture
async def scheduled_case(owner, tmp_path):
    database = owner.tool_effect_store.database
    witness = tmp_path / 'scheduled-workflow-effect'
    manager = manager_for(database.backend, NativeRecipientScope(owner.did), owns_backend=False)
    await manager.initialize()
    observed = []

    class FileFeature(Feature):
        @property
        def tool_description(self):
            return 'Owned scheduled workflow file effect'

        async def initialize(self):
            pass

        @tool(name='scheduled_file_effect', description='Write the original scheduled marker')
        async def scheduled_file_effect(self) -> dict:
            execution = get_current_scheduler_execution()
            assert execution is not None and execution.agent_id == owner.did
            observed.append(execution.idempotency_key)
            assert await database.fetchone('SELECT state FROM tool_effect_claims') == ('STARTED',)
            witness.write_text('one original scheduled effect')
            return {'success': True}

    feature = FileFeature(owner)
    task_feature = TaskFeature(owner)
    task_feature.set_task_manager(manager)
    owner.features[feature.name] = feature
    owner.features[task_feature.name] = task_feature
    manager.register_provider(feature.name, feature.get_agent_skills(), feature)
    scheduler = SchedulerFeature(owner)
    scheduler._db, scheduler._agent_id = database, owner.did
    # The real Runner supplies the revocable occurrence scope. Invoke the
    # same canonical tool lookup used by cron sources and custom schedules.
    case = SimpleNamespace(owner=owner, database=database, witness=witness,
        feature=feature, task_feature=task_feature, scheduler=scheduler, observed=observed,
        manager=manager)
    try:
        yield case
    finally:
        await manager.close()


async def seed_occurrence(case, runner, task_name, arguments):
    await runner._ensure_tables()
    due = (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat()
    await case.database.execute('''INSERT INTO scheduled_tasks
        (id,agent_id,task_name,cron_expression,args_json,enabled,next_run_at,
         created_at,schedule_kind,timezone_name,misfire_policy,idempotency_key,
         scheduler_protocol_version)
        VALUES (?,?,?,?,?,1,?,?,'cron','UTC','skip',?,?)''',
        ('owned-workflow-schedule', runner._agent_id, task_name, '* * * * *',
         json.dumps(arguments), due, due, 'owned-workflow-source', SCHEDULER_PROTOCOL_VERSION))


@pytest.mark.parametrize('lane', ['workflow', 'direct'])
async def test_native_scheduled_workflow_admits_its_original_effect_before_dispatch(scheduled_case, lane):
    case = scheduled_case
    owner, database, witness = case.owner, case.database, case.witness
    feature, scheduler, observed = case.feature, case.scheduler, case.observed
    runner = SchedulerRunner(database, owner.did, scheduler._lookup_and_run_tool,
        owner_id='owned-native-workflow-scheduler')
    try:
        arguments = {'steps': [{'feature': feature.name, 'skill': 'scheduled_file_effect'}]} if lane == 'workflow' else {}
        task_name = 'run_workflow' if lane == 'workflow' else 'scheduled_file_effect'
        await seed_occurrence(case, runner, task_name, arguments)
        await tick_and_settle(runner)
        row = await database.fetchone('SELECT status,result_text FROM task_execution_log WHERE task_id=?',
            ('owned-workflow-schedule',))
        assert witness.exists(), f'Scheduled native workflow bypassed journal admission: {row}'
        assert witness.read_text() == 'one original scheduled effect'
        assert len(observed) == 1
        assert row[0] == 'success'
        assert await database.fetchone('SELECT state FROM tool_effect_operations') == ('CLOSED',)
        assert await database.fetchone('SELECT state FROM tool_effect_claims') == ('RETURNED_OK',)
    finally:
        await runner.stop()


async def test_scheduled_redelivery_retains_its_identity_across_transport_principals(scheduled_case):
    from kestrel_sovereign.agent.invocation import invocation_scope, request_provenance
    from kestrel_sovereign.auth import CallerContext, caller_context_scope
    from kestrel_sovereign.tool_effect_store import ToolEffectReconciliationRequired

    case = scheduled_case
    completed = []

    async def execute(name, arguments):
        with invocation_scope(provenance=request_provenance(
                source_kind='http_request', source_locator='/owned/scheduler/first',
                actor='owned-first-controller')), caller_context_scope(
                    CallerContext.sovereign(credential='owned-scheduler-controller')):
            await case.scheduler._lookup_and_run_tool(name, arguments)
        with invocation_scope(provenance=request_provenance(
                source_kind='http_request', source_locator='/owned/scheduler/redelivery',
                actor='owned-second-controller')), caller_context_scope(CallerContext.anonymous()):
            with pytest.raises(ToolEffectReconciliationRequired):
                await case.scheduler._lookup_and_run_tool(name, arguments)
        completed.append(True)
        return 'Original delivery completed; repeated delivery fenced'

    runner = SchedulerRunner(case.database, case.owner.did, execute,
        owner_id='owned-native-redelivery-scheduler')
    try:
        await seed_occurrence(case, runner, 'scheduled_file_effect', {})
        await tick_and_settle(runner)
        assert completed, 'A transport principal changed the original scheduled intent identity'
        assert len(case.observed) == 1, 'Repeated scheduled delivery reached another effect'
        assert await case.database.fetchone('SELECT COUNT(*) FROM tool_effect_operations') == (1,)
        assert await case.database.fetchone('SELECT COUNT(*) FROM tool_effect_claims') == (1,)
    finally:
        await runner.stop()


async def test_scheduled_tool_without_an_occurrence_cannot_open_a_journal(scheduled_case):
    from kestrel_sovereign.tools.effect_journal import ToolEffectJournalUnavailable

    case = scheduled_case
    selected, = case.feature.get_tools()
    with pytest.raises(ToolEffectJournalUnavailable, match='scheduler_occurrence_unavailable'):
        await case.scheduler._run_tool_hook_gated(type(case.feature).__name__, selected, {})
    assert not case.witness.exists()
    assert await case.database.fetchone('SELECT COUNT(*) FROM tool_effect_operations') == (0,)


async def test_foreign_scheduled_occurrence_cannot_open_this_agents_journal(scheduled_case):
    case = scheduled_case
    entered = []

    async def execute(name, arguments):
        from kestrel_sovereign.tools.effect_journal import ToolEffectJournalUnavailable

        with pytest.raises(ToolEffectJournalUnavailable, match='scheduler_occurrence_unavailable'):
            await case.scheduler._lookup_and_run_tool(name, arguments)
        entered.append(True)
        return 'Foreign scheduled dispatch refused'

    runner = SchedulerRunner(case.database, 'did:test:foreign-scheduler-owner', execute,
        owner_id='owned-native-foreign-scheduler')
    try:
        await seed_occurrence(case, runner, 'scheduled_file_effect', {})
        await tick_and_settle(runner)
        assert entered, 'The native foreign occurrence was not exercised'
        assert not case.witness.exists()
        assert await case.database.fetchone('SELECT COUNT(*) FROM tool_effect_operations') == (0,), \
            'A foreign scheduled occurrence opened this agent\'s journal'
    finally:
        await runner.stop()


@pytest.mark.parametrize('withdrawal', ['feature', 'skill'])
async def test_scheduled_tool_withdrawn_during_original_admission_cannot_execute(scheduled_case, monkeypatch, withdrawal):
    case = scheduled_case
    original_admit = case.owner.tool_effect_store.admit
    disabled = []

    async def admit_then_disable(**kwargs):
        attempt = await original_admit(**kwargs)
        if withdrawal == 'feature':
            await case.owner._unregister_feature_runtime(case.feature, unload=False)
        else:
            case.feature.disabled_skills.add('scheduled_file_effect')
        disabled.append(True)
        return attempt

    monkeypatch.setattr(case.owner.tool_effect_store, 'admit', admit_then_disable)
    runner = SchedulerRunner(case.database, case.owner.did, case.scheduler._lookup_and_run_tool,
        owner_id='owned-native-registration-scheduler')
    try:
        await seed_occurrence(case, runner, 'scheduled_file_effect', {})
        await tick_and_settle(runner)
        assert disabled
        if withdrawal == 'feature':
            assert case.feature.enabled is False
        assert not case.witness.exists(), 'Scheduled tool executed a registration disabled during admission'
        assert not case.observed
        assert await case.database.fetchone('SELECT state FROM tool_effect_claims') == ('NOT_EXECUTED',)
    finally:
        await runner.stop()


@pytest.mark.parametrize('wrapper', ['native', 'sdk'])
async def test_native_workflow_does_not_repeat_a_tool_effect_after_its_ordinary_timeout(scheduled_case, wrapper, monkeypatch):
    from kestrel_sdk.features.base import Feature as SDKFeature

    case = scheduled_case
    observed = []
    dispatched = []
    execute_skill = case.manager.execute_skill

    async def record_dispatch(*args, **kwargs):
        dispatched.append(True)
        return await execute_skill(*args, **kwargs)

    monkeypatch.setattr(case.manager, 'execute_skill', record_dispatch)

    class UncertainFeature(Feature):
        get_tools = SDKFeature.get_tools if wrapper == 'sdk' else Feature.get_tools

        @property
        def tool_description(self):
            return 'Native uncertain file effect through the original tool wrapper'

        async def initialize(self):
            pass

        @tool(name='uncertain_file_effect', description='Write before losing the original acknowledgement')
        async def uncertain_file_effect(self) -> dict:
            with case.witness.open('a') as stream:
                stream.write('original uncertain effect\n')
            observed.append(True)
            raise TimeoutError('Original acknowledgement unavailable after the owned file effect')

    feature = UncertainFeature(case.owner)
    case.owner.features[feature.name] = feature
    case.manager.register_provider(feature.name, feature.get_agent_skills(), feature)
    runner = SchedulerRunner(case.database, case.owner.did, case.scheduler._lookup_and_run_tool,
        owner_id='owned-native-uncertain-workflow')
    try:
        await seed_occurrence(case, runner, 'run_workflow', {'steps': [{
            'feature': feature.name, 'skill': 'uncertain_file_effect',
            'max_retries': 1, 'retry_delay_ms': 0,
        }]})
        await tick_and_settle(runner)
        assert len(observed) == 1, 'Native tool normalization allowed an uncertain workflow effect to repeat'
        assert len(dispatched) == 1, 'Workflow retried before consuming the original typed unknown outcome'
        assert case.witness.read_text() == 'original uncertain effect\n'
        states = await case.database.fetchall('SELECT state FROM tool_effect_claims')
        assert states == [('RETURNED_ERROR',), ('UNCONFIRMED',)]
    finally:
        await runner.stop()
