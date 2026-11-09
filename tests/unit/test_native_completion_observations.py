"""Real callback admission and physical closure keep finite observer evidence local."""
import asyncio
import sys


from contextlib import AsyncExitStack

import pytest

from kestrel_sdk.resources import ResourceOutcomeUnknown
from kestrel_sovereign._async_ownership import OwnedCallbackScope, await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionCommitOutcomeError, execution_evidence_carriers, execution_native_source_error,
    is_execution_control_error, execution_observer_cancellations,
)
from kestrel_sovereign.llm.codex_app_server import CodexAppServerClient
from kestrel_sovereign.storage.privacy_wrapper import ReentrantTransitionLock
from tests.unit.test_tool_effect_execution import owner as owner
from tests.unit.test_tool_effect_invocation import governed


@pytest.mark.asyncio
@pytest.mark.skipif(sys.version_info < (3, 12), reason="native eager Task requires Python 3.12")
@pytest.mark.parametrize("kind", [None, "ordinary", "sdk", "audit"])
async def test_original_sqlite_read_shield_observes_admission_receipt(owner, monkeypatch, kind):
    from kestrel_sovereign.execution_custody import ToolAuditUnconfirmedError, _explicit_control_chain
    backend = owner.tool_effect_store.database.backend
    connection = backend._connection
    cursor = await connection.execute("SELECT 7 AS original_native_value")
    original_close, original_execute = cursor.close, connection.execute
    closes, results, owners = [], [], []
    publication = (ResourceOutcomeUnknown("original SQL read admission uncertain") if kind == "sdk"
        else ToolAuditUnconfirmedError("original SQL read audit uncertain") if kind == "audit"
        else RuntimeError("original SQL read callback publication refused"))
    async def close():
        await original_close()
        closes.append(cursor)
    def execute(*args, **kwargs):
        result = original_execute(*args, **kwargs)
        results.append(result)
        return result
    class OriginalTask(asyncio.Task):
        def add_done_callback(self, callback, *, context=None):
            if kind is not None and getattr(callback, "__name__", "") == "close_unstarted":
                raise publication
            return super().add_done_callback(callback, context=context)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    def factory(loop, coroutine, **options):
        frame = getattr(coroutine, "cr_frame", None)
        operation = frame.f_locals.get("operation") if frame is not None else None
        if not results or operation is not results[-1]:
            return previous(loop, coroutine, **options) if previous else asyncio.Task(coroutine, loop=loop, **options)
        async def outer():
            coroutine.close()
            return cursor
        options["eager_start"] = True
        task = OriginalTask(outer(), loop=loop, **options)
        owners.append(task)
        return task
    try:
        monkeypatch.setattr(cursor, "close", close)
        monkeypatch.setattr(connection, "execute", execute)
        loop.set_task_factory(factory)
        observed = None
        try:
            value = await backend._fetch_on_connection(connection, "SELECT 7 AS original_native_value", (), one=True)
        except BaseException as error:
            observed = error
        loop.set_task_factory(previous)
        if kind is None:
            assert observed is None and value == (7,)
        else:
            assert publication in tuple(_explicit_control_chain(observed)), (
                "Native SQL read shield bypassed original admission receipt")
        assert len(owners) == len(results) == 1 and owners[0].done()
        assert closes == [cursor]
        assert not backend._pending_read_executions
    finally:
        loop.set_task_factory(previous)
        for task in owners:
            await await_owned_task(task)
        if not closes:
            await original_close()

def native_training_component(*, release_session=None):
    from kestrel_sovereign.features.training.adapters._session_training_lifecycle import (
        SessionProviderHooks, SessionTrainingLifecycle,
    )
    async def unused_provider(*args, **kwargs):
        raise AssertionError("Native component must not call an external provider")
    return SessionTrainingLifecycle(SessionProviderHooks(
        provider_name="native-component", display_name="Native component",
        acquire_session=unused_provider, session_id=lambda session: str(session.name),
        submit_job=unused_provider,
        release_session=unused_provider if release_session is None else release_session,
    ))


@pytest.mark.asyncio
@pytest.mark.skipif(sys.version_info < (3, 12), reason="native eager Task requires Python 3.12")
@pytest.mark.parametrize("late", [False, True])
@pytest.mark.parametrize("kind", [None, "ordinary", "sdk", "audit"])
@pytest.mark.parametrize("refuse", [False, True])
async def test_original_bounded_native_owner_keeps_sdk_completion(tmp_path, kind, refuse, late):
    from kestrel_sovereign._async_ownership import _retain_owned_admission_error
    from kestrel_sovereign.execution_custody import ToolAuditUnconfirmedError, _explicit_control_chain
    lifecycle = native_training_component()
    physical = asyncio.CancelledError("original bounded native completion")
    if kind == "ordinary":
        physical = RuntimeError("original bounded physical failure")
    elif kind is not None:
        physical.__cause__ = (ResourceOutcomeUnknown("original bounded effect uncertain") if kind == "sdk"
            else ToolAuditUnconfirmedError("original bounded audit uncertain"))
    publication = RuntimeError("original bounded completion publication refused")
    effect = tmp_path / "original-bounded-native-effect"
    owners, bodies = [], []
    def physical_effect():
        effect.write_text("original bounded native effect completed once")
        if kind is not None:
            raise physical
        return "original bounded physical success"
    async def operation():
        bodies.append(asyncio.current_task())
        return physical_effect()
    class OriginalTask(asyncio.Task):
        def add_done_callback(self, callback, *, context=None):
            if refuse and not late and getattr(callback, "__name__", "") == "close_unstarted":
                raise publication
            return super().add_done_callback(callback, context=context)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    def publish(task):
        if refuse:
            _retain_owned_admission_error(task, publication)
    def factory(loop, coroutine, **options):
        async def outer():
            coroutine.close()
            return physical_effect()
        options["eager_start"] = True
        task = OriginalTask(coroutine if late else outer(), loop=loop, **options)
        owners.append(task)
        if late:
            task.add_done_callback(publish)
        return task
    try:
        loop.set_task_factory(factory)
        observed = None
        try:
            value = await lifecycle._bounded(operation())
        except BaseException as error:
            observed = error
        loop.set_task_factory(previous)
        chain = tuple(_explicit_control_chain(observed))
        if kind is not None:
            assert physical in chain, "Bounded native owner discarded original SDK or audit completion"
        if refuse:
            assert publication in chain
        if kind is None and not refuse:
            assert observed is None and value == "original bounded physical success"
        assert len(owners) == 1 and owners[0].done() and len(bodies) == int(late)
        assert effect.read_text() == "original bounded native effect completed once"
    finally:
        loop.set_task_factory(previous)
        for task in owners:
            await await_owned_task(task)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [None, "sdk", "audit"])
@pytest.mark.parametrize("cancellation", [False, True])
async def test_original_release_uncertainty_never_replays_file_resource(tmp_path, kind, cancellation):
    from datetime import datetime, timezone
    from kestrel_sovereign._async_ownership import run_blocking_operation
    from kestrel_sovereign.execution_custody import ToolAuditUnconfirmedError, _explicit_control_chain
    from kestrel_sovereign.features.training.adapters._session_training_lifecycle import (
        ReleaseIntent, SessionJobPhase, SessionTrainingRecord,
    )
    from kestrel_sovereign.features.training.types import TrainingConfig
    original = (ResourceOutcomeUnknown("original physical release uncertain") if kind == "sdk"
        else ToolAuditUnconfirmedError("original release audit uncertain"))
    if cancellation and kind is not None:
        carrier = asyncio.CancelledError("original physical release cancellation")
        carrier.__cause__ = original
        original = carrier
    resource = (tmp_path / "original-native-release-resource").open("w")
    calls = []
    async def release(session):
        await run_blocking_operation(session.close)
        calls.append(session)
        if kind is not None:
            raise original
    lifecycle = native_training_component(release_session=release)
    now = datetime.now(timezone.utc)
    record = SessionTrainingRecord(
        job_id="original-native-release", companion_id="native-component", trigger_word="native",
        config=TrainingConfig(), session=resource, created_at=now, started_at=now,
        phase=SessionJobPhase.TRAINING,
    )
    lifecycle.records[record.job_id] = record
    outcomes, errors, tasks = [], [], []
    try:
        for _ in range(2):
            error, value = None, None
            try:
                value = await lifecycle.release(record.job_id, ReleaseIntent.CLEANUP)
            except BaseException as observed:
                error = observed
            outcomes.append(value)
            errors.append(error)
            tasks.append(record.release_task)
        assert calls == [resource], "Uncertain original release was replayed on its actual resource"
        assert tasks[0] is tasks[1] and tasks[0].done()
        if kind is None:
            assert outcomes == [True, True] and errors == [None, None]
            assert record.session_released and record.phase is SessionJobPhase.RELEASED
        else:
            assert record.native_release_control is not None and not record.session_released
            assert lifecycle.records[record.job_id] is record
            assert record.phase is SessionJobPhase.RELEASE_FAILED
            assert all(original in tuple(_explicit_control_chain(error)) for error in errors)
        assert resource.closed
    finally:
        await run_blocking_operation(resource.close)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [None, "sdk", "audit"])
@pytest.mark.parametrize("withdraw", [False, True])
async def test_original_training_opening_receipt_releases_unpublished_file(tmp_path, monkeypatch, kind, withdraw):
    from kestrel_sovereign._async_ownership import _retain_owned_admission_error
    from kestrel_sovereign.execution_custody import ToolAuditUnconfirmedError
    from kestrel_sovereign.features.training.adapters._session_training_lifecycle import (
        SessionLifecycleTimeouts, SessionProviderHooks, SessionTrainingLifecycle,
    )
    from kestrel_sovereign.features.training.types import TrainingConfig
    import kestrel_sovereign.features.training.adapters._session_training_lifecycle as module
    original = (ResourceOutcomeUnknown("original opening publication uncertain") if kind == "sdk"
        else ToolAuditUnconfirmedError("original opening audit uncertain"))
    resources, released, openings = [], [], []
    submitted = asyncio.Event()
    async def acquire(*args):
        resource = (tmp_path / "original-training-session").open("w")
        resource.write("original acquired session")
        resources.append(resource)
        return resource
    async def submit(*args):
        submitted.set()
        await asyncio.Event().wait()
    async def release(resource):
        resource.close()
        released.append(resource)
    lifecycle = SessionTrainingLifecycle(SessionProviderHooks(
        provider_name="native-component", display_name="Native component",
        acquire_session=acquire, session_id=lambda resource: resource.name,
        submit_job=submit, release_session=release,
    ), SessionLifecycleTimeouts(submission_drain=.5, release=.5, close=2))
    create = module.create_owned_task
    caller = asyncio.current_task()
    def publish(task):
        if kind is not None:
            _retain_owned_admission_error(task, original)
        if withdraw:
            caller.cancel("original opening observer withdrew")
    def create_with_publication(operation, **options):
        task = create(operation, **options)
        if options.get("name", "").startswith("native-component-training-open:"):
            openings.append(task)
            task.add_done_callback(publish)
        return task
    monkeypatch.setattr(module, "create_owned_task", create_with_publication)
    observed = None
    try:
        try:
            record = await lifecycle.start(job_id="original-opening", companion_id="native",
                trigger_word="native", config=TrainingConfig(), avatar_data=b"original")
        except BaseException as error:
            observed = error
        assert len(resources) == len(openings) == 1 and openings[0].done()
        if kind is not None or withdraw:
            assert observed is not None, "Successful acquisition discarded its original completion receipt"
            if kind is not None:
                assert original in tuple(execution_evidence_carriers(observed))
            assert released == resources, "Rejected original acquisition lost its physically successful resource"
            assert resources[0].closed and not lifecycle.records
        else:
            assert observed is None and record.session is resources[0] and not resources[0].closed
    finally:
        await lifecycle.close()
        for resource in resources:
            resource.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [None, "sdk", "audit"])
async def test_original_bounded_deadline_preserves_source_exception_graph(tmp_path, kind):
    from kestrel_sovereign.execution_custody import ToolAuditUnconfirmedError
    from kestrel_sovereign.features.training.adapters._session_training_lifecycle import SessionLifecycleTimeouts
    lifecycle = native_training_component()
    lifecycle._timeouts = SessionLifecycleTimeouts(submission_drain=.5, release=.02, close=2)
    source = asyncio.CancelledError("original deadline cancellation")
    cause = (ResourceOutcomeUnknown("original deadline effect uncertain") if kind == "sdk"
        else ToolAuditUnconfirmedError("original deadline audit uncertain"))
    source.__cause__ = cause if kind is not None else None
    cause.__cause__ = ValueError("original provider diagnostic")
    source_cause, cause_cause = source.__cause__, cause.__cause__
    effect = tmp_path / "original-deadline-effect"
    async def operation():
        effect.write_text("one original physical operation")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise source
    observed = None
    try:
        await lifecycle._bounded(operation())
    except BaseException as error:
        observed = error
    assert observed is not None and effect.read_text() == "one original physical operation"
    assert source.__cause__ is source_cause, "Deadline observation mutated the original source cause"
    assert cause.__cause__ is cause_cause
    chain = tuple(execution_evidence_carriers(observed))
    assert any(isinstance(error, TimeoutError) for error in chain)
    if kind is not None:
        assert source in chain and cause in chain


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [None, "sdk", "audit"])
@pytest.mark.parametrize("provider", ["vastai", "gcp", "runpod"])
async def test_original_training_public_status_keeps_terminal_delivery(tmp_path, provider, kind):
    from datetime import datetime, timezone
    from kestrel_sovereign.execution_custody import ToolAuditUnconfirmedError, is_tool_execution_terminal_error
    from kestrel_sovereign.features.training.adapters._session_training_lifecycle import SessionTrainingRecord, SessionJobPhase
    from kestrel_sovereign.features.training.adapters.vastai_adapter import VastAITrainingAdapter
    from kestrel_sovereign.features.training.adapters.gcp_compute_adapter import GCPComputeTrainingAdapter
    from kestrel_sovereign.features.training.adapters.runpod_adapter import RunPodTrainingAdapter
    from kestrel_sovereign.features.training.types import TrainingConfig, TrainingState
    adapter = {"vastai": VastAITrainingAdapter, "gcp": GCPComputeTrainingAdapter,
        "runpod": RunPodTrainingAdapter}[provider](manager=object())
    original = (ResourceOutcomeUnknown("original submission uncertain") if kind == "sdk"
        else ToolAuditUnconfirmedError("original submission audit uncertain"))
    resource = (tmp_path / "original-status-resource").open("w")
    resource.instance_id = 1
    resource.instance_name = "original-status-resource"
    resource.pod_id = "original-status-resource"
    now = datetime.now(timezone.utc)
    record = SessionTrainingRecord(job_id="original-status", companion_id="native", trigger_word="native",
        config=TrainingConfig(), session=resource, created_at=now, started_at=now,
        phase=SessionJobPhase.FAILED)
    if kind is not None:
        record.task_opening_errors["submission_task"] = original
    adapter._lifecycle.records[record.job_id] = record
    observed, result = None, None
    try:
        try:
            result = await adapter.get_status(record.job_id)
        except BaseException as error:
            observed = error
        if kind is None:
            assert observed is None and result.state is TrainingState.FAILED
        else:
            assert is_tool_execution_terminal_error(observed), "Public training status erased original terminal classification"
            assert original in tuple(execution_evidence_carriers(observed))
        assert adapter._lifecycle.records[record.job_id] is record
    finally:
        resource.close()


def sqlite_commit_ack_loss(connection, table=None, *, after, ready=None):
    import aiosqlite
    assert (table is None) != (ready is None), 'select one original commit readiness source'
    native_commit = aiosqlite.Connection.commit
    original, lost = OSError('original prebody SQLite COMMIT acknowledgment lost'), []

    async def lose_ack(target):
        if target is connection and not lost:
            if ready is not None:
                present = ready()
            else:
                async with target.execute('SELECT COUNT(*) FROM ' + table) as cursor:
                    present = (await cursor.fetchone())[0]
            if present:
                lost.append(True)
                if after:
                    await native_commit(target)
                raise original
        await native_commit(target)

    return original, lost, lose_ack


@pytest.mark.asyncio
@pytest.mark.parametrize('phase,loss', [
    ('claim', 'before'), ('claim', 'after'), ('audit', 'before'), ('audit', 'after'),
    ('claim', 'known-refusal'),
])
async def test_original_prebody_commit_failure_fences_live_effects(owner, monkeypatch, tmp_path, phase, loss):
    import aiosqlite
    from kestrel_sovereign.agent.orchestrator_engine import _run_tool_execution
    from kestrel_sovereign.agent.tool_registry import _ToolExecution
    from kestrel_sovereign.execution_custody import current_execution_custody, execution_commit_outcome
    from kestrel_sovereign.tools.effect_journal import current_tool_effect_turn, execute_tool_effect

    database = owner.tool_effect_store.database
    connection = database.backend._connection
    seen = {}
    witness = tmp_path / 'live-effect-after-prebody-commit-loss'
    table = 'tool_effect_claims' if phase == 'claim' else 'a2a_tool_dispatches'

    original, lost, lose_ack = sqlite_commit_ack_loss(connection, table, after=loss == 'after')

    async def first_effect():
        witness.write_text('first unadmitted effect ran')

    async def later_effect():
        witness.write_text('later effect admitted after original claim/audit uncertainty')
        return 'confirmed later effect'

    async def propagate(error, *args):
        raise error

    @governed
    async def invoke(agent, user_input, *, invocation_id=None):
        assert not current_execution_custody(agent), 'probe requires the supported standalone SQLite host'
        if loss != 'known-refusal':
            monkeypatch.setattr(aiosqlite.Connection, 'commit', lose_ack)
        try:
            if phase == 'claim':
                await execute_tool_effect(agent, registration='original-uncertain-claim',
                    arguments=({'invalid_json': object()} if loss == 'known-refusal' else {}),
                    execute=first_effect)
            else:
                await _run_tool_execution(agent, lambda: write_required_denial_audit(agent),
                    execution=_ToolExecution(), dispatch_meta={}, session_id='original-native-audit',
                    on_failure=propagate, on_terminal=propagate)
        except BaseException as error:
            seen['original_error'] = error
        seen['unconfirmed'] = tuple(current_tool_effect_turn(agent).unconfirmed)
        try:
            seen['later_result'] = await execute_tool_effect(agent,
                registration='later-live-effect', arguments={}, execute=later_effect)
        except BaseException as error:
            seen['later_error'] = error

    try:
        await invoke(owner, 'original prebody commit uncertainty', invocation_id='original-prebody-commit')
    except BaseException as error:
        seen['root_error'] = error
    if loss == 'known-refusal':
        assert not seen['unconfirmed'] and 'root_error' not in seen and 'later_error' not in seen
        assert seen['later_result'] == 'confirmed later effect'
        assert await database.fetchone('SELECT state,latched FROM tool_effect_operations') == ('CLOSED', 0)
        return
    assert not witness.exists(), 'prebody native COMMIT uncertainty admitted another live effect'
    assert lost == [True] and execution_commit_outcome(seen['original_error']) == 'unknown'
    assert any(error is original for error in execution_evidence_carriers(seen['original_error']))
    assert seen['unconfirmed'] and 'later_error' in seen and 'root_error' in seen
    assert any(error is original for error in execution_evidence_carriers(seen['root_error']))
    assert original.__cause__ is None
    assert await database.fetchone('SELECT state,latched FROM tool_effect_operations') == ('CLOSED', 1)
    assert await database.fetchone('SELECT COUNT(*) FROM ' + table) == (int(loss == 'after'),)
    assert await database.fetchone('SELECT COUNT(*) FROM tool_effect_claims') == (int(phase == 'claim' and loss == 'after'),)


def native_failure(kind):
    error = (asyncio.CancelledError('original source cancellation') if kind == 'cancel' else
        ResourceOutcomeUnknown('ordinary SDK outcome uncertainty') if kind == 'ordinary' else
        ExecutionCommitOutcomeError('unknown') if kind == 'unknown' else
        ExecutionCommitOutcomeError('committed') if kind == 'committed' else
        BaseExceptionGroup('original native source group',
            [ExecutionCommitOutcomeError('unknown'), asyncio.CancelledError('source withdrawal')]))
    cause = ValueError('original physical source cause')
    error.__cause__ = cause
    return error, cause


def assert_original_graph(error, cause, observed, marker):
    assert any(item is error for item in execution_evidence_carriers(observed))
    assert any(isinstance(item, asyncio.CancelledError) and item.args == (marker,)
        for item in execution_evidence_carriers(observed)), 'finite observation lost its own withdrawal'
    assert error.__cause__ is cause, 'observer withdrawal rewrote the original shared source graph'
    native = execution_native_source_error(observed)
    if is_execution_control_error(error):
        assert any(item is error for item in execution_evidence_carriers(native))
    else:
        assert native is None
    assert not any(isinstance(item, asyncio.CancelledError) and item.args == (marker,)
        for item in execution_evidence_carriers(native)), 'observer withdrawal escaped into native source projection'
    client = CodexAppServerClient(binary='unused-offline-codex')
    client._retain_request_failure(observed)
    shared = getattr(client, '_server_request_failure', None)
    assert not any(isinstance(item, asyncio.CancelledError) and item.args == (marker,)
        for item in execution_evidence_carriers(shared)), 'shared Codex connection retained a foreign observer withdrawal'
    if native is not None:
        assert any(item is error for item in execution_evidence_carriers(shared))


@pytest.mark.asyncio
@pytest.mark.parametrize('mode,kind', [(mode,kind)
    for mode in ['park', 'yield', 'scope-close', 'join']
    for kind in ['cancel', 'ordinary', 'unknown', 'committed', 'group']
    if mode != 'join' or kind not in {'cancel', 'ordinary'}])
async def test_original_source_graph_survives_cancelled_writer_restoration(mode, kind, tmp_path):
    lock = ReentrantTransitionLock()
    entered, release, body_failed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original, cause = native_failure(kind)
    receipt = tmp_path / 'original-writer-settlement'
    children = []
    marker = 'observer withdrew during original writer restoration'

    async def child():
        entered.set()
        await release.wait()
        receipt.write_text('physical writer settled')

    error_release = asyncio.Event()

    async def source():
        children.append(lock.start_owned_native_work(child))
        await entered.wait()
        await error_release.wait()
        body_failed.set()
        raise original

    async def fail_body():
        await entered.wait()
        body_failed.set()
        raise original

    async def operation():
        async with lock.policy_span() if mode in {'scope-close', 'join'} else lock:
            if mode == 'join':
                children.append(lock.start_owned_native_work(source, acquire_span=False))
                await entered.wait()
                error_release.set()
                await lock.join_owned_native_work()
            else:
                task = lock.start_owned_native_work(child)
                children.append(task)
                if mode == 'scope-close':
                    await fail_body()
                else:
                    span = lock.park_native_wait((task,)) if mode == 'park' else lock.yield_native_writer()
                    async with span:
                        await fail_body()

    calling = create_owned_task(operation())
    try:
        await asyncio.wait_for(body_failed.wait(), .5)
        await asyncio.sleep(0)
        assert lock._writer in children and not calling.done()
        calling.cancel(marker)
        await asyncio.sleep(0)
        assert not calling.done(), 'observer escaped before restoring the original writer'
    finally:
        error_release.set()
        release.set()
        outcome = await await_owned_task(calling)
        for task in children:
            await await_owned_task(task)
    assert_original_graph(original, cause, outcome.error, marker)
    if kind == 'cancel':
        observers = execution_observer_cancellations(outcome.error)
        assert sum(error.args == original.args for error in observers) == 1, (
            'finite cancellation envelope counted as another actual withdrawal')
        assert sum(error.args == (marker,) for error in observers) == 1
    assert receipt.read_text() == 'physical writer settled'
    assert not lock.locked() and not lock._native_scopes


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['unknown', 'committed'])
async def test_original_journal_error_does_not_absorb_close_observer_withdrawal(owner, kind, tmp_path):
    from kestrel_sovereign.tools.effect_journal import execute_tool_effect
    entered, release, body_failed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original, cause = native_failure(kind)
    marker = 'observer withdrew while original journal closed'
    receipt = tmp_path / 'original-journal-effect'
    children = []

    @governed
    async def invoke(agent, user_input, *, invocation_id=None):
        async def effect():
            entered.set()
            await release.wait()
            receipt.write_text('original journal effect acknowledged')
            return 'acknowledged'
        children.append(create_owned_task(execute_tool_effect(agent,
            registration='original-closure-effect', arguments={}, execute=effect)))
        await entered.wait()
        body_failed.set()
        raise original

    calling = create_owned_task(invoke(owner, 'one physical effect', invocation_id='original-closure-observer'))
    try:
        await asyncio.wait_for(body_failed.wait(), 1)
        await asyncio.sleep(0)
        calling.cancel(marker)
        await asyncio.sleep(0)
        assert not calling.done(), 'journal closure abandoned its original claimant'
    finally:
        release.set()
        outcome = await await_owned_task(calling)
        for task in children:
            await await_owned_task(task)
    assert_original_graph(original, cause, outcome.error, marker)
    assert receipt.read_text() == 'original journal effect acknowledged'
    assert await owner.tool_effect_store.database.fetchall('SELECT state FROM tool_effect_claims') == [('RETURNED_OK',)]
    assert await owner.tool_effect_store.database.fetchall('SELECT state FROM tool_effect_operations') == [('CLOSED',)]


@pytest.mark.asyncio
@pytest.mark.parametrize('failure,privacy', [(failure, privacy)
    for failure in ['start-gate', 'accepted-factory', 'completion', 'metadata', 'cohort', 'active', 'both-subscriptions', 'start-gate-before-step', 'both-subscriptions-before-step']
    for privacy in [False, True] if failure != 'completion' or privacy])
async def test_failed_native_codex_callback_notifies_only_its_original_turn(privacy, failure, tmp_path):
    stop_before_step = failure.endswith('-before-step')
    failure = failure.removesuffix('-before-step')
    loop, prior = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    original = RuntimeError('original native callback opening failure')
    allocated = []
    receipt = tmp_path / 'unadmitted-codex-effect'
    client = CodexAppServerClient(binary='unused-offline-codex')
    own = client.open_turn_sink('original-thread')
    foreign = client.open_turn_sink('foreign-thread')
    lock = ReentrantTransitionLock()
    if failure == 'metadata':
        class RefusedRequestMetadata(dict):
            def __setitem__(self, task, completion):
                super().__setitem__(task, completion)
                raise original
        client._server_request_tasks = RefusedRequestMetadata()

    class FailedStartTask(asyncio.Task):
        def add_done_callback(self, callback, **kwargs):
            if (failure in ('start-gate', 'both-subscriptions') and getattr(callback, '__name__', '') == 'close_unstarted'
                    or failure in ('completion', 'both-subscriptions') and (
                        getattr(callback, '__name__', '') == 'completed'
                        or any(name in (getattr(getattr(callback, '__code__', None), 'co_names', ())
                            + getattr(getattr(callback, '__code__', None), 'co_freevars', ()))
                            for name in ('_server_request_done', 'report_completion')))):
                raise original
            return super().add_done_callback(callback, **kwargs)

    def refuse_after_acceptance(loop, coroutine, **kwargs):
        task = (FailedStartTask if failure in ('start-gate', 'completion', 'metadata', 'both-subscriptions') else asyncio.Task)(coroutine, loop=loop, **kwargs)
        allocated.append(task)
        if failure == 'accepted-factory':
            raise original
        return task

    async def handler(params):
        receipt.write_text('unadmitted callback dispatched')
        return {'success': True}

    async with AsyncExitStack() as resources:
        if privacy:
            await resources.enter_async_context(lock.policy_span())
        scope = OwnedCallbackScope()
        handler._codex_native_callback_scope = scope
        class RefusedInventory(set):
            def add(self, task):
                super().add(task)
                raise original
        handler._codex_active_tasks = RefusedInventory() if failure == 'active' else set()
        if failure == 'cohort':
            accept = scope.accept
            def refuse_cohort(task):
                accept(task)
                raise original
            scope.accept = refuse_cohort
        unregister = client.register_server_request_handler('item/tool/call', handler, thread_id='original-thread')
        try:
            loop.set_task_factory(refuse_after_acceptance)
            escaped = None
            try:
                client._dispatch({'id': 41, 'method': 'item/tool/call', 'params': {'threadId': 'original-thread'}})
            except BaseException as error:
                escaped = error
            finally:
                loop.set_task_factory(prior)
            if escaped is not None:
                # A broken producer can strand this test's exact accepted gate.
                # Dispose only these owned Tasks before asserting the escape;
                # a pytest timeout is not evidence that the guard caught it.
                for task in tuple(scope.tasks):
                    task.cancel('dispose fixture-owned unadmitted callback')
                    await await_owned_task(task)
            assert escaped is None, 'native callback publication escaped resident dispatch'
            if stop_before_step:
                await client._join_server_requests()
                assert all(task.cancelled() for task in allocated)
            for task in tuple(scope.tasks):
                await await_owned_task(task)
            await asyncio.sleep(0)
            assert not own.empty(), 'failed native callback admission left its original Codex turn unanswered'
            message = own.get_nowait()
            assert '__bridge_error__' in message
            assert any(error is original for error in execution_evidence_carriers(message['__bridge_error__']))
            assert foreign.empty() and client._closed_error is None
            client._dispatch({'method': 'turn/completed', 'params': {'threadId': 'foreign-thread'}})
            assert foreign.get_nowait()['method'] == 'turn/completed'
            assert not receipt.exists() and len(allocated) == 1 and all(task.done() for task in allocated)
            assert not client._server_request_tasks and not handler._codex_active_tasks
        finally:
            loop.set_task_factory(prior)
            scope.close()
            unregister()
            for task in tuple(scope.tasks):
                await await_owned_task(task)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["sdk", "audit"])
@pytest.mark.parametrize("composition", ["observation", "terminal"])
async def test_sdk_unknown_envelope_keeps_first_message_and_immutable_originals(kind, composition):
    from kestrel_sovereign.execution_custody import (
        ToolAuditUnconfirmedError, execution_observation_error, execution_terminal_error,
    )
    error_type = ToolAuditUnconfirmedError if kind == "audit" else ResourceOutcomeUnknown
    first = error_type('failed task could not be recorded')
    second = error_type('provider reply lost')
    first_cause, second_cause = ValueError('original task write'), ValueError('original provider')
    first.__cause__, second.__cause__ = first_cause, second_cause
    observed = (execution_observation_error(first, second) if composition == "observation" else
        execution_terminal_error(first, second, retain_originals=True, isolate_control_originals=True))
    assert type(observed) is error_type, 'finite join erased public SDK outcome type'
    assert str(observed) == str(first)
    assert observed is not first and observed is not second
    assert first.__cause__ is first_cause and second.__cause__ is second_cause
    assert all(any(node is original for node in execution_evidence_carriers(observed))
        for original in (first, second, first_cause, second_cause))


@pytest.mark.asyncio
@pytest.mark.parametrize('interrupted', [False, True])
async def test_no_claim_uncertainty_fences_reopened_operation(owner, interrupted):
    from kestrel_sovereign.a2a.scope import NativeRecipientScope
    from kestrel_sovereign.tool_effect_store import ToolEffectReconciliationRequired, ToolEffectStore
    from kestrel_sovereign.tools.effect_journal import (
        bind_tool_effect_turn, record_tool_effect_completion_failure,
    )
    from tests.unit.test_tool_effect_execution import admit
    turn = await admit(owner, key='original-preclaim-uncertainty')
    store, db = owner.tool_effect_store, owner.tool_effect_store.database
    original = ResourceOutcomeUnknown('required audit ACK unavailable before business claim')
    with bind_tool_effect_turn((turn, None)):
        record_tool_effect_completion_failure(owner, original)
    if not interrupted:
        with pytest.raises(ResourceOutcomeUnknown):
            await turn.close()
    else:
        # Model an interrupted process: no terminal write. Release only local
        # privacy resources so the fixture can dispose this in-process probe.
        turn.live = False
        await turn.privacy_lock.close_native_scope(turn.privacy_scope, None)
    assert await db.fetchone('SELECT COUNT(*) FROM tool_effect_claims') == (0,)
    expected = ('OPEN', 0) if interrupted else ('CLOSED', 1)
    assert await db.fetchone('SELECT state,latched FROM tool_effect_operations') == expected, (
        'Original completion uncertainty was not durably recorded')
    await store.close()
    await db.backend.close()
    await db.backend.connect()
    restarted = ToolEffectStore(db, scope=NativeRecipientScope(owner.did))
    await restarted.initialize()
    owner.tool_effect_store = restarted
    with pytest.raises(ToolEffectReconciliationRequired) as refused:
        await admit(owner, key='original-preclaim-uncertainty')
    assert all(count == 0 for count in refused.value.counts.values()), 'audit uncertainty fabricated a business claim'
    assert await db.fetchone('SELECT COUNT(*) FROM tool_effect_claims') == (0,)
    fresh = await admit(owner, key='independent-new-operation')
    await fresh.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('borrowed', [False, True])
async def test_original_preclaim_audit_settles_before_operation_seal(owner, monkeypatch, borrowed):
    from kestrel_sovereign.tool_effect_store import ToolEffectReconciliationRequired
    from kestrel_sovereign.tools.effect_journal import (
        bind_tool_effect_turn, record_tool_effect_completion_failure,
    )
    from tests.unit.test_tool_effect_execution import admit

    loop, prior = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    entered, release, sealing = asyncio.Event(), asyncio.Event(), asyncio.Event()
    joined = loop.create_future()
    original_worker = []
    original = ResourceOutcomeUnknown('original required audit delivery unavailable')
    store, db = owner.tool_effect_store, owner.tool_effect_store.database
    finish = store.finish

    class OriginalAuditTask(asyncio.Task):
        def add_done_callback(self, callback, **kwargs):
            if (original_worker and self is original_worker[0]
                    and getattr(callback, '__name__', '') == 'mark_complete' and not joined.done()):
                joined.set_result(None)
            return super().add_done_callback(callback, **kwargs)

    def allocate(loop, coroutine, **kwargs):
        return OriginalAuditTask(coroutine, loop=loop, **kwargs)

    async def record_seal(*args, **kwargs):
        sealing.set()
        return await finish(*args, **kwargs)

    monkeypatch.setattr(store, 'finish', record_seal)
    async with AsyncExitStack() as resources:
        if borrowed:
            await resources.enter_async_context(owner._get_privacy_transition_lock().policy_span())
        turn = await admit(owner, key='original-pending-preclaim-audit')
        assert turn.owns_privacy_scope is not borrowed

        async def audit():
            entered.set()
            await release.wait()
            await write_required_denial_audit(owner)
            raise record_tool_effect_completion_failure(owner, original)

        with bind_tool_effect_turn((turn, None)):
            try:
                loop.set_task_factory(allocate)
                worker = turn.privacy_lock.start_owned_native_work(audit, acquire_span=False)
                original_worker.append(worker)
            finally:
                loop.set_task_factory(prior)
            await entered.wait()
            closing = create_owned_task(turn.close(), retain_originals=True)
        try:
            # Observe the real original Task's native join, or an erroneous
            # early Root return. Neither outcome requires a synthetic collector.
            done, _ = await asyncio.wait((closing, joined), timeout=1,
                return_when=asyncio.FIRST_COMPLETED)
            made_progress = bool(done)
            sealed_while_pending = sealing.is_set()
        finally:
            release.set()
            outcome = await await_owned_task(closing)
            worker_outcome = await await_owned_task(worker)
        assert made_progress, 'original audit closure made no progress'
    assert not sealed_while_pending, 'journal seal preceded its accepted preclaim audit'
    assert any(error is original for error in execution_evidence_carriers(worker_outcome.error))
    assert any(error is original for error in execution_evidence_carriers(outcome.error)), 'Root discarded its late original audit uncertainty'
    assert await db.fetchone('SELECT COUNT(*) FROM a2a_tool_dispatches') == (1,)
    assert await db.fetchone('SELECT COUNT(*) FROM tool_effect_claims') == (0,)
    assert await db.fetchone('SELECT state,latched FROM tool_effect_operations') == ('CLOSED', 1)
    with pytest.raises(ToolEffectReconciliationRequired) as refused:
        await admit(owner, key='original-pending-preclaim-audit')
    assert all(count == 0 for count in refused.value.counts.values())
    assert worker.done() and closing.done()


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['unknown', 'committed'])
async def test_stream_journal_close_keeps_original_native_cause_immutable(owner, kind):
    from tests.unit.test_tool_effect_invocation import governed_stream
    original, cause = native_failure(kind)

    @governed_stream
    async def invoke(agent, user_input, *, request_id=None):
        try:
            yield 'original stream item'
        finally:
            raise original

    stream = invoke(owner, 'original streaming input', request_id='original-stream-native-closer')
    assert await anext(stream) == 'original stream item'
    observed = None
    try:
        await stream.aclose()
    except BaseException as error:
        observed = error
    assert observed is not None
    assert original.__cause__ is cause, 'stream journal teardown rewrote its original native source cause'
    assert any(error is original for error in execution_evidence_carriers(observed))
    assert any(isinstance(error, GeneratorExit) for error in execution_evidence_carriers(observed))
    assert await owner.tool_effect_store.database.fetchone(
        'SELECT state FROM tool_effect_operations') == ('CLOSED',)


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['success', 'ordinary', 'unknown', 'committed', 'group'])
async def test_original_task_completion_publisher_failure_preserves_all_sources(tmp_path, kind):
    original, cause = (None, None) if kind == 'success' else native_failure(kind)
    publication, publication_cause = OSError('original completion publisher failed'), ValueError('original publication cause')
    publication.__cause__ = publication_cause
    receipt = tmp_path / 'physically-settled-original'
    observed = []

    async def body():
        receipt.write_text('one original physical operation')
        if original is not None:
            raise original
        return 'acknowledged original result'

    def publish(task, error):
        observed.append((task, error))
        raise publication

    task = create_owned_task(body(), retain_originals=True, on_completion=publish)
    outcome = await await_owned_task(task)
    assert task.done() and len(observed) == 1 and observed[0][0] is task
    assert receipt.read_text() == 'one original physical operation'
    assert any(error is publication for error in execution_evidence_carriers(outcome.error))
    assert publication.__cause__ is publication_cause
    if original is not None:
        assert any(error is original for error in execution_evidence_carriers(outcome.error))
        assert any(error is original for error in execution_evidence_carriers(observed[0][1]))
        assert original.__cause__ is cause
    else:
        assert observed[0][1] is None


@pytest.mark.asyncio
@pytest.mark.parametrize('declared_async', [False, True])
async def test_native_completion_publisher_must_finish_synchronously(tmp_path, declared_async):
    receipt, observer_receipt = tmp_path / 'body', tmp_path / 'async-observer'
    async def body():
        receipt.write_text('body settled')
    async def asynchronous_publication(task, error):
        observer_receipt.write_text('invalid deferred completion')
    def concealed_publication(task, error):
        return asynchronous_publication(task, error)
    task, refused = None, None
    try:
        task = create_owned_task(body(), on_completion=(asynchronous_publication
            if declared_async else concealed_publication))
        outcome = await await_owned_task(task)
        refused = outcome.error
    except TypeError as error:
        refused = error
    assert isinstance(refused, TypeError), 'native completion silently accepted an awaitable publisher'
    assert not observer_receipt.exists()
    assert receipt.exists() is not declared_async
    if task is not None:
        assert task.done()


@pytest.mark.asyncio
@pytest.mark.parametrize('form', ['async-callable', 'async-generator', 'async-generator-callable'])
async def test_declared_deferred_publisher_is_refused_before_native_allocation(tmp_path, form):
    body_receipt = tmp_path / 'body'
    allocated = []
    loop, prior = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()

    async def body():
        body_receipt.write_text('body ran before declared async publisher refusal')

    async def generator(task, error):
        yield None

    class DeferredCallable:
        async def __call__(self, task, error):
            return None

    class GeneratorCallable:
        async def __call__(self, task, error):
            yield None

    publisher = {'async-callable': DeferredCallable(), 'async-generator': generator,
        'async-generator-callable': GeneratorCallable()}[form]

    def allocate(loop, coroutine, **kwargs):
        task = asyncio.Task(coroutine, loop=loop, **kwargs)
        allocated.append(task)
        return task

    operation, task, refused = body(), None, None
    try:
        loop.set_task_factory(allocate)
        try:
            task = create_owned_task(operation, on_completion=publisher)
        except TypeError as error:
            refused = error
    finally:
        loop.set_task_factory(prior)
        for accepted in allocated:
            await await_owned_task(accepted)
    assert not allocated, 'declared deferred publisher was accepted by the native Task constructor'
    assert isinstance(refused, TypeError) and task is None
    assert operation.cr_frame is None and not body_receipt.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize('borrowed', [False, True])
@pytest.mark.parametrize('kind', ['unknown', 'committed', 'group'])
async def test_completed_original_native_failure_is_harvested_before_journal_seal(owner, borrowed, kind):
    from kestrel_sovereign.tool_effect_store import ToolEffectReconciliationRequired
    from kestrel_sovereign.tools.effect_journal import bind_tool_effect_turn
    from tests.unit.test_tool_effect_execution import admit
    original, cause = native_failure(kind)
    async with AsyncExitStack() as resources:
        if borrowed:
            await resources.enter_async_context(owner._get_privacy_transition_lock().policy_span())
        turn = await admit(owner, key='completed-original-native-failure')

        async def audit():
            await write_required_denial_audit(owner)
            raise original

        with bind_tool_effect_turn((turn, None)):
            worker = turn.privacy_lock.start_owned_native_work(audit, acquire_span=False)
            outcome = await await_owned_task(worker)
        await asyncio.sleep(0)
        assert worker.done() and worker not in turn.privacy_lock._native_work
        assert not turn.pending and not turn.unconfirmed
        assert any(error is original for error in execution_evidence_carriers(outcome.error))
        terminal = None
        try:
            await turn.close()
        except BaseException as error:
            terminal = error
        # A borrowed lifetime must itself retain its genuine native failure;
        # dispose it here before making assertions about the journal boundary.
        if borrowed:
            try:
                await resources.aclose()
            except BaseException as error:
                assert any(node is original for node in execution_evidence_carriers(error))
    assert any(error is original for error in execution_evidence_carriers(terminal)), (
        'journal retirement discarded an already completed original native failure')
    assert original.__cause__ is cause
    db = owner.tool_effect_store.database
    assert await db.fetchone('SELECT COUNT(*) FROM a2a_tool_dispatches') == (1,)
    assert await db.fetchone('SELECT COUNT(*) FROM tool_effect_claims') == (0,)
    assert await db.fetchone('SELECT state,latched FROM tool_effect_operations') == ('CLOSED', 1)
    with pytest.raises(ToolEffectReconciliationRequired):
        await admit(owner, key='completed-original-native-failure')


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['ordinary', 'unknown', 'committed'])
async def test_original_uncertain_root_terminal_fences_claim_free_replay(owner, kind):
    from kestrel_sovereign.tool_effect_store import ToolEffectReconciliationRequired
    from tests.unit.test_tool_effect_execution import admit
    turn = await admit(owner, key='original-root-finalization-uncertainty')
    original, cause = native_failure(kind)
    await write_required_denial_audit(owner)
    assert not turn.unconfirmed
    observed = None
    try:
        await turn.close(original)
    except BaseException as error:
        observed = error
    assert any(error is original for error in execution_evidence_carriers(observed))
    assert original.__cause__ is cause
    assert await owner.tool_effect_store.database.fetchone('SELECT COUNT(*) FROM a2a_tool_dispatches') == (1,)
    assert await owner.tool_effect_store.database.fetchone('SELECT COUNT(*) FROM tool_effect_claims') == (0,)
    assert await owner.tool_effect_store.database.fetchone(
        'SELECT state,latched FROM tool_effect_operations') == ('CLOSED', 1), (
        'Original root terminal uncertainty was not durably recorded')
    with pytest.raises(ToolEffectReconciliationRequired):
        await admit(owner, key='original-root-finalization-uncertainty')


async def write_required_denial_audit(owner):
    from kestrel_sovereign.a2a.stores.unified.observability_store import ToolDispatchEntry
    await owner.observability_store.log_tool_dispatch(ToolDispatchEntry(
        agent_did=owner.did, session_id='original-audit-session', turn_id='original-audit-turn',
        tool_name='refused-native-tool', adapter='native-audit', args_redacted={},
        result_status='policy_denied', error_class='PolicyDenied', error_message=None,
        latency_ms=0, result_size_bytes=0), required=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('native', [False, True])
async def test_known_claim_free_refusal_retains_classification_and_requires_fresh_intent(owner, native):
    from kestrel_sdk.resources import ResourceConflict
    from kestrel_sovereign.execution_custody import ExecutionAuthorityError
    from kestrel_sovereign.tool_effect_store import ToolEffectReconciliationRequired
    from tests.unit.test_tool_effect_execution import admit
    turn = await admit(owner, key='known-safe-claim-free-refusal')
    original = (ExecutionAuthorityError if native else ResourceConflict)('known admission refusal')
    await write_required_denial_audit(owner)
    with pytest.raises(type(original)) as caught:
        await turn.close(original)
    assert caught.value is original
    assert await owner.tool_effect_store.database.fetchone('SELECT state,latched FROM tool_effect_operations') == ('CLOSED', 0)
    assert await owner.tool_effect_store.database.fetchone('SELECT COUNT(*) FROM tool_effect_claims') == (0,)
    with pytest.raises(ToolEffectReconciliationRequired):
        await admit(owner, key='known-safe-claim-free-refusal')
    successor = await admit(owner, key='fresh-explicit-intent-after-known-refusal')
    await successor.close()


@pytest.mark.asyncio
async def test_journal_closure_joins_its_original_subtree_without_waiting_for_foreign_work(owner, tmp_path):
    from kestrel_sovereign.tools.effect_journal import bind_tool_effect_turn
    from tests.unit.test_tool_effect_execution import admit
    lock = owner._get_privacy_transition_lock()
    loop, prior = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    foreign_join = loop.create_future()
    release_foreign, foreign_entered = asyncio.Event(), asyncio.Event()
    foreign_owner = []
    witness = tmp_path / 'unrelated-native-operation'

    class ForeignNativeTask(asyncio.Task):
        def add_done_callback(self, callback, **kwargs):
            if (foreign_owner and self is foreign_owner[0]
                    and getattr(callback, '__name__', '') == 'mark_complete' and not foreign_join.done()):
                foreign_join.set_result(None)
            return super().add_done_callback(callback, **kwargs)

    def allocate(loop, coroutine, **kwargs):
        return ForeignNativeTask(coroutine, loop=loop, **kwargs)

    async def unrelated():
        foreign_entered.set()
        await release_foreign.wait()
        witness.write_text('original unrelated native operation settled')

    async def journal():
        turn = await admit(owner, key='original-subtree-journal')
        assert not turn.owns_privacy_scope and turn.native_root is asyncio.current_task()
        assert turn.privacy_scope.root is not turn.native_root
        with bind_tool_effect_turn((turn, None)):
            turn.privacy_lock.start_owned_native_work(
                lambda: write_required_denial_audit(owner), acquire_span=False)
        await turn.close()

    async with lock.policy_span():
        try:
            loop.set_task_factory(allocate)
            foreign = create_owned_task(unrelated())
            foreign_owner.append(foreign)
        finally:
            loop.set_task_factory(prior)
        await foreign_entered.wait()
        own = create_owned_task(journal(), retain_originals=True)
        try:
            await asyncio.wait((own, foreign_join), return_when=asyncio.FIRST_COMPLETED)
            own_settled = own.done()
            joined_foreign = foreign_join.done()
            foreign_still_pending = not foreign.done() and not witness.exists()
        finally:
            release_foreign.set()
            own_outcome = await await_owned_task(own)
            foreign_outcome = await await_owned_task(foreign)
    assert own_settled and not joined_foreign, 'journal closure joined an unrelated native cohort or its parent'
    assert foreign_still_pending
    assert own_outcome.error is foreign_outcome.error is None
    assert await owner.tool_effect_store.database.fetchone('SELECT COUNT(*) FROM a2a_tool_dispatches') == (1,)
    assert await owner.tool_effect_store.database.fetchone('SELECT state,latched FROM tool_effect_operations') == ('CLOSED', 0)
    assert witness.read_text() == 'original unrelated native operation settled'


@pytest.mark.asyncio
async def test_copied_loop_callback_cannot_construct_an_admitting_journal_root(owner):
    from kestrel_sovereign.tools.effect_journal import ToolEffectTurn
    from tests.unit.test_tool_effect_execution import admit
    turn = await admit(owner, key='original-root-task-required')
    loop = asyncio.get_running_loop()
    result = loop.create_future()
    def callback():
        try:
            ToolEffectTurn(owner, owner.tool_effect_store, turn.attempt, 'unadmitted-loop-callback')
        except RuntimeError as error:
            result.set_result(error)
        else:
            result.set_result(None)
    loop.call_soon(callback)
    refused = await result
    await turn.close()
    assert isinstance(refused, RuntimeError)
    assert str(refused) == 'tool effect journal requires its original admitting Task'


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['ordinary', 'cancel', 'unknown', 'committed', 'group'])
async def test_original_owned_handoff_preserves_every_source_graph(tmp_path, kind):
    from kestrel_sovereign._async_ownership import OwnedTaskHandoff
    original, cause = native_failure(kind)
    delivery = OSError('original consumer result delivery failed')
    witness = tmp_path / 'original-physical-result'

    async def producer():
        witness.write_text('original result physically settled')
        raise original

    task = create_owned_task(producer(), retain_originals=True)
    outcome = await await_owned_task(task)
    before = execution_evidence_carriers(original)
    handoff = OwnedTaskHandoff(outcome, operation='original owned result delivery')
    observed = None
    try:
        with handoff.handoff():
            raise delivery
    except BaseException as error:
        observed = error
    assert execution_evidence_carriers(original) == before, 'finite handoff rewrote its original owned source graph'
    assert original.__cause__ is cause
    if kind == 'cancel':
        assert isinstance(observed, asyncio.CancelledError), 'owned handoff changed its original cancellation contract'
    assert all(any(node is error for node in execution_evidence_carriers(observed))
        for error in (original, cause, delivery))
    assert witness.read_text() == 'original result physically settled'


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['native', 'ordinary-consumer'])
async def test_completed_original_native_control_fences_live_effect_before_root_close(owner, monkeypatch, tmp_path, kind):
    import aiosqlite
    import contextvars
    from kestrel_sovereign.tools.effect_journal import current_tool_effect_turn, execute_tool_effect
    database = owner.tool_effect_store.database
    original, lost, lose_ack = sqlite_commit_ack_loss(database.backend._connection,
        'a2a_tool_dispatches', after=True)
    ordinary, cause = native_failure('ordinary')
    witness, seen = tmp_path / 'effect-after-completed-native-control', {}

    async def audit():
        await write_required_denial_audit(owner)
        if kind == 'ordinary-consumer':
            raise ordinary

    async def effect():
        witness.write_text('later effect')
        return 'acknowledged'

    @governed
    async def invoke(agent, user_input, *, invocation_id=None):
        turn = current_tool_effect_turn(agent)
        if kind == 'native':
            monkeypatch.setattr(aiosqlite.Connection, 'commit', lose_ack)
        # Business context can belong to another consumer. Physical native
        # membership is retained by the original factory, independently.
        worker = turn.privacy_lock.start_owned_native_work(audit,
            context=contextvars.Context(), acquire_span=False)
        outcome = await await_owned_task(worker)
        await asyncio.sleep(0)
        assert worker.done() and worker not in turn.privacy_lock._native_work
        assert not turn.unconfirmed and outcome.error is not None
        seen['source'] = outcome.error
        try:
            seen['result'] = await execute_tool_effect(agent, registration='after-completed-native',
                arguments={}, execute=effect)
        except BaseException as error:
            seen['refused'] = error

    try:
        await invoke(owner, 'original completed native control', invocation_id='completed-native-live-fence')
    except BaseException as error:
        seen['root_error'] = error
    if kind == 'ordinary-consumer':
        assert witness.read_text() == 'later effect' and seen['result'] == 'acknowledged'
        assert 'refused' not in seen and 'root_error' not in seen
        assert ordinary.__cause__ is cause
        assert await database.fetchone('SELECT state,latched FROM tool_effect_operations') == ('CLOSED', 0)
    else:
        assert not witness.exists(), 'completed original native control admitted another live effect'
        assert lost == [True] and 'refused' in seen and 'root_error' in seen
        assert any(error is original for error in execution_evidence_carriers(seen['refused']))
        assert await database.fetchone('SELECT COUNT(*) FROM tool_effect_claims') == (0,)
        assert await database.fetchone('SELECT state,latched FROM tool_effect_operations') == ('CLOSED', 1)
    assert await database.fetchone('SELECT COUNT(*) FROM a2a_tool_dispatches') == (1,)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [None, "ordinary", "sdk", "audit", "control-cancel"])
@pytest.mark.parametrize("publication", [None, "sdk", "audit"])
async def test_original_durable_route_result_reads_complete_native_receipt(tmp_path, kind, publication):
    from types import SimpleNamespace
    from kestrel_sdk.signals import SignalMode, SignalResult, Status
    from kestrel_sovereign.execution_custody import ToolAuditUnconfirmedError, is_tool_execution_terminal_error
    from kestrel_sovereign.signals.dispatcher import SignalDispatcher
    physical = (OSError("original route failed") if kind == "ordinary" else
        ResourceOutcomeUnknown("original route effect uncertain") if kind == "sdk" else
        ToolAuditUnconfirmedError("original route audit uncertain") if kind == "audit" else
        asyncio.CancelledError("original route cancelled"))
    if kind == "control-cancel":
        physical.__cause__ = ExecutionCommitOutcomeError("unknown")
    published = (ResourceOutcomeUnknown("original route publication uncertain") if publication == "sdk"
        else ToolAuditUnconfirmedError("original route publication audit uncertain"))
    effect = tmp_path / "original-route-effect"
    value = SignalResult(signal_id="original-route", status=Status.OK, mode=SignalMode.COGNITION, duration_ms=0)
    async def operation():
        effect.write_text("one original physical route")
        if kind is not None:
            raise physical
        return value
    from kestrel_sovereign._async_ownership import _retain_owned_admission_error
    def publish(task):
        if publication is not None:
            _retain_owned_admission_error(task, published)
    task = create_owned_task(operation(), retain_originals=True)
    task.add_done_callback(publish)
    await asyncio.wait({task})
    observed, result = None, None
    try:
        try:
            result = SignalDispatcher._completed_durable_cognition_result(task,
                signal=SimpleNamespace(id="original-route", mode=SignalMode.COGNITION), start=asyncio.get_running_loop().time())
        except BaseException as error:
            observed = error
        terminal = publication is not None or kind in {"sdk", "audit", "control-cancel"}
        if terminal:
            assert observed is not None and is_tool_execution_terminal_error(observed), (
                "Durable route completion hid original terminal receipt behind its physical result")
            graph = tuple(execution_evidence_carriers(observed))
            if publication is not None:
                assert published in graph
            if kind is not None:
                assert physical in graph
        elif kind is None:
            assert observed is None and result is value
        else:
            assert observed is None and result.status is Status.FAILED
        assert effect.read_text() == "one original physical route" and task.done()
    finally:
        await await_owned_task(task)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [None, "ordinary", "sdk", "audit"])
@pytest.mark.parametrize("completion", ["immediate", "retained", "retry-settlement"])
async def test_original_durable_dispatch_keeps_late_terminal_delivery_nonretryable(tmp_path, monkeypatch, kind, completion):
    await _original_durable_dispatch_case(tmp_path, monkeypatch, kind, completion)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [None, "ordinary", "sdk", "audit", "reconcile", "conflict", "superseded", "closed"])
@pytest.mark.parametrize("retained", [False, True])
async def test_original_durable_result_disposition_survives_retained_route(tmp_path, monkeypatch, kind, retained):
    await _original_durable_dispatch_case(tmp_path, monkeypatch, kind,
        "retained" if retained else "immediate", route_error=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["sdk", "audit", "reconcile", "conflict", "superseded", "closed"])
@pytest.mark.parametrize("retained", [False, True])
async def test_original_reconciliation_result_retains_failed_terminal_write_owner(tmp_path, monkeypatch, kind, retained):
    await _original_durable_dispatch_case(tmp_path, monkeypatch, kind,
        "retained" if retained else "immediate", route_error=True, fail_terminal=True)


async def _original_durable_dispatch_case(tmp_path, monkeypatch, kind, completion, *, route_error=False, fail_terminal=False):
    import json
    from datetime import datetime, timedelta, timezone
    from kestrel_sovereign._async_ownership import _retain_owned_admission_error
    from kestrel_sovereign.execution_custody import ToolAuditUnconfirmedError, is_tool_execution_terminal_error
    from kestrel_sovereign.tool_effect_store import (ToolEffectReconciliationRequired, ToolEffectOperationConflict, ToolEffectAttemptSuperseded, ToolEffectAttemptClosed)
    from kestrel_sovereign.signals import DurableConsumerRegistration
    from kestrel_sovereign.signals.sources.channels import DURABLE_COGNITION_CONSUMER_ID
    from tests.unit.test_durable_signal_delivery import _channel_dispatcher, _channel_signal, _close

    backend, agent, dispatcher = await _channel_dispatcher(
        tmp_path / "late-original-cognition.db", "did:test:late-original-cognition")
    consumer = DurableConsumerRegistration(consumer_id=DURABLE_COGNITION_CONSUMER_ID,
        source="channel.message", agent_id=agent.did, max_attempts=0)
    published = (ResourceOutcomeUnknown("original route late effect uncertainty") if kind == "sdk"
        else ToolAuditUnconfirmedError("original route late audit uncertainty") if kind == "audit"
        else OSError("ordinary late route failure"))
    if kind in {"reconcile", "conflict", "superseded", "closed"}:
        published = (ToolEffectReconciliationRequired({"unsettled": 1}) if kind == "reconcile"
            else {"conflict": ToolEffectOperationConflict, "superseded": ToolEffectAttemptSuperseded,
                  "closed": ToolEffectAttemptClosed}[kind]("original operation requires reconciliation"))
    if fail_terminal and kind != "reconcile":
        published.args = ("private original reconciliation diagnostic",)
    terminal_calls, metadata_error = [], OSError("actual original terminal write failed before SQL")
    if fail_terminal:
        terminal = dispatcher._apply_cognition_control_terminal
        async def fail_first_terminal(delivery, error):
            debt = dispatcher._retained_cognition_control_debt.get(delivery.delivery_id)
            assert debt is not None and debt[0] is delivery and published in tuple(execution_evidence_carriers(debt[1]))
            terminal_calls.append((delivery, delivery.lease_token, error))
            if len(terminal_calls) == 1:
                raise metadata_error
            return await terminal(delivery, error)
        monkeypatch.setattr(dispatcher, "_apply_cognition_control_terminal", fail_first_terminal)
    retained = completion == "retained"
    settlement = completion == "retry-settlement"
    entered, release = asyncio.Event(), asyncio.Event()
    effect, routes = tmp_path / "original-route-effects", []
    handle = None

    async def process_input(prompt):
        task = asyncio.current_task()
        routes.append(task)
        effect.write_text("one original effect")
        if kind is not None and not settlement and not route_error:
            task.add_done_callback(lambda completed: _retain_owned_admission_error(completed, published))
        entered.set()
        if retained:
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
        if route_error and kind is not None:
            raise published
        if settlement:
            raise OSError("original unsuccessful route requires exact-token retry settlement")
        return "original route completed"

    agent.process_input = process_input
    if settlement:
        nack = dispatcher.nack_durable_delivery
        async def publish_after_retry(*args, **kwargs):
            receipt = await nack(*args, **kwargs)
            assert receipt is not None and receipt.status == "retry"
            if kind is not None:
                raise published
            return receipt
        monkeypatch.setattr(dispatcher, "nack_durable_delivery", publish_after_retry)
    outcome = None
    try:
        await dispatcher.register_durable_consumer(consumer)
        handle = await dispatcher.enqueue_durable_cognition(
            _channel_signal(agent.did, "late-original"), source_event_id="late-original-event",
            consumer_id=consumer.consumer_id)
        if retained:
            await entered.wait()
            handle.task.cancel()
            await await_owned_task(handle.task)
            assert routes[0] in dispatcher._retained_durable_cognition_tasks
            release.set()
            await asyncio.wait({routes[0]})
            await asyncio.sleep(0)
            await dispatcher._drain_retained_durable_cognition_cleanup_tasks()
        else:
            outcome = await await_owned_task(handle.task)
        [delivery] = await dispatcher.list_durable_deliveries()
        terminal = kind in {"sdk", "audit", "reconcile", "conflict", "superseded", "closed"}
        if terminal:
            assert delivery.status == "failed" and delivery.next_attempt_at is None, (
                "Late original tool outcome permitted durable cognition replay")
            assert await dispatcher._durable_store.claim_delivery_for_event(
                agent_id=agent.did, consumer_id=consumer.consumer_id, event_id=delivery.event_id,
                executor_id="dispatcher:replacement", now=datetime.now(timezone.utc) + timedelta(hours=1)) is None
        else:
            expected = "acknowledged" if kind is None and completion == "immediate" else "retry"
            assert delivery.status == expected
        if fail_terminal:
            assert len(terminal_calls) == 2
            assert terminal_calls[0][0] is terminal_calls[1][0] and terminal_calls[0][1] == terminal_calls[1][1]
            assert all(published in tuple(execution_evidence_carriers(call[2])) for call in terminal_calls)
            assert "private" not in delivery.last_error
        if terminal and not retained and route_error and fail_terminal:
            assert outcome.error is not None and is_tool_execution_terminal_error(outcome.error)
            assert published in tuple(execution_evidence_carriers(outcome.error))
            assert metadata_error in tuple(execution_evidence_carriers(outcome.error))
        if terminal and not retained and route_error and not fail_terminal:
            from dataclasses import asdict
            from kestrel_sdk.signals import Status
            assert outcome.error is None and outcome.result.status is Status.FAILED
            json.dumps(asdict(outcome.result))
        if terminal and not retained and not route_error:
            assert outcome.error is not None and is_tool_execution_terminal_error(outcome.error)
            assert published in tuple(execution_evidence_carriers(outcome.error))
            assert not is_execution_control_error(outcome.error), "Delivery uncertainty broadened Core authority loss"
        assert effect.read_text() == "one original effect" and len(routes) == 1 and routes[0].done()
    finally:
        release.set()
        if handle is not None:
            await await_owned_task(handle.task)
        for task in routes:
            await await_owned_task(task)
        await dispatcher._drain_retained_durable_cognition_cleanup_tasks()
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.parametrize("kind", ["sdk", "audit", "control", "ordinary"])
def test_original_terminal_delivery_projection_excludes_finite_observer_history(kind):
    from kestrel_sovereign.execution_custody import ToolAuditUnconfirmedError, execution_observation_error
    original = (ResourceOutcomeUnknown("original provider uncertainty") if kind == "sdk"
        else ToolAuditUnconfirmedError("original audit uncertainty") if kind == "audit"
        else ExecutionCommitOutcomeError("unknown") if kind == "control"
        else OSError("ordinary original failure"))
    cause = OSError("original provider diagnostic")
    original.__cause__ = cause
    shared = None
    for index in range(6):
        withdrawal = asyncio.CancelledError("finite observer", index)
        observed = execution_observation_error(original, withdrawal)
        assert (execution_native_source_error(observed) is original) == (kind == "control")
        shared = execution_native_source_error(shared, observed, include_tool_terminal=True)
        if kind == "ordinary":
            assert shared is None
        else:
            assert shared is original, "Shared delivery latch retained finite observer history"
            assert withdrawal not in tuple(execution_evidence_carriers(shared))
        assert original.__cause__ is cause


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["sdk", "audit"])
@pytest.mark.parametrize("physical_failure", [False, True])
async def test_original_public_terminal_delivery_cannot_become_retryable_tooling_failure(tmp_path, kind, physical_failure):
    from kestrel_sdk.signals import SignalHandle, SignalMode, SignalResult, Status
    from kestrel_sovereign._async_ownership import _retain_owned_admission_error
    from kestrel_sovereign.execution_custody import ToolAuditUnconfirmedError, is_tool_execution_terminal_error
    from kestrel_sovereign.signals.delivery import await_terminal_delivery

    error = (ResourceOutcomeUnknown("original uncertain effect") if kind == "sdk"
        else ToolAuditUnconfirmedError("original uncertain audit"))
    cause = OSError("original provider diagnostic")
    error.__cause__ = cause
    effect = tmp_path / "original-delivery-effect"
    async def operation():
        effect.write_text("one original effect")
        if physical_failure:
            raise error
        return SignalResult(signal_id="original-terminal-delivery", status=Status.OK,
            mode=SignalMode.COGNITION, duration_ms=0)
    task = create_owned_task(operation(), retain_originals=True)
    if not physical_failure:
        task.add_done_callback(lambda completed: _retain_owned_admission_error(completed, error))
    handle = SignalHandle(signal_id="original-terminal-delivery", task=task)
    observed = None
    try:
        try:
            await await_terminal_delivery(handle, label="original-terminal-delivery")
        except BaseException as failure:
            observed = failure
        assert observed is not None and is_tool_execution_terminal_error(observed), (
            "Original public SDK/audit delivery authorized ordinary checkpoint restoration")
        assert error in tuple(execution_evidence_carriers(observed))
        assert error.__cause__ is cause
        assert effect.read_text() == "one original effect"
    finally:
        await await_owned_task(task)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["sdk", "audit"])
async def test_original_supervisor_finalizer_retains_independent_terminal_evidence(tmp_path, kind):
    from kestrel_sdk.signals import SignalHandle
    from kestrel_sovereign.execution_custody import ToolAuditUnconfirmedError
    from kestrel_sovereign.signals.delivery import supervise_terminal_delivery
    from tests.unit.test_detached_signal_delivery_harvest import _OwnedTaskFeature

    error = (ResourceOutcomeUnknown("original uncertain effect") if kind == "sdk"
        else ToolAuditUnconfirmedError("original uncertain audit"))
    error.__cause__ = OSError("original provider diagnostic")
    original = tuple(execution_evidence_carriers(error))
    cleanup = ValueError("original consumer finalizer failed")
    effect = tmp_path / "original-supervised-effect"
    async def operation():
        effect.write_text("one original effect")
        raise error
    def settle():
        raise cleanup
    task = create_owned_task(operation(), retain_originals=True)
    owner = _OwnedTaskFeature()
    supervisor = supervise_terminal_delivery(owner,
        SignalHandle(signal_id="original-terminal", task=task), label="original-terminal",
        task_name="original-terminal-supervisor", on_settled=settle)
    try:
        outcome = await await_owned_task(supervisor)
        assert tuple(execution_evidence_carriers(error)) == original
        graph = tuple(execution_evidence_carriers(outcome.error))
        assert error in graph and cleanup in graph, (
            "Original supervisor finalizer failure disappeared behind terminal delivery")
        assert isinstance(outcome.error, type(error))
        assert effect.read_text() == "one original effect"
    finally:
        await await_owned_task(task)


@pytest.mark.asyncio
async def test_actual_supervisor_withdrawal_retains_native_cancellation_contract(tmp_path):
    from kestrel_sdk.signals import SignalHandle
    from kestrel_sovereign.signals.delivery import supervise_terminal_delivery
    from tests.unit.test_detached_signal_delivery_harvest import _OwnedTaskFeature

    entered = asyncio.Event()
    cleanup = ValueError("actual finalizer failed after supervisor withdrawal")
    effect = tmp_path / "original-supervisor-withdrawal"
    async def operation():
        effect.write_text("one original effect")
        entered.set()
        await asyncio.Event().wait()
    def settle():
        raise cleanup
    task = create_owned_task(operation(), retain_originals=True)
    supervisor = supervise_terminal_delivery(_OwnedTaskFeature(),
        SignalHandle(signal_id="original-withdrawal", task=task), label="original-withdrawal",
        task_name="original-withdrawal", on_settled=settle)
    try:
        await entered.wait()
        supervisor.cancel()
        outcome = await await_owned_task(supervisor)
        assert supervisor.cancelled(), (
            "Actual supervisor withdrawal lost native cancellation behind finalizer failure")
        assert cleanup in tuple(execution_evidence_carriers(outcome.error))
        assert effect.read_text() == "one original effect" and task.done()
    finally:
        await await_owned_task(task)


@pytest.mark.parametrize("kind", ["reconcile", "conflict", "superseded", "closed"])
def test_original_tool_reconciliation_projection_excludes_finite_observer_history(kind):
    from kestrel_sovereign.execution_custody import execution_observation_error
    from kestrel_sovereign.tool_effect_store import (
        ToolEffectReconciliationRequired, ToolEffectOperationConflict,
        ToolEffectAttemptSuperseded, ToolEffectAttemptClosed,
    )
    original = (ToolEffectReconciliationRequired({"unsettled": 1}) if kind == "reconcile" else
        {"conflict": ToolEffectOperationConflict, "superseded": ToolEffectAttemptSuperseded,
         "closed": ToolEffectAttemptClosed}[kind]("original operation needs reconciliation"))
    cause = OSError("original provider diagnostic")
    original.__cause__ = cause
    shared = None
    for index in range(6):
        withdrawal = asyncio.CancelledError("actual finite observer", index)
        observed = execution_observation_error(original, withdrawal)
        assert execution_native_source_error(observed) is None
        shared = execution_native_source_error(shared, observed, include_tool_terminal=True)
        assert shared is original, "Shared reconciliation source retained finite observer history"
        assert withdrawal not in tuple(execution_evidence_carriers(shared))
        assert original.__cause__ is cause


def test_reconciliation_result_retains_original_without_adding_public_fields():
    import json
    from dataclasses import asdict
    from kestrel_sdk.signals import SignalMode, SignalResult, Status
    from kestrel_sovereign.signals.dispatcher import _ReconciliationCognitionResult
    original = ResourceOutcomeUnknown("private original provider diagnostic")
    public = SignalResult(signal_id="original-reconciliation", status=Status.FAILED,
        mode=SignalMode.COGNITION, duration_ms=0, error="tool outcome requires reconciliation")
    result = _ReconciliationCognitionResult(**vars(public), original_error=original)
    assert result._original_error is original
    assert vars(result) == asdict(result) == vars(public)
    assert "private" not in json.dumps(asdict(result))
    with pytest.raises(TypeError, match="requires its original failure"):
        _ReconciliationCognitionResult(**vars(public), original_error=None)
