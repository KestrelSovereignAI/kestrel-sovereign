"""Outer lifecycle consumers preserve finite originals and actual withdrawals.

These controlled lifecycle seams execute real SQLite/PG effects. They are not
whole-Agent, financial, authenticated-provider or production acceptance.
"""
import asyncio
import sqlite3

import pytest
from fastapi import FastAPI
from kestrel_sdk.features.host_base import HostFeature
from kestrel_sdk.signals import Signal, SignalMode, SignalResult, Status

from kestrel_sovereign import _async_ownership as ownership
from kestrel_sovereign._async_ownership import OwnedTaskAdmission, await_owned_task, create_owned_task, owned_task_admission_error
from kestrel_sovereign.execution_custody import _explicit_control_chain
from kestrel_sovereign.host_features import SovereignHostContext, start_host_features
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg
from tests.integration.test_execution_custody_review18_pg import setup_delivery, join_owned
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as _onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.unit.test_durable_signal_delivery import _registration
from tests.utils.native_admission_fault import NativeAdmissionFault
from tests.utils.original_failure import OriginalFailure

native_pg = _native_pg
onboarding_owner = _onboarding_owner


async def unstarted_native_close():
    resource = SQLiteBackend(":memory:")
    await resource.connect()
    admission = OwnedTaskAdmission()
    fault = NativeAdmissionFault(lambda op: getattr(op, "cr_code", None) is SQLiteBackend.close.__code__, OSError("actual native close admission refused"), "refuse")
    with fault:
        try:
            create_owned_task(resource.close(), retirement_resource=resource, retain_originals=True, admission=admission)
        except BaseException as error:
            original = owned_task_admission_error(resource, admission, error)
    fault.assert_terminal_unstarted()
    assert admission.task is None and resource._connection is not None
    return resource, OriginalFailure.capture(original)


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["host", "scheduler", "agents", "shutdown-drain", "liveness"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_outer_consumers_keep_independent_native_sources_immutable(
    native_pg, tmp_path, monkeypatch, surface, source_cancel,
):
    from kestrel_sovereign import server
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    resource, independent = await unstarted_native_close()
    calls, databases, paths = [], [], []
    agent = dispatcher = None
    app = FastAPI()
    async def database(index):
        path = tmp_path / ("outer-" + str(index) + ".db")
        db = SQLiteBackend(path)
        await db.connect()
        await db.execute("CREATE TABLE effects (id INTEGER)")
        databases.append(db)
        paths.append(path)
        return db
    class ClosingRole:
        def __init__(self, name, db, failure):
            self.name, self.db, self.failure = name, db, failure
        def __getattr__(self, name):
            return getattr(self.db, name)
        async def close(self):
            calls.append(self.name)
            await self.db.execute("INSERT INTO effects VALUES (1)")
            await self.db.close()
            raise self.failure
        stop = close
        shutdown_all = close
    try:
        if surface == "host":
            db = await database(0)
            ctx = SovereignHostContext(db=ClosingRole("context", db, independent.error))
            class Failing(HostFeature):
                name = "outer-native-host"
                async def on_host_stop(self, ctx):
                    calls.append("hook")
                    await db.execute("INSERT INTO effects VALUES (2)")
                    raise source.error
            feature = Failing()
            await start_host_features([feature], ctx)
            app.state.host_context, app.state.host_features = ctx, [feature]
            operation = server._shutdown_host_features(app)
        elif surface in ("scheduler", "agents"):
            first = ClosingRole("first", await database(0), source.error)
            second = ClosingRole("second", await database(1), independent.error)
            if surface == "scheduler":
                app.state.host_scheduler_runner, app.state.host_scheduler_storage = first, second
                operation = server._shutdown_host_scheduler(app)
            else:
                app.state.agent_manager, app.state.startup_cleanup_agent_manager = first, second
                operation = server._shutdown_server_agents(app)
        else:
            agent, dispatcher, delivery, _ = await setup_delivery(native_pg)
            if surface == "shutdown-drain":
                async def first(*, finite_errors=None):
                    calls.append("first")
                    await native_pg[1].execute('INSERT INTO "' + native_pg[2] + '".effects VALUES (381,\'first actual drain\')')
                    raise source.error
                async def second(*, finite_errors=None):
                    calls.append("second")
                    await native_pg[1].execute('INSERT INTO "' + native_pg[2] + '".effects VALUES (382,\'second actual drain\')')
                    raise independent.error
                monkeypatch.setattr(dispatcher, "_drain_outcome_log_tasks", first)
                monkeypatch.setattr(dispatcher, "_drain_post_commit_reservation_repairs", second)
                operation = dispatcher._complete_durable_shutdown()
            else:
                # Declared stale debt snapshots, no fabricated lease grant.
                dispatcher._retained_cognition_control_debt = {
                    "first-stale-source": (delivery, source.error),
                    "second-stale-source": (delivery, independent.error),
                }
                dispatcher._durable_delivery_owner = "dispatcher:unregistered-native-cleanup-owner"
                actual = dispatcher._durable_store.backend.retain_cognition_cleanup_owner
                async def zero(**kwargs):
                    result = await actual(**kwargs)
                    assert result is False
                    calls.append("zero-original-cas")
                    return result
                monkeypatch.setattr(dispatcher._durable_store.backend, "retain_cognition_cleanup_owner", zero)
                operation = dispatcher._retain_runtime_owner_cleanup_liveness()
        result = await await_owned_task(create_owned_task(operation, retain_originals=True))
        source.assert_retained(result.error)
        independent.assert_retained(result.error)
        source.assert_retained(source.error)
        independent.assert_retained(independent.error)
        expected = ["hook", "context"] if surface == "host" else ["zero-original-cas"] if surface == "liveness" else ["first", "second"]
        assert calls == expected
        for db, path in zip(databases, paths, strict=True):
            assert db._connection is None
            with sqlite3.connect(path) as observer:
                assert observer.execute("SELECT count(*) FROM effects").fetchone()[0] >= 1
    finally:
        if dispatcher is not None:
            await join_owned(agent, dispatcher)
        for db in databases:
            await db.close()
        await resource.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["writer-join", "log-safe"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_writer_consumers_keep_actual_current_join_withdrawal(
    native_pg, monkeypatch, surface, source_cancel,
):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    agent, dispatcher, _, _ = await setup_delivery(native_pg)
    entered, release, withdrawals = asyncio.Event(), asyncio.Event(), []
    owner = None
    actual_join = ownership.await_owned_task
    async def observe(task, *args, **kwargs):
        result = await actual_join(task, *args, **kwargs)
        if asyncio.current_task() is owner and result.cancellation is not None:
            withdrawals.append(result.cancellation)
        return result
    async def writer(*args, **kwargs):
        await native_pg[1].execute('INSERT INTO "' + native_pg[2] + '".effects VALUES (383,\'actual original writer\')')
        entered.set()
        await release.wait()
        raise source.error
    monkeypatch.setattr(ownership, "await_owned_task", observe)
    original = None
    if surface == "writer-join":
        original = create_owned_task(writer(), retain_originals=True)
        operation = dispatcher._join_outcome_log_tasks([original])
    else:
        monkeypatch.setattr(dispatcher, "_write_outcome_log", writer)
        signal = Signal(source="native.finite.writer", kind="component", mode=SignalMode.ARTIFACT, target_agent=agent.did, payload={})
        result = SignalResult(signal_id=signal.id, status=Status.OK, mode=signal.mode, duration_ms=0)
        operation = dispatcher._log_safe(signal, _registration(agent), result)
    owner = create_owned_task(operation, retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        owner.cancel("actual native writer observer withdrew")
        await asyncio.sleep(0)
        release.set()
        result = await await_owned_task(owner)
        source.assert_retained(result.error)
        assert withdrawals and all(error in tuple(_explicit_control_chain(result.error)) for error in withdrawals)
        if dispatcher._outcome_log_control_error is not None:
            source.assert_retained(dispatcher._outcome_log_control_error)
            assert all(error not in tuple(_explicit_control_chain(dispatcher._outcome_log_control_error)) for error in withdrawals)
        assert await native_pg[1].fetchval('SELECT count(*) FROM "' + native_pg[2] + '".effects WHERE id=383') == 1
    finally:
        release.set()
        await await_owned_task(owner)
        if original is not None:
            await await_owned_task(original)
        await join_owned(agent, dispatcher)
