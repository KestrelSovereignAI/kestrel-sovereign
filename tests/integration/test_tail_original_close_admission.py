"""Actual partial Agent SQLite tail keeps failed admission and attempts peers."""
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import execution_cleanup_debts
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.integration.test_snapshot_agent_original_retirement import _retire_partial_agent_fixture
from tests.unit.test_cleanup_task_admission import _partial_native_agent
from tests.utils.native_admission_fault import NativeAdmissionFault
from tests.utils.original_failure import OriginalFailure

native_pg = _native_pg


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["refuse", "publish-before", "publish-after"])
@pytest.mark.parametrize("source_cancel", [False, True])
@pytest.mark.parametrize("target", ["snapshot", "stop", "storage"])
@pytest.mark.parametrize("eager", [False, True])
async def test_tail_original_admission_keeps_debt_and_independent_physical_cleanup(
    native_pg, tmp_path, monkeypatch, mode, source_cancel, target, eager,
):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    agent, db = await _partial_native_agent(tmp_path, monkeypatch)
    connection = db.backend._connection
    worker = getattr(connection, "_thread", connection)
    effects = SQLiteBackend(":memory:")
    await effects.connect()
    await effects.execute("CREATE TABLE effects (id INTEGER)")

    class Sync:
        is_running = True

        async def force_snapshot(self):
            await effects.execute("INSERT INTO effects VALUES (351)")

        async def stop(self):
            await effects.execute("INSERT INTO effects VALUES (352)")
            self.is_running = False

    sync = Sync()
    agent._sync_service = sync
    agent._resource_custody().acquire("sync_service")

    def selected(operation):
        code, frame = getattr(operation, "cr_code", None), getattr(operation, "cr_frame", None)
        if code is None or frame is None:
            return False
        if target == "snapshot":
            return code is Sync.force_snapshot.__code__
        return code.co_qualname == "ResourceCustody.release" and frame.f_locals.get("name") == ("sync_service" if target == "stop" else "storage")

    fault = NativeAdmissionFault(selected, source.error, mode, eager=eager)
    owner = create_owned_task(agent._run_durable_shutdown_tail(3), retain_originals=True)
    try:
        with fault:
            result = await await_owned_task(owner)
        fault.assert_terminal_unstarted()
        source.assert_retained(result.error)
        resource = agent.storage if target == "storage" else sync
        assert any(value is resource and task is fault.task for value, task in execution_cleanup_debts(result.error))
        ids = [row[0] for row in await effects.fetch_all("SELECT id FROM effects ORDER BY id")]
        assert ids == ([352] if target == "snapshot" else [351] if target == "stop" else [351, 352])
        assert (db.backend._connection is connection and worker.is_alive()) if target == "storage" else (db.backend._connection is None and not worker.is_alive())
        assert not agent._runtime_publication_ready.is_set(), "partial fixture is not Agent READY acceptance"
    finally:
        await await_owned_task(owner)
        for task in fault.accepted:
            await await_owned_task(task)
        await _retire_partial_agent_fixture(agent, db)
        await effects.close()
