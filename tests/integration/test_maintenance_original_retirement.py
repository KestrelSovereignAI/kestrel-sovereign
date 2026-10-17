"""Actual maintenance/storage/sleep cannot normalize native original control."""
import asyncio
from time import monotonic

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.agent.sleep import SleepReport
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain, bind_execution_custody, execution_cleanup_debts,
    execution_commit_outcome,
)
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.knowledge.maintenance import SemanticMaintenanceService
from kestrel_sovereign.knowledge.maintenance import SemanticMaintenanceError, _MaintenanceBudgetExceeded
from kestrel_sovereign.storage.async_assertion_store import MaintenanceLeaseLostError
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as native_pg
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg_maintenance
from tests.integration.test_execution_custody_review56_pg import lose_native_ack
from tests.integration.test_storage_backend_parity import _incepted_assertion_identity
from kestrel_sovereign.security.assertion_tenant_resolver import _resolve_authenticated_agent_assertion_capability
from kestrel_sovereign.storage import AsyncStorage
from tests.unit.test_semantic_inference import (
    _GOVERNED_STORAGES, assertion_store as assertion_store,
)
from tests.unit.test_host_context_native_cleanup import _actual_operation

native_pg_maintenance = _native_pg_maintenance


async def _original_control(backend, monkeypatch, source_cancel=False):
    async def native_original():
        with bind_execution_custody(GenerationFence()), monkeypatch.context() as fault:
            receipts = lose_native_ack(fault, boundary="commit", original=OSError("actual original maintenance COMMIT ACK lost"))
            try:
                await backend.execute("INSERT INTO effects VALUES (231, 'original maintenance control')")
            except BaseException as error:
                assert receipts == [True]
                if source_cancel:
                    asyncio.current_task().cancel("actual original maintenance source withdrew")
                    try:
                        await asyncio.sleep(0)
                    except asyncio.CancelledError as cancellation:
                        cancellation.__cause__ = error
                        error = cancellation
                return error
        pytest.fail("native maintenance original did not lose COMMIT ACK")
    source = await await_owned_task(create_owned_task(native_original(), retain_originals=True))
    assert source.error is None and source.result is not None
    return source.result


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["service", "storage", "sleep"])
@pytest.mark.parametrize("source_cancel", [False, True])
@pytest.mark.parametrize("close_failure", [False, True])
@pytest.mark.parametrize("withdraw", [False, True])
async def test_maintenance_keeps_original_body_and_lease_retirement_control(
    native_pg, assertion_store, monkeypatch, door, source_cancel, close_failure, withdraw,
):
    backend, observer, schema = native_pg
    original = await _original_control(backend, monkeypatch, source_cancel)
    graph, debts = tuple(_explicit_control_chain(original)), tuple(execution_cleanup_debts(original))
    close_error = OSError("actual native lease release observer failed")
    close_graph = tuple(_explicit_control_chain(close_error))
    storage = _GOVERNED_STORAGES[id(assertion_store)]
    database = assertion_store._database
    native_read, native_execute = database.fetchone, database.execute
    entered, release = asyncio.Event(), asyncio.Event()
    states, closers = [], []
    async def read(sql, *args, **kwargs):
        result = await native_read(sql, *args, **kwargs)
        if "FROM semantic_maintenance_state " in sql and not states:
            states.append(True)
            raise original
        return result
    async def execute(sql, *args, **kwargs):
        result = await native_execute(sql, *args, **kwargs)
        if sql.startswith("DELETE FROM semantic_maintenance_leases "):
            closers.append(asyncio.current_task())
            entered.set()
            await release.wait()
            if close_failure:
                raise close_error
        return result
    monkeypatch.setattr(database, "fetchone", read)
    monkeypatch.setattr(database, "execute", execute)
    report = SleepReport(success=False)
    agent = KestrelAgent.__new__(KestrelAgent)
    agent.storage = storage
    agent.semantic_maintenance_configured = True
    async def invoke():
        if door == "service":
            return await SemanticMaintenanceService(assertion_store, inference_profile=None).run()
        if door == "storage":
            return await storage.run_semantic_maintenance(None)
        return await agent._run_semantic_maintenance(report)
    owner = create_owned_task(invoke(), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        if withdraw:
            owner.cancel("actual maintenance cleanup observer withdrew")
            await asyncio.sleep(0)
            assert not owner.done(), "maintenance returned before original native lease retirement"
        release.set()
        outcome = await await_owned_task(owner)
        chain = tuple(_explicit_control_chain(outcome.error))
        assert original in chain, "maintenance normalized or replaced original native body control"
        if close_failure:
            assert close_error in chain, "maintenance lost independent original lease retirement control"
        assert execution_commit_outcome(outcome.error) == "unknown"
        assert len(closers) == 1 and closers[0].done()
        assert tuple(_explicit_control_chain(original)) == graph
        assert tuple(execution_cleanup_debts(original)) == debts
        assert tuple(_explicit_control_chain(close_error)) == close_graph
        assert await database.fetchval("SELECT count(*) FROM semantic_maintenance_leases") == 0
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=231') == 1
        if door == "sleep":
            assert report.semantic_maintenance is None, "native control was disclosed as ordinary sleep diagnostics"
    finally:
        release.set()
        await await_owned_task(owner)


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["capability", "readiness", "budget", "lease", "failed-report", "report-source"])
async def test_maintenance_domain_handlers_preserve_original_control_graph(native_pg, assertion_store, monkeypatch, target):
    original = await _original_control(native_pg[0], monkeypatch)
    service = SemanticMaintenanceService(assertion_store, inference_profile=None)
    database = assertion_store._database
    selected = (_MaintenanceBudgetExceeded if target == "budget" else
                MaintenanceLeaseLostError if target in ("lease", "report-source") else
                SemanticMaintenanceError)
    carrier = selected("semantic_maintenance_lease_lost")
    carrier.__cause__ = original
    graph, debts = tuple(_explicit_control_chain(original)), tuple(execution_cleanup_debts(original))
    carrier_graph = tuple(_explicit_control_chain(carrier))
    actual_capability, actual_read, actual_log, actual_execute = (
        service._capability_versions, database.fetchone, service._log_result, database.execute,
    )
    seen, leases = [], []
    body_read_error = ValueError("ordinary worker read failure before terminal report")
    def capability():
        actual_capability()
        seen.append(True)
        raise carrier
    async def read(sql, *args, **kwargs):
        result = await actual_read(sql, *args, **kwargs)
        if "FROM semantic_maintenance_state " in sql and not seen:
            seen.append(True)
            if target == "failed-report":
                raise body_read_error
            raise carrier
        return result
    def log(result):
        actual_log(result)
        raise carrier
    async def execute(sql, *args, **kwargs):
        result = await actual_execute(sql, *args, **kwargs)
        if sql.startswith("INSERT INTO semantic_maintenance_reports "):
            seen.append(True)
            raise carrier
        return result
    if target in ("capability", "readiness"):
        monkeypatch.setattr(service, "_capability_versions", capability)
    elif target == "report-source":
        monkeypatch.setattr(database, "execute", execute)
    else:
        monkeypatch.setattr(database, "fetchone", read)
        if target == "failed-report":
            monkeypatch.setattr(service, "_log_result", log)
    async def invoke():
        if target == "readiness":
            return await service.training_readiness()
        if target == "report-source":
            lease = await service._acquire_lease("original-domain-handler")
            leases.append(lease)
            return await service._record_candidate(
                "contradiction_candidate", "review_required", {"source": "content-free"},
                started=monotonic(), reports_created=0, lease=lease,
            )
        return await service.run()
    owner = create_owned_task(invoke(), retain_originals=True)
    try:
        result = await await_owned_task(owner)
        assert seen, f"actual domain observer was not reached: {result.error!r}"
        assert seen and carrier in tuple(_explicit_control_chain(result.error)), "maintenance domain handler stripped original source carrier"
        assert original in tuple(_explicit_control_chain(result.error)), "maintenance domain handler normalized original native uncertainty"
        assert execution_commit_outcome(result.error) == "unknown"
        if target == "failed-report":
            assert body_read_error in tuple(_explicit_control_chain(result.error)), "diagnostic failure replaced the original maintenance body error"
        assert tuple(_explicit_control_chain(original)) == graph
        assert tuple(execution_cleanup_debts(original)) == debts
        assert tuple(_explicit_control_chain(carrier)) == carrier_graph, "maintenance rewrote original source graph during domain unwrapping"
    finally:
        await await_owned_task(owner)
        for lease in leases:
            await service._release_lease(lease)


@pytest.mark.asyncio
async def test_maintenance_postgres_report_source_preserves_original_graph(
    native_pg, native_pg_maintenance, tmp_path, monkeypatch,
):
    original = await _original_control(native_pg[0], monkeypatch)
    backend, observer, schema = native_pg_maintenance
    assert backend is not native_pg[0]
    did, identity = await _incepted_assertion_identity(tmp_path, "maintenance-original")
    storage = AsyncStorage(
        backend=backend, agent_id=did,
        _assertion_tenant_capability=_resolve_authenticated_agent_assertion_capability(did, identity),
    )
    await storage.initialize()
    service = SemanticMaintenanceService(storage._assertion_store(), inference_profile=None)
    database = service._database
    carrier = MaintenanceLeaseLostError("semantic_maintenance_lease_lost")
    carrier.__cause__ = original
    graph = tuple(_explicit_control_chain(carrier))
    debts = tuple(execution_cleanup_debts(original))
    actual_execute, seen = database.execute, []
    async def execute(sql, *args, **kwargs):
        result = await actual_execute(sql, *args, **kwargs)
        if sql.startswith("INSERT INTO semantic_maintenance_reports "):
            seen.append(True)
            raise carrier
        return result
    lease = await service._acquire_lease("original-postgres-report")
    monkeypatch.setattr(database, "execute", execute)
    owner = create_owned_task(service._record_candidate(
        "contradiction_candidate", "review_required", {"source": "content-free"},
        started=monotonic(), reports_created=0, lease=lease,
    ), retain_originals=True)
    try:
        outcome = await await_owned_task(owner)
        assert seen, f"actual PostgreSQL report observer was not reached: {outcome.error!r}"
        assert original in tuple(_explicit_control_chain(outcome.error)), "maintenance domain handler normalized original native uncertainty"
        assert tuple(_explicit_control_chain(carrier)) == graph, "maintenance rewrote original source graph during PostgreSQL domain unwrapping"
        assert tuple(execution_cleanup_debts(original)) == debts
        assert execution_commit_outcome(outcome.error) == "unknown"
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".semantic_maintenance_reports') == 0
    finally:
        await await_owned_task(owner)
        await service._release_lease(lease)
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("after", [False, True])
@pytest.mark.parametrize("eager", [False, True] if hasattr(asyncio, "eager_task_factory") else [False])
async def test_maintenance_close_receipt_publication_retains_original_unstarted_lease(
    native_pg, assertion_store, monkeypatch, after, eager,
):
    original = await _original_control(native_pg[0], monkeypatch)
    service = SemanticMaintenanceService(assertion_store, inference_profile=None)
    database = assertion_store._database
    actual_read = database.fetchone
    seen, accepted, leases = [], [], []
    async def read(sql, *args, **kwargs):
        result = await actual_read(sql, *args, **kwargs)
        if "FROM semantic_maintenance_state " in sql and not seen:
            seen.append(True)
            raise original
        return result
    monkeypatch.setattr(database, "fetchone", read)
    refusal = RuntimeError("actual original lease-close receipt publication failed")
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    class OriginalTask(asyncio.Task):
        failed = False
        def add_done_callback(self, callback, *, context=None):
            if self.failed or getattr(callback, "__name__", None) != "close_unstarted":
                return super().add_done_callback(callback, context=context)
            self.failed = True
            if after:
                super().add_done_callback(callback, context=context)
            raise refusal
    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        frame = getattr(operation, "cr_frame", None)
        stream = frame.f_locals.get("stream") if frame is not None else None
        if (getattr(getattr(operation, "cr_code", None), "co_qualname", "") == "close_native_execution_stream.<locals>.owned_close"
                and hasattr(stream, "fencing_token")):
            leases.append(stream)
            options = {"eager_start": True} if eager else {}
            task = OriginalTask(coroutine, loop=loop, **kwargs, **options)
            accepted.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    loop.set_task_factory(factory)
    owner = create_owned_task(service.run(), retain_originals=True)
    try:
        result = await await_owned_task(owner)
        chain = tuple(_explicit_control_chain(result.error))
        assert original in chain and refusal in chain
        assert len(accepted) == 1 and accepted[0].done(), "lease retirement escaped after canonical receipt publication failed"
        assert any(resource is leases[0] and task is accepted[0] for resource, task in execution_cleanup_debts(result.error)), "unstarted lease retirement lost original resource/Task debt"
        assert await database.fetchval("SELECT count(*) FROM semantic_maintenance_leases") == 1, "unpublished original closer spent lease release after failed receipt publication"
    finally:
        loop.set_task_factory(previous)
        await await_owned_task(owner)
        for task in accepted:
            await await_owned_task(task)
        for lease in leases:
            await service._release_lease(lease)



@pytest.mark.asyncio
@pytest.mark.parametrize("body_kind", ["ordinary", "budget", "lease"])
@pytest.mark.parametrize("retirement", ["failed", "admission", "withdraw"])
async def test_handled_worker_with_successful_diagnostics_survives_retirement_control(
    native_pg, assertion_store, monkeypatch, body_kind, retirement,
):
    database = assertion_store._database
    service = SemanticMaintenanceService(assertion_store, inference_profile=None)
    kinds = {"ordinary": ValueError, "budget": _MaintenanceBudgetExceeded, "lease": MaintenanceLeaseLostError}
    body = kinds[body_kind]("actual handled maintenance worker failure")
    body_graph = tuple(_explicit_control_chain(body))
    native = await _original_control(native_pg[0], monkeypatch)
    native_graph, native_debts = tuple(_explicit_control_chain(native)), tuple(execution_cleanup_debts(native))
    actual_read, actual_execute, actual_record = database.fetchone, database.execute, service._record_run_if_current
    seen, diagnostics, leases = [], [], []
    entered, release = asyncio.Event(), asyncio.Event()
    async def read(sql, *args, **kwargs):
        result = await actual_read(sql, *args, **kwargs)
        if "FROM semantic_maintenance_state " in sql and not seen:
            seen.append(True)
            raise body
        return result
    async def record(*args, **kwargs):
        result = await actual_record(*args, **kwargs)
        diagnostics.append(True)
        return result
    async def execute(sql, *args, **kwargs):
        result = await actual_execute(sql, *args, **kwargs)
        if sql.startswith("DELETE FROM semantic_maintenance_leases "):
            entered.set()
            await release.wait()
            if retirement == "failed":
                raise native
        return result
    monkeypatch.setattr(database, "fetchone", read)
    monkeypatch.setattr(database, "execute", execute)
    monkeypatch.setattr(service, "_record_run_if_current", record)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    refused = RuntimeError("original handled-worker lease closer refused")
    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        frame = getattr(operation, "cr_frame", None)
        stream = frame.f_locals.get("stream") if frame is not None else None
        if (retirement == "admission"
                and getattr(getattr(operation, "cr_code", None), "co_qualname", "") == "close_native_execution_stream.<locals>.owned_close"
                and hasattr(stream, "fencing_token")):
            leases.append(stream)
            raise refused
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    owner = create_owned_task(service.run(), retain_originals=True)
    loop.set_task_factory(factory)
    try:
        if retirement != "admission":
            await asyncio.wait_for(entered.wait(), 5)
            if retirement == "withdraw":
                owner.cancel("actual handled-worker lease retirement observer withdrew")
                await asyncio.sleep(0)
                assert not owner.done()
            release.set()
        outcome = await await_owned_task(owner)
        chain = tuple(_explicit_control_chain(outcome.error))
        assert diagnostics, "actual successful diagnostic persistence was not reached"
        assert body in chain, "successful diagnostics dropped handled worker during failed retirement"
        if retirement == "failed":
            assert native in chain
            assert tuple(_explicit_control_chain(native)) == native_graph
            assert tuple(execution_cleanup_debts(native)) == native_debts
        elif retirement == "admission":
            assert refused in chain and leases
            assert any(resource is leases[0] and task is None for resource, task in execution_cleanup_debts(outcome.error))
        else:
            assert any(isinstance(error, asyncio.CancelledError) for error in chain)
        assert tuple(_explicit_control_chain(body)) == body_graph, "handled worker source graph changed during retirement composition"
        assert await database.fetchval("SELECT count(*) FROM semantic_maintenance_runs") == 1
    finally:
        loop.set_task_factory(previous)
        release.set()
        await await_owned_task(owner)
        for lease in leases:
            await service._release_lease(lease)
