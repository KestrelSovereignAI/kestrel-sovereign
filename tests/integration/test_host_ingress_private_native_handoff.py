"""Host ingress retains backend-established originals without public secrets.

The facade is declared component input; COMMIT, reset and observed rows are
actual localhost PostgreSQL operations, not fabricated control receipts.
"""
import asyncio

import asyncpg
import pytest
from kestrel_sdk.isolated_feature import HostIngressCapabilities

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionCustody, bind_execution_custody_snapshot, execution_commit_outcome,
    is_execution_control_error,
)
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as native_pg
from tests.integration.test_execution_custody_review56_pg import lose_native_ack
from tests.unit.test_isolated_feature_runtime import (
    _HostIngressClient, _initialized_host_ingress_proxy,
    _assert_runtime_traceback_locals_are_detached, _traceback_locals_reach_any,
)
from tests.utils.original_failure import OriginalFailure


@pytest.mark.asyncio
@pytest.mark.parametrize("suppress", [False, True])
@pytest.mark.parametrize("withdraw", [False, True])
@pytest.mark.parametrize("producer", ["same", "owned", "direct", "gather", "taskgroup", "nested", "groupbody", "groupdirect", "groupgather", "eager", "eagernested"])
async def test_ingress_hands_off_actual_native_failure_before_foreign_redaction(
    native_pg, tmp_path, monkeypatch, suppress, withdraw, producer,
):
    backend, observer, schema = native_pg
    secret = "private-ingress-request-and-facade-secret-205"
    request = {"credential": secret}
    marker = OSError("actual original ingress COMMIT ACK lost")
    source = OriginalFailure.capture(marker)
    entered, release, exits, resets, calls, failures = asyncio.Event(), asyncio.Event(), [], [], [], []
    class Client(_HostIngressClient):
        async def call_host_ingress(self, name, payload=None):
            calls.append(asyncio.current_task())
            try:
                work = backend.execute("INSERT INTO effects VALUES (205, 'private ingress actual effect')")
                if producer == "same":
                    await work
                elif producer == "owned":
                    child = create_owned_task(work, retain_originals=True)
                    child_outcome = await await_owned_task(child)
                    if child_outcome.error is not None:
                        raise child_outcome.error
                elif producer == "direct":
                    await asyncio.create_task(work)
                elif producer == "gather":
                    await asyncio.gather(work)
                elif producer in ("eager", "eagernested"):
                    def eager_child(group, operation):
                        loop = asyncio.get_running_loop()
                        previous = loop.get_task_factory()
                        loop.set_task_factory(asyncio.eager_task_factory)
                        try:
                            return group.create_task(operation)
                        finally:
                            loop.set_task_factory(previous)
                    async def nested_eager():
                        async with asyncio.TaskGroup() as inner:
                            eager_child(inner, work)
                    async with asyncio.TaskGroup() as outer:
                        eager_child(outer, nested_eager() if producer == "eagernested" else work)
                elif producer == "nested":
                    async def nested():
                        async with asyncio.TaskGroup() as inner:
                            inner.create_task(work)
                    async with asyncio.TaskGroup() as outer:
                        outer.create_task(nested())
                elif producer in ("groupdirect", "groupgather"):
                    async def mixed():
                        if producer == "groupdirect":
                            await asyncio.create_task(work)
                        else:
                            await asyncio.gather(work)
                    async with asyncio.TaskGroup() as group:
                        group.create_task(mixed())
                elif producer == "groupbody":
                    async with asyncio.TaskGroup() as group:
                        group.create_task(work)
                        # The child reaches its native phase BEFORE the
                        # parent starts TaskGroup.__aexit__ or owns its join
                        # Future. Membership is already a physical obligation.
                        await entered.wait()
                else:
                    async with asyncio.TaskGroup() as group:
                        group.create_task(work)
            except BaseException as error:
                failures.append(type(error).__name__)
                if suppress:
                    return {"private-response": secret}
                raise RuntimeError(secret) from None
            pytest.fail("fixture failed to lose actual native ACK")
    def client_factory(**kwargs):
        return Client(ingress_capabilities=HostIngressCapabilities(names=("private-input",)), **kwargs)
    feature, _ = await _initialized_host_ingress_proxy(monkeypatch, tmp_path, client_factory)
    client = feature._client
    assert feature._host_ingress_capabilities().names == ("private-input",)
    lose_native_ack(monkeypatch, boundary="commit", original=marker)
    native_exit = asyncpg.transaction.Transaction.__aexit__
    native_reset = asyncpg.pool.PoolAcquireContext.__aexit__
    async def exit_transaction(transaction, *args):
        exits.append((transaction, asyncio.current_task()))
        return await native_exit(transaction, *args)
    async def reset(checkout, *args):
        if checkout.pool is backend._pool:
            resets.append(asyncio.current_task())
            entered.set()
            await release.wait()
        return await native_reset(checkout, *args)
    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", exit_transaction)
    monkeypatch.setattr(asyncpg.pool.PoolAcquireContext, "__aexit__", reset)
    scope = ExecutionCustody(GenerationFence())
    async def call():
        with bind_execution_custody_snapshot((scope,)):
            return await feature.call_host_ingress("private-input", request)
    caller = create_owned_task(call(), retain_originals=True)
    signal = create_owned_task(entered.wait())
    try:
        done, _ = await asyncio.wait({caller, signal}, timeout=5, return_when=asyncio.FIRST_COMPLETED)
        assert signal in done, f"fixture never reached reset; calls={len(calls)} exits={len(exits)} failures={failures}"
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=205') == 1
        if withdraw:
            caller.cancel("actual private ingress caller withdrew")
            await asyncio.sleep(0)
        assert not caller.done(), "public ingress abandoned original native reset"
        release.set()
        outcome = await await_owned_task(caller)
        assert is_execution_control_error(outcome.error), "redaction erased actual native handoff"
        assert execution_commit_outcome(outcome.error) == "unknown"
        records = tuple(feature._host_ingress_admissions)
        assert len(records) == 1 and records[0].admission.task.done()
        record = records[0]
        source.assert_retained(record.native_error)
        assert record.scopes == (scope,)
        assert record.client is client and record.observer is not None
        assert any(fact.owner is task and fact.resource is transaction
                   for fact in record.observer.native_failures for transaction, task in exits)
        assert all(task.done() for _, task in exits) and all(task.done() for task in resets)
        assert (record.cancellation is not None) is withdraw
        forbidden = (secret, request, client, feature, record, record.observer,
                     record.admission, record.admission.task, marker, *resets)
        _assert_runtime_traceback_locals_are_detached(outcome.error, forbidden_values=forbidden)
        assert not _traceback_locals_reach_any(outcome.error, forbidden)
        assert secret not in str(outcome.error) and secret not in repr(outcome.error)
        repeated = await await_owned_task(create_owned_task(feature.call_host_ingress("private-input", request), retain_originals=True))
        assert is_execution_control_error(repeated.error) and len(calls) == 1
        assert not _traceback_locals_reach_any(repeated.error, forbidden)
        rejected = await await_owned_task(create_owned_task(feature.initialize(), retain_originals=True))
        assert is_execution_control_error(rejected.error)
        assert not _traceback_locals_reach_any(rejected.error, forbidden)
        for operation in (
            feature.get_config(), feature.set_config({"credential": secret}),
            feature.reload(), feature.call_isolated_tool("private-input", request),
        ):
            later = await await_owned_task(create_owned_task(operation, retain_originals=True))
            assert is_execution_control_error(later.error)
            assert not _traceback_locals_reach_any(later.error, forbidden)
        async def fenced():
            async with feature.config_transition_ingress_fence():
                pytest.fail("native-unresolved ingress reopened configuration")
        later = await await_owned_task(create_owned_task(fenced(), retain_originals=True))
        assert is_execution_control_error(later.error)
        assert not _traceback_locals_reach_any(later.error, forbidden)
        stopped = await await_owned_task(create_owned_task(feature.shutdown(), retain_originals=True))
        assert is_execution_control_error(stopped.error) and client.stopped
        assert not _traceback_locals_reach_any(stopped.error, forbidden)
        assert tuple(feature._host_ingress_admissions) == records
    finally:
        release.set()
        await await_owned_task(caller)
        signal.cancel()
        await await_owned_task(signal)
        supervisor = feature._supervision_task
        if supervisor is not None:
            feature._supervision_task = None
            supervisor.cancel()
            await await_owned_task(supervisor)
