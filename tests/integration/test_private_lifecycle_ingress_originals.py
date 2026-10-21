"""Reserved lifecycle RPCs use the SAME private native handoff, not data admission."""
import asyncio

import asyncpg
import pytest
from kestrel_sdk.isolated_feature import HostIngressCapabilities

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import ExecutionCustody, bind_execution_custody_snapshot, execution_commit_outcome, _explicit_control_chain
from kestrel_sovereign.storage.db import postgres as postgres_module
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as native_pg
from tests.integration.test_execution_custody_review56_pg import lose_native_ack
from tests.unit.test_isolated_feature_runtime import _HostIngressClient, _initialized_host_ingress_proxy, _traceback_locals_reach_any
from tests.utils.original_failure import OriginalFailure

NAMES = ("external-ingress-quiesce", "external-ingress-resume")


async def lifecycle_component(monkeypatch, tmp_path, backend, records):
    class Client(_HostIngressClient):
        lifecycle_call_count = 0
        async def call_host_ingress(self, name, payload=None):
            self.lifecycle_call_count += 1
            records.extend(tuple(feature._host_ingress_admissions))
            await backend.execute("INSERT INTO effects VALUES ($1, 'declared private lifecycle')", (290 + self.lifecycle_call_count,))
            return {"status": "ok", "http_status": 200,
                    "state": "quiesced" if name == NAMES[0] else "resumed"}
    def factory(**kwargs):
        return Client(ingress_capabilities=HostIngressCapabilities(names=NAMES), **kwargs)
    feature, _ = await _initialized_host_ingress_proxy(monkeypatch, tmp_path, factory)
    return feature


@pytest.mark.asyncio
@pytest.mark.parametrize("name", NAMES)
async def test_reserved_lifecycle_rpc_keeps_actual_original_native_handoff(native_pg, tmp_path, monkeypatch, name):
    backend, observer, schema = native_pg
    source = OriginalFailure.capture(OSError("actual reserved ingress COMMIT ACK lost"))
    feature = await lifecycle_component(monkeypatch, tmp_path, backend, [])
    quiesce = feature._new_external_ingress_quiesce()
    lose_native_ack(monkeypatch, boundary="commit", original=source.error)
    async def invoke():
        with bind_execution_custody_snapshot((ExecutionCustody(GenerationFence()),)):
            return await feature._call_exact_external_ingress_lifecycle(quiesce, name)
    owner = create_owned_task(invoke(), retain_originals=True)
    try:
        result = await await_owned_task(owner)
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=291') == 1
        assert execution_commit_outcome(result.error) == "unknown"
        (record,) = feature._host_ingress_admissions
        assert record.admission.task is owner and owner.done()
        source.assert_retained(record.native_error)
        public = await await_owned_task(create_owned_task(feature.get_config(), retain_originals=True))
        assert execution_commit_outcome(public.error) == "unknown"
        assert not _traceback_locals_reach_any(public.error, (feature, record, source.error, owner, quiesce))
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=291') == 1
    finally:
        await await_owned_task(owner)
        await await_owned_task(create_owned_task(feature.shutdown(), retain_originals=True))


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["transaction", "reset"])
async def test_known_ack_with_actual_private_caller_withdrawal_does_not_poison_next_call(native_pg, tmp_path, monkeypatch, phase):
    backend, observer, schema = native_pg
    records, withdrawals = [], []
    feature = await lifecycle_component(monkeypatch, tmp_path, backend, records)
    quiesce = feature._new_external_ingress_quiesce()
    entered, release = asyncio.Event(), asyncio.Event()
    actual_exit, actual_join = asyncpg.transaction.Transaction.__aexit__, postgres_module.await_owned_task
    held = []
    async def exit(transaction, error_type, *args):
        result = await actual_exit(transaction, error_type, *args)
        if error_type is None and not held:
            held.append(transaction)
            entered.set()
            await release.wait()
        return result
    async def join(*args, **kwargs):
        result = await actual_join(*args, **kwargs)
        if result.cancellation is not None:
            withdrawals.append(result.cancellation)
        return result
    if phase == "transaction":
        monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", exit)
    else:
        actual_reset = asyncpg.pool.PoolAcquireContext.__aexit__
        async def reset(checkout, *args):
            result = await actual_reset(checkout, *args)
            if checkout.pool is backend._pool and not held:
                held.append(checkout)
                entered.set()
                await release.wait()
            return result
        monkeypatch.setattr(asyncpg.pool.PoolAcquireContext, "__aexit__", reset)
    monkeypatch.setattr(postgres_module, "await_owned_task", join)
    async def invoke():
        with bind_execution_custody_snapshot((ExecutionCustody(GenerationFence()),)):
            return await feature._call_exact_external_ingress_lifecycle(quiesce, NAMES[0])
    owner = create_owned_task(invoke(), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        owner.cancel("actual reserved lifecycle caller withdrew")
        await asyncio.sleep(0)
        release.set()
        result = await await_owned_task(owner)
        assert execution_commit_outcome(result.error) == "committed"
        assert records and withdrawals and records[0].cancellation is withdrawals[0]
        assert withdrawals[0] in tuple(_explicit_control_chain(records[0].native_error))
        assert not feature._host_ingress_admissions and feature._host_ingress_public_control is None
        assert vars(feature).get("_event_ingress_control") is None
        second = await await_owned_task(create_owned_task(invoke(), retain_originals=True))
        assert second.error is None and second.result["state"] == "quiesced"
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id IN (291,292)') == 2
    finally:
        release.set()
        await await_owned_task(owner)
        await feature.shutdown()
