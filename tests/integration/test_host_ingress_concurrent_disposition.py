"""A later independent rollback/reset failure cannot weaken retained UNKNOWN."""
import asyncio
from contextvars import ContextVar

import asyncpg
import pytest
from kestrel_sdk.isolated_feature import HostIngressCapabilities

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import ExecutionCustody, bind_execution_custody_snapshot, execution_commit_outcome
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as native_pg
from tests.unit.test_isolated_feature_runtime import _HostIngressClient, _initialized_host_ingress_proxy
from tests.utils.original_failure import OriginalFailure


@pytest.mark.asyncio
async def test_concurrent_native_ingress_keeps_strongest_public_disposition(native_pg, tmp_path, monkeypatch):
    backend, observer, schema = native_pg
    mode = ContextVar("declared_ingress_native_case", default=None)
    entered, release = asyncio.Event(), asyncio.Event()
    lost = OriginalFailure.capture(OSError("actual original COMMIT ACK lost"))
    reset_lost = OriginalFailure.capture(OSError("actual original reset ACK lost"))
    actual_exit = asyncpg.transaction.Transaction.__aexit__
    actual_reset = asyncpg.pool.PoolAcquireContext.__aexit__
    async def exit(transaction, error_type, *args):
        result = await actual_exit(transaction, error_type, *args)
        if mode.get() == "unknown" and error_type is None:
            raise lost.error
        return result
    async def reset(checkout, *args):
        selected = mode.get() == "authority" and checkout.pool is backend._pool
        if selected:
            entered.set()
            await release.wait()
        result = await actual_reset(checkout, *args)
        if selected:
            raise reset_lost.error
        return result
    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", exit)
    monkeypatch.setattr(asyncpg.pool.PoolAcquireContext, "__aexit__", reset)
    calls = []
    class Client(_HostIngressClient):
        async def call_host_ingress(self, name, payload=None):
            calls.append(name)
            token = mode.set(name)
            try:
                if name == "unknown":
                    await backend.execute("INSERT INTO effects VALUES (288, 'actual committed')")
                else:
                    async with backend.transaction():
                        await backend.execute("INSERT INTO effects VALUES (289, 'actual rolled back')")
                        raise OSError("declared ordinary body rollback")
            finally:
                mode.reset(token)
    def factory(**kwargs):
        return Client(ingress_capabilities=HostIngressCapabilities(names=("unknown", "authority")), **kwargs)
    feature, _ = await _initialized_host_ingress_proxy(monkeypatch, tmp_path, factory)
    async def call(name):
        with bind_execution_custody_snapshot((ExecutionCustody(GenerationFence()),)):
            return await feature.call_host_ingress(name)
    authority = create_owned_task(call("authority"), retain_originals=True)
    unknown = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        unknown = create_owned_task(call("unknown"), retain_originals=True)
        first = await await_owned_task(unknown)
        assert execution_commit_outcome(first.error) == "unknown"
        release.set()
        await await_owned_task(authority)
        later = await await_owned_task(create_owned_task(call("unknown"), retain_originals=True))
        assert execution_commit_outcome(later.error) == "unknown", "later failed reset weakened the prior actual UNKNOWN"
        assert len(calls) == 2 and len(feature._host_ingress_admissions) == 2
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=288') == 1
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=289') == 0
    finally:
        release.set()
        await await_owned_task(authority)
        if unknown is not None:
            await await_owned_task(unknown)
        await await_owned_task(create_owned_task(feature.shutdown(), retain_originals=True))
