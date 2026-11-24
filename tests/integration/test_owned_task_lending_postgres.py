"""Physical PostgreSQL epilogues never require spare host WORK-loan capacity."""

import os

import asyncpg
import pytest

from kestrel_sovereign.async_ownership import bind_owned_task_lender
from kestrel_sovereign.execution_custody import execution_evidence_carriers
from kestrel_sovereign.storage.db.interface import TransactionError
from tests.integration.test_execution_authority_postgres import native_pg as native_pg


class NoSpareWork:
    def __init__(self):
        self.requests = []

    def reserve(self, **request):
        self.requests.append(request)
        raise AssertionError("physical PostgreSQL retirement requested a WORK loan")

    def lend(self, *_args, **_kwargs):
        pytest.fail("capacity-refused retirement was lent")

    def retire(self, *_args, **_kwargs):
        pytest.fail("no WORK loan should have been reserved")


@pytest.mark.asyncio
@pytest.mark.parametrize("rollback", [False, True])
async def test_original_commit_rollback_reset_and_pool_close_without_work_capacity(native_pg, rollback):
    backend, observer, schema = native_pg
    policy = NoSpareWork()
    original = OSError("original body requests rollback")
    with bind_owned_task_lender(policy):
        if rollback:
            with pytest.raises(TransactionError) as caught:
                async with backend.transaction():
                    await backend.execute("INSERT INTO effects VALUES (991, 'original rollback')")
                    raise original
            assert original in execution_evidence_carriers(caught.value)
        else:
            async with backend.transaction():
                await backend.execute("INSERT INTO effects VALUES (991, 'original commit')")
        assert await backend.fetch_one("SELECT count(*) AS count FROM effects WHERE id=991") == (0 if rollback else 1,)
        # from_pool deliberately borrows this fixture-owned pool. Exercise the
        # same original retirement producer used by an owning backend, rather
        # than mistaking borrowed-adapter close for physical pool retirement.
        assert await backend._retire_owned_pool("_pool", backend._pool) is None
    assert not policy.requests
    assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=991') == (0 if rollback else 1)


@pytest.mark.asyncio
async def test_original_advisory_session_close_without_work_capacity(native_pg):
    backend, _, _ = native_pg
    connection = await asyncpg.connect(os.environ["TEST_POSTGRES_URL"])
    policy = NoSpareWork()
    original = OSError("original advisory release failure")
    try:
        with bind_owned_task_lender(policy):
            outcome = await backend._close_failed_advisory_session(connection, original)
        assert connection.is_closed() and outcome is original
        assert not policy.requests
    finally:
        await connection.close()
