"""Original native assertion commit ACK cannot become projection counters."""
import aiosqlite
import pytest

from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError
from tests.unit.test_execution_custody_review43_native import carries
from tests.unit.test_strategic_memory_ledger_assertions import (
    governed as governed, ledger as ledger, project, seed,
    tenant_identity as tenant_identity,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("carrier", ["error", "group"])
async def test_actual_assertion_commit_preserves_original_projection_control(governed, ledger, monkeypatch, carrier):
    wrapper, raw, _ = governed
    seed(ledger)
    connection = raw.db.backend._connection
    native_commit = aiosqlite.Connection.commit
    original = OSError("native assertion publication acknowledgement")
    original.__cause__ = ExecutionCommitOutcomeError("unknown")
    if carrier == "group":
        original = ExceptionGroup("native assertion publication", [ValueError("ordinary"), original])
    commits = []
    async def lost_ack(target):
        await native_commit(target)
        if target is connection and not commits:
            async with target.execute("SELECT COUNT(*) FROM semantic_assertions") as cursor:
                count = (await cursor.fetchone())[0]
            if count:
                commits.append(True)
                raise original
    monkeypatch.setattr(aiosqlite.Connection, "commit", lost_ack)
    try:
        with pytest.raises(BaseException) as caught:
            await project(wrapper, ledger)
        assert carries(caught.value, original)
        assert commits == [True]
        assert await raw.db.fetchval("SELECT COUNT(*) FROM semantic_assertions") == 1
    finally:
        monkeypatch.undo()
