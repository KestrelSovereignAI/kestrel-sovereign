"""Bounded actual memory work, declared retired-lease callbacks, repeated reads."""

import gc
import weakref

import pytest

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionCommitOutcomeError,
    execution_commit_outcome,
    is_execution_control_error,
)
from tests.integration.native_memory_original_control import (
    native_memory_owners as native_memory_owners,
)
from tests.integration.test_execution_authority_postgres import native_pg as native_pg


@pytest.mark.asyncio
@pytest.mark.parametrize("reader", ["retrieve", "drain"])
async def test_original_memory_receipt_keeps_one_payload_and_bounded_reader_frames(
    native_memory_owners, monkeypatch, reader
):
    n = native_memory_owners
    retriever = n.sqlite_memory.retriever
    db = n.sqlite_saved.db
    await db.execute(
        "WITH RECURSIVE seq(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM seq WHERE n<40) "
        "INSERT INTO conversation_history (id,agent_id,role,content,metadata) "
        "SELECT n,?,'assistant','actual native rehearsal','{}' FROM seq",
        (n.did,),
    )
    actual = retriever.update_access
    references = []

    class Payload:
        pass

    async def actual_write_then_declared_native_callback(message_id, agent_id):
        payload = Payload()
        references.append(weakref.ref(payload))
        await actual(message_id, agent_id)
        # Forward real SQLite work first. The original retired PG lease
        # supplies a declared callback control input, not writer ACK loss.
        try:
            n.lease.require_live()
        except BaseException as error:
            if message_id == 1:
                # Declared legacy UNKNOWN input around a real original
                # retired-lease denial, not a fabricated driver ACK loss.
                raise ExecutionCommitOutcomeError("unknown") from error
            raise

    monkeypatch.setattr(
        retriever, "update_access", actual_write_then_declared_native_callback
    )
    tasks = [retriever._schedule_access_update(row, n.did) for row in range(1, 41)]
    try:
        for task in tasks:
            await await_owned_task(task)
        tasks.clear()
        del task
        gc.collect()
        assert len(references) == 40
        assert sum(reference() is not None for reference in references) == 1
        assert not retriever._access_update_tasks
        assert (
            await db.fetchval(
                "SELECT COUNT(*) FROM conversation_history WHERE json_extract(metadata, '$.access_count')=1"
            )
            == 40
        )
        control = retriever._access_update_control
        assert is_execution_control_error(control)
        assert execution_commit_outcome(control) == "unknown"

        def depth():
            result, traceback = 0, control.__traceback__
            while traceback is not None:
                result += 1
                traceback = traceback.tb_next
            return result

        depths = []
        for _ in range(40):
            try:
                if reader == "retrieve":
                    await retriever.retrieve("retained original", n.did, read_only=True)
                else:
                    await retriever.drain_access_updates()
            except BaseException as error:
                assert error is control
            else:
                pytest.fail("retained original control became an ordinary result")
            depths.append(depth())
        assert max(depths) <= depths[0] + 2, (
            f"{reader} retained unbounded new consumer frames: {depths}"
        )
    finally:
        for task in tasks:
            await await_owned_task(task)
