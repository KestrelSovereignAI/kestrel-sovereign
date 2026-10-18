"""Real stdio multiplexing localizes only its original admitted PG scopes.

The stdio peer only echoes protocol replies; no Codex binary, credentials,
model requests or remote provider is used. Actual PG leases produce denials.
"""

import asyncio
import json
import sys

import asyncpg
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    _explicit_control_chain,
    bind_execution_custody_snapshot,
    execution_cleanup_debts,
    execution_commit_outcome,
    execution_denial_scopes,
    execution_terminal_error,
)
from kestrel_sovereign.llm.codex_app_server import CodexAppServerClient
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as native_pg


@pytest.mark.asyncio
@pytest.mark.parametrize("denied", ["own", "foreign", "mixed", "committed"])
@pytest.mark.parametrize("retirement", ["public", "cancel", "join"])
async def test_codex_registered_callback_compares_actual_original_scope(
    native_pg, monkeypatch, denied, retirement
):
    backend, _, _ = native_pg
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-u", "-c",
        "import sys\nfor line in sys.stdin:\n sys.stdout.write(line); sys.stdout.flush()",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    client = CodexAppServerClient(binary="owned-protocol-echo-only")
    client._proc, client._initialized = process, True
    queues = {key: asyncio.Queue() for key in ("original", "other")}
    client._turn_sinks.update(queues)
    entered, release = asyncio.Event(), asyncio.Event()
    cancellation_ready = asyncio.Event()
    originals, carried = [], []

    async def callback(params):
        entered.set()
        await release.wait()
        if denied == "committed":
            original_exit = asyncpg.transaction.Transaction.__aexit__

            async def commit_then_retire(transaction, error_type, error, traceback):
                result = await original_exit(transaction, error_type, error, traceback)
                if error is None and not transaction._nested:
                    owner.revoke("actual original callback scope retired after native COMMIT")
                return result

            monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", commit_then_retire)
            try:
                with bind_execution_custody_snapshot((owner,)):
                    await backend.execute("INSERT INTO effects VALUES (178, 'original callback committed effect')")
            except BaseException as error:
                # The SAME original native receipt has physically retired with
                # its checkout. It is still commit control, not a pure refusal.
                assert execution_commit_outcome(error) == "committed"
                assert execution_denial_scopes(error) == (owner,)
                assert not execution_cleanup_debts(error)
                originals.append(error)
            assert await native_pg[1].fetchval(f'SELECT count(*) FROM "{native_pg[2]}".effects') == 1
        else:
            for scope in ((owner,) if denied == "own" else (foreign,) if denied == "foreign" else (owner, foreign)):
                try:
                    scope.require_work()
                except BaseException as error:
                    originals.append(error)
        failure = execution_terminal_error(*originals, retain_originals=True, isolate_control_originals=True)
        if retirement != "public":
            if retirement == "cancel":
                asyncio.current_task().cancel("actual original callback withdrawal")
            try:
                if retirement == "join":
                    cancellation_ready.set()
                    await asyncio.Event().wait()
                else:
                    await asyncio.sleep(0)
            except asyncio.CancelledError as cancellation:
                cancellation.__cause__ = failure
                carried.append(cancellation)
                raise
        carried.append(failure)
        raise failure

    async def other_callback(params):
        return {"success": True}

    tasks = ()
    try:
        async with backend.advisory_locks([(3569, 178221)]) as own_lease:
            owner = ExecutionCustody(GenerationFence(lease=None if denied == "committed" else own_lease))
            with bind_execution_custody_snapshot((owner,)):
                client.register_server_request_handler("item/tool/call", callback, thread_id="original")
        async with backend.advisory_locks([(3569, 178222)]) as foreign_lease:
            foreign = ExecutionCustody(GenerationFence(lease=foreign_lease))
        # Use successive sessions: this fixture deliberately has ONE advisory
        # slot. Both native leases are retired before the callback observes them.
        client.register_server_request_handler("item/tool/call", other_callback, thread_id="other")
        client._dispatch({"id": 17822, "method": "item/tool/call", "params": {"threadId": "original"}})
        tasks = tuple(client._server_request_tasks)
        await asyncio.wait_for(entered.wait(), 5)
        release.set()
        join_failure = None
        if retirement == "join":
            await asyncio.wait_for(cancellation_ready.wait(), 5)
            try:
                await client._join_server_requests()
            except BaseException as error:
                join_failure = error
        for task in tasks:
            await await_owned_task(task)
        assert originals and carried
        if denied == "own":
            assert client._server_request_failure is None
            assert join_failure is None
            assert client._closed_error is None and queues["other"].empty()
            await client.ensure_started()
            client._dispatch({"id": 17823, "method": "item/tool/call", "params": {"threadId": "other"}})
            for task in tuple(client._server_request_tasks):
                await await_owned_task(task)
            # A cancelled callback does not synthesize an ordinary RPC reply.
            reply = json.loads(await asyncio.wait_for(process.stdout.readline(), 5))
            if reply["id"] == 17822:
                reply = json.loads(await asyncio.wait_for(process.stdout.readline(), 5))
            assert reply["id"] == 17823 and reply["result"]["success"]
            assert process.returncode is None
        else:
            failure = client._server_request_failure
            assert failure is not None, (
                "registered callback localized a foreign or mixed original PG denial"
            )
            chain = tuple(_explicit_control_chain(failure))
            assert carried[0] in chain and all(error in chain for error in originals)
            with pytest.raises(BaseException) as repeated:
                await client.ensure_started()
            assert repeated.value is failure
            assert client._closed_error is failure
            if retirement == "join":
                assert join_failure is failure
    finally:
        release.set()
        for task in tasks:
            await await_owned_task(task)
        await await_owned_task(create_owned_task(client.aclose(), retain_originals=True))
        assert process.returncode is not None
