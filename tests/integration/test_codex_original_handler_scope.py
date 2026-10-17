"""Real stdio multiplexing localizes only its original admitted PG scopes.

The stdio peer only echoes protocol replies; no Codex binary, credentials,
model requests or remote provider is used. Actual PG leases produce denials.
"""

import asyncio
import json
import sys

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    _explicit_control_chain,
    bind_execution_custody_snapshot,
    execution_terminal_error,
)
from kestrel_sovereign.llm.codex_app_server import CodexAppServerClient
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as native_pg


@pytest.mark.asyncio
@pytest.mark.parametrize("denied", ["own", "foreign", "mixed"])
@pytest.mark.parametrize("retirement", ["public", "cancel", "join"])
async def test_codex_registered_callback_compares_actual_original_scope(
    native_pg, denied, retirement
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
            owner = ExecutionCustody(GenerationFence(lease=own_lease))
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
