"""Original scoped tool refusal must not poison a physically live multiplexer."""
import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ExecutionCommitOutcomeError,
    ExecutionCustody, ProcessRuntimeExecutionFence,
    bind_execution_custody_snapshot, execution_denial_scopes,
    require_execution_work,
)
from kestrel_sovereign.llm.codex_app_server import CodexAppServerClient
from kestrel_sovereign.turn_scope import capture_turn_scope


@pytest.mark.asyncio
@pytest.mark.parametrize("receipt", ["scoped", "unscoped", "committed", "unknown"])
async def test_native_codex_tool_refusal_keeps_original_turn_and_connection_boundaries(receipt):
    # A real owned stdio peer; no Codex binary, auth file or provider is used.
    process = await asyncio.create_subprocess_exec(sys.executable, "-u", "-c",
        "import sys\nfor line in sys.stdin:\n sys.stdout.write(line); sys.stdout.flush()",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)
    client = CodexAppServerClient(binary="offline-owned-stdio")
    client._proc, client._initialized = process, True
    original = ExecutionCustody(ProcessRuntimeExecutionFence("original-tool-occurrence"))
    with bind_execution_custody_snapshot((original,)):
        captured = capture_turn_scope(SimpleNamespace())
    entered, release = asyncio.Event(), asyncio.Event()
    queues = {turn: asyncio.Queue() for turn in ("original", "other")}
    client._turn_sinks.update(queues)

    async def original_tool(params):
        with captured.bind():
            entered.set()
            await release.wait()
            if receipt == "scoped":
                require_execution_work()
            if receipt in {"committed", "unknown"}:
                raise ExecutionCommitOutcomeError(receipt)
            raise ExecutionAuthorityError("original unscoped connection control")

    async def other_tool(params):
        return {"success": True, "original_other_turn": True}

    with captured.bind():
        client.register_server_request_handler("item/tool/call", original_tool, thread_id="original")
    client.register_server_request_handler("item/tool/call", other_tool, thread_id="other")
    client._dispatch({"id": 67, "method": "item/tool/call", "params": {"threadId": "original"}})
    tasks = tuple(client._server_request_tasks)
    try:
        await asyncio.wait_for(entered.wait(), 2)
        original.revoke("only original tool occurrence retired while awaiting approval")
        release.set()
        for task in tasks:
            await await_owned_task(task)
        reply = json.loads(await asyncio.wait_for(process.stdout.readline(), 2))
        assert reply["id"] == 67 and "error" in reply
        denied = await queues["original"].get()
        failure = denied["__bridge_error__"]
        if receipt == "scoped":
            assert execution_denial_scopes(failure) == (original,)
            assert client._server_request_failure is None, "one turn's refusal poisoned the shared connection"
            assert client._closed_error is None and queues["other"].empty()
            await client.ensure_started()
            client._dispatch({"id": 68, "method": "item/tool/call", "params": {"threadId": "other"}})
            for task in tuple(client._server_request_tasks):
                await await_owned_task(task)
            reply = json.loads(await asyncio.wait_for(process.stdout.readline(), 2))
            assert reply["id"] == 68 and reply["result"]["original_other_turn"] is True
            assert process.returncode is None
        else:
            assert client._server_request_failure is failure
            with pytest.raises(BaseException) as repeated:
                await client.ensure_started()
            assert repeated.value is failure
    finally:
        release.set()
        for task in tuple(client._server_request_tasks):
            await await_owned_task(task)
        closed = await await_owned_task(create_owned_task(client.aclose()))
        if receipt != "scoped":
            assert closed.error is client._server_request_failure
        assert process.returncode is not None
