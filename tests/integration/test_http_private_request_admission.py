"""Request extensions describe transport state; they never grant authority."""
import asyncio

import aiosqlite
import httpx
import httpx2
import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ExecutionCustody, ProcessRuntimeExecutionFence,
    bind_execution_custody_snapshot, bind_native_execution_admission,
    retain_native_execution_http_client_close, is_execution_control_error,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("http", [httpx, httpx2], ids=["httpx1", "httpx2"])
@pytest.mark.parametrize("mutation", ["clear-all", "replace-validator", "replace-custody", "nested-send"])
async def test_request_observer_cannot_replace_private_native_admission(tmp_path, http, mutation):
    async with aiosqlite.connect(tmp_path / "original-policy.db") as ledger:
        await ledger.execute("CREATE TABLE policy(epoch INTEGER)")
        await ledger.execute("INSERT INTO policy VALUES(1)")
        await ledger.execute("CREATE TABLE effects(id INTEGER)")
        await ledger.commit()
        peers, bodies = set(), []

        async def peer(reader, writer):
            task = asyncio.current_task()
            peers.add(task)
            try:
                await reader.readuntil(b"\r\n\r\n")
                bodies.append(await reader.readexactly(1))
                await ledger.execute("INSERT INTO effects VALUES(1)")
                await ledger.commit()
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK")
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()
                peers.discard(task)

        server = await asyncio.start_server(peer, "127.0.0.1", 0)
        scope = ExecutionCustody(ProcessRuntimeExecutionFence("original-http-request"))

        async def policy():
            if await (await ledger.execute("SELECT epoch FROM policy")).fetchone() != (1,):
                raise ExecutionAuthorityError("original SQL policy retired")

        async def permissive_replacement():
            return None

        async def observer(request):
            if request.url.path == "/nested":
                return
            if mutation == "nested-send":
                # The same client completes a distinct inner send before the
                # outer observer retires policy. Reset to the outer ORIGINAL,
                # never the inner send or an observer-writable extension.
                await client.post(f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/nested", content=b"n")
            if mutation == "replace-custody":
                scope.revoke("original HTTP scope retired during request observer")
                request.extensions["kestrel.execution_custody"] = ()
            else:
                await ledger.execute("UPDATE policy SET epoch=2")
                await ledger.commit()
                if mutation in {"clear-all", "nested-send"}:
                    request.extensions.clear()
                else:
                    request.extensions["kestrel.native_admission"] = (permissive_replacement,)

        client = http.AsyncClient(event_hooks={"request": [observer]}, timeout=2)
        retain_native_execution_http_client_close(client)
        try:
            terminal = None
            with bind_execution_custody_snapshot((scope,)), bind_native_execution_admission(policy):
                try:
                    await client.post(f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/", content=b"x")
                except BaseException as error:
                    terminal = error
            expected = [b"n"] if mutation == "nested-send" else []
            assert bodies == expected, "mutable request metadata authorized an actual remote effect"
            assert await (await ledger.execute("SELECT COUNT(*) FROM effects")).fetchone() == (len(expected),)
            assert is_execution_control_error(terminal), "original private admission did not deny dispatch"
        finally:
            await client.aclose()
            server.close()
            await server.wait_closed()
            if peers:
                await asyncio.gather(*tuple(peers))
