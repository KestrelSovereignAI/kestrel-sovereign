"""Exercise the same external effect peer through its actual native stdio transport."""

import asyncio
import os
from pathlib import Path
import sys

import pytest

from kestrel_sovereign._bounded_subprocess import run_bounded_subprocess
pytestmark = [pytest.mark.mcp_integration, pytest.mark.timeout(30)]


@pytest.mark.asyncio
async def test_stdio_permission_peer_writes_only_after_native_tool_dispatch(tmp_path):
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters

    peer = Path(__file__).with_name("mcp_permission_peer.py")
    witness = tmp_path / "stdio-effect-witness"
    parameters = StdioServerParameters(command=sys.executable, args=[
        str(peer), "--transport", "stdio", "--witness", str(witness),
    ], env={"PATH": os.environ["PATH"], "PYTHONUNBUFFERED": "1"})
    async with asyncio.timeout(15), Client(parameters) as client:
        tools = await client.list_tools()
        assert [tool.name for tool in tools.tools] == ["record_marker"]
        assert not witness.exists(), "Discovery executed the external tool"
        result = await client.call_tool("record_marker", {"marker": "owned-stdio-effect"})
        assert not result.is_error
        assert witness.read_text() == "owned-stdio-effect\n"
        assert [part.text for part in result.content] == ["Recorded: owned-stdio-effect"]


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments,message", [
    ([], "Stdio mode requires an explicit witness path"),
    (["--tls-cert", "unused.pem"], "Stdio mode cannot use HTTP/TLS or provider options"),
    (["--ready", "unused.json"], "Stdio mode cannot use HTTP/TLS or provider options"),
    (["--provider-model", "unused"], "Stdio mode cannot use HTTP/TLS or provider options"),
    (["--drop-first-tool-reply"], "Stdio mode cannot use HTTP/TLS or provider options"),
    (["--transport", "streamable-http", "--hold-tool-reply"], "Held tool reply requires stdio transport"),
    (["--transport", "streamable-http", "--linger-after-stdio", "--ready", "unused.json"],
     "Lingering disconnect requires stdio mode"),
    (["--require-profile-mode", "expected-owned-mode"], "Native profile did not supply the required configured mode"),
    (["--transport", "streamable-http", "--require-profile-mode", "unused", "--ready", "unused.json"],
     "Configured mode requires stdio transport"),
])
async def test_stdio_permission_peer_rejects_incompatible_options(tmp_path, arguments, message):
    witness = tmp_path / "refused-stdio-effect"
    supplied = ["--witness", str(witness), *arguments] if arguments else []
    result = await _run_stdio_peer(supplied)
    assert not result.timed_out and not result.stdout_truncated and not result.stderr_truncated
    assert result.returncode == 2 and not result.stdout
    assert message in result.stderr.decode()
    assert not witness.exists()


async def _run_stdio_peer(supplied, *, timeout=15):
    return await run_bounded_subprocess(
        [sys.executable, str(Path(__file__).with_name("mcp_permission_peer.py")),
         "--transport", "stdio", *supplied],
        env={"PATH": os.environ["PATH"]}, timeout=timeout,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_path", ["timeout", "cancellation"])
async def test_stdio_refusal_retires_the_original_child(tmp_path, monkeypatch, exit_path):
    process = None
    created = asyncio.Event()
    original_create = asyncio.create_subprocess_exec

    async def observe_original(*args, **kwargs):
        nonlocal process
        process = await original_create(*args, **kwargs)
        created.set()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", observe_original)
    witness = tmp_path / "owned-timeout-peer"
    witness.write_text("owned test setup\n")
    operation = asyncio.create_task(_run_stdio_peer(
        ["--witness", str(witness), "--linger-after-stdio"],
        timeout=0.2 if exit_path == "timeout" else 15,
    ))
    try:
        if exit_path == "cancellation":
            await asyncio.wait_for(created.wait(), 5)
            operation.cancel()
            with pytest.raises(asyncio.CancelledError):
                await operation
        else:
            result = await operation
            assert result.timed_out
        assert process is not None
        assert process.returncode is not None, "stdio termination abandoned its original child"
    finally:
        # Reconcile only this test's original operation and captured child.
        operation.cancel()
        await asyncio.gather(operation, return_exceptions=True)
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
