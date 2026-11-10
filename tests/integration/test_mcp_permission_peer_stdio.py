"""Exercise the same external effect peer through its actual native stdio transport."""

import asyncio
import os
from pathlib import Path
import sys

import pytest
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
    process = await asyncio.create_subprocess_exec(
        sys.executable, str(Path(__file__).with_name("mcp_permission_peer.py")),
        "--transport", "stdio", *supplied,
        env={"PATH": os.environ["PATH"]},
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    async with asyncio.timeout(15):
        stdout, stderr = await process.communicate()
    assert process.returncode == 2 and not stdout
    assert message in stderr.decode()
    assert not witness.exists()
