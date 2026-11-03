"""Native external MCP peer with a real filesystem side-effect witness."""

import argparse
import json
from pathlib import Path
import socket

from mcp import types
from mcp.server import MCPServer
from mcp.server.apps import Apps
import uvicorn


def create_server(witness: Path, *, apps: bool = False):
    """One native effect witness for both permission and Apps campaigns."""
    extension = Apps()

    def record_marker(marker: str) -> str:
        with witness.open("a") as stream:
            stream.write(marker + "\n")
        return "Recorded: " + marker

    if apps:
        def native_result(marker):
            text = record_marker(marker)
            return types.CallToolResult(content=[types.TextContent(type="text", text=text)],
                structured_content={"marker": marker}, meta={"private-witness": marker})

        @extension.tool(resource_uri="ui://witness/app.html", visibility=["model", "app"])
        def open_marker(marker: str) -> types.CallToolResult:
            return native_result(marker)

        @extension.tool(resource_uri="ui://witness/app.html", visibility=["app"])
        def mutate_marker(marker: str) -> types.CallToolResult:
            return native_result(marker)

        extension.add_html_resource("ui://witness/app.html", "<!doctype html><p>Owned native MCP App</p>")

    server = MCPServer("permission-boundary-witness", extensions=[extension] if apps else [])
    server.tool()(record_marker)
    return server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tls-cert")
    parser.add_argument("--tls-key")
    parser.add_argument("--witness")
    parser.add_argument("--ready")
    args = parser.parse_args()
    tls = any((args.tls_cert, args.tls_key, args.witness, args.ready))
    if tls and not all((args.tls_cert, args.tls_key, args.witness, args.ready)):
        parser.error("TLS mode requires certificate, key, witness and ready paths")
    witness = Path(args.witness) if tls else Path("/tmp/permission-witness")
    server = create_server(witness)

    if not tls:
        server.run(transport="streamable-http", host="0.0.0.0", port=8000,
                   stateless_http=True, json_response=True)
        return
    app = server.streamable_http_app(stateless_http=True, json_response=True,
                                    host="remote-permission.example")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        ready = Path(args.ready)
        temporary = ready.with_suffix(".starting")
        temporary.write_text(json.dumps({"port": listener.getsockname()[1]}))
        temporary.replace(ready)
        config = uvicorn.Config(app, ssl_certfile=args.tls_cert, ssl_keyfile=args.tls_key,
                                log_level="error")
        uvicorn.Server(config).run(sockets=[listener])


if __name__ == "__main__":
    main()
