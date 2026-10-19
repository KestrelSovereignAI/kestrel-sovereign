"""Native external MCP peer with a real filesystem side-effect witness."""

import argparse
import json
from pathlib import Path
import socket

from mcp.server import MCPServer
import uvicorn


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
    server = MCPServer("permission-boundary-witness")

    @server.tool()
    def record_marker(marker: str) -> str:
        with witness.open("a") as stream:
            stream.write(marker + "\n")
        return "Recorded: " + marker

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
