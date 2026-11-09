"""Native external MCP peer with a real filesystem side-effect witness."""

import argparse
import json
from pathlib import Path
import socket

from mcp import types
from mcp.server import MCPServer
from mcp.server.apps import Apps
import uvicorn
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route


class DropFirstToolReply:
    """Lose one real native reply after the external witness was written."""

    def __init__(self, app, witness: Path):
        self.app, self.witness, self.dropped = app, witness, False

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['method'] != 'POST' or scope['path'] != '/mcp':
            return await self.app(scope, receive, send)
        messages = []
        while True:
            message = await receive()
            assert message['type'] == 'http.request'
            messages.append(message)
            if not message.get('more_body', False):
                break
        payload = json.loads(b''.join(message.get('body', b'') for message in messages))
        index = 0

        async def replay_receive():
            nonlocal index
            if index < len(messages):
                message = messages[index]
                index += 1
                return message
            return await receive()

        async def lose_reply(message):
            if (payload.get('method') == 'tools/call' and not self.dropped
                    and message['type'] == 'http.response.body'):
                assert self.witness.exists() and self.witness.read_text().splitlines()
                self.dropped = True
                raise ConnectionResetError('Owned MCP reply lost after its real external effect')
            await send(message)

        await self.app(scope, replay_receive, lose_reply)


def add_owned_provider_routes(app, model: str, request_witness: Path, *, fault: str | None = None):
    """Serve finite native provider responses beside the owned MCP witness."""
    calls = 0

    async def models(request):
        with request_witness.open("a") as stream:
            stream.write(json.dumps({"path": request.url.path}) + "\n")
        return JSONResponse({"object": "list", "data": [
            {"id": model, "object": "model", "created": 0, "owned_by": "owned-test"},
        ]})

    async def completion(request):
        nonlocal calls
        body = await request.json()
        with request_witness.open("a") as stream:
            stream.write(json.dumps({"path": request.url.path, "model": body.get("model"),
                "tools": [tool["function"]["name"] for tool in body.get("tools", [])],
                "last_role": body["messages"][-1]["role"],
                "has_response_format": bool(body.get("response_format"))}) + "\n")
        if body.get("model") != model:
            return JSONResponse({"error": "Owned provider requires its configured model"}, status_code=400)
        calls += 1
        tools = [tool["function"] for tool in body.get("tools", [])]
        candidates = [tool for tool in tools if tool["name"].endswith("record_marker")]
        finished = body["messages"][-1]["role"] == "tool"
        message = {"role": "assistant", "content": "Owned provider response complete"}
        reason = "stop"
        if candidates and not finished:
            if len(candidates) != 1:
                return JSONResponse({"error": "Owned provider requires one marker tool"}, status_code=400)
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": f"owned-provider-call-{calls}", "type": "function", "function": {
                    "name": candidates[0]["name"],
                    "arguments": json.dumps({"marker": "owned-provider-effect"}),
                },
            }]}
            reason = "tool_calls"
        response = {
            "id": f"owned-provider-response-{calls}", "object": "chat.completion",
            "created": 0, "model": model,
            "choices": [{"index": 1 if fault == "wrong_choice" and candidates and not finished else 0,
                         "finish_reason": reason, "message": message}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        if body.get("stream"):
            delta = dict(message)
            if "tool_calls" in delta:
                delta["tool_calls"] = [{"index": index, **call} for index, call in enumerate(delta["tool_calls"])]
            response["object"] = "chat.completion.chunk"
            response["choices"][0]["delta"] = delta
            del response["choices"][0]["message"]
            usage = response.pop("usage")

            async def events():
                yield "data: " + json.dumps(response) + "\n\n"
                yield "data: " + json.dumps({**response, "choices": [], "usage": usage}) + "\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")
        return JSONResponse(response)

    app.routes.extend([
        Route("/v1/models", models, methods=["GET"]),
        Route("/v1/chat/completions", completion, methods=["POST"]),
    ])


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
    parser.add_argument("--provider-model")
    parser.add_argument("--provider-fault", choices=["wrong_choice"])
    parser.add_argument("--drop-first-tool-reply", action="store_true")
    args = parser.parse_args()
    tls = any((args.tls_cert, args.tls_key, args.witness, args.ready))
    if tls and not all((args.tls_cert, args.tls_key, args.witness, args.ready)):
        parser.error("TLS mode requires certificate, key, witness and ready paths")
    if args.provider_model and not tls:
        parser.error("The owned provider requires TLS mode")
    witness = Path(args.witness) if tls else Path("/tmp/permission-witness")
    server = create_server(witness)

    if not tls:
        server.run(transport="streamable-http", host="0.0.0.0", port=8000,
                   stateless_http=True, json_response=True)
        return
    app = server.streamable_http_app(stateless_http=True, json_response=True,
                                    host="remote-permission.example")
    if args.provider_model:
        add_owned_provider_routes(app, args.provider_model, witness.with_suffix(".provider.jsonl"), fault=args.provider_fault)
    if args.drop_first_tool_reply:
        app = DropFirstToolReply(app, witness)
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
