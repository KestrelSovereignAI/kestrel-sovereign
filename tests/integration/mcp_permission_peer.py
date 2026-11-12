"""Native external MCP peer with a real filesystem side-effect witness."""

import argparse
import asyncio
import json
import re
import os
from pathlib import Path
import socket
import signal
import threading

from mcp import types
from mcp.server import MCPServer
from mcp.server.apps import Apps
import uvicorn
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from kestrel_sovereign.security.input_guardrails import extract_raw_user_content


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
                "message_shapes": [{"role": row['role'],
                    "content_type": type(row.get('content')).__name__,
                    "markers": [marker for marker in ['A2A_TASK_SUBMITTED', 'A2A_COMPLETE',
                        'Record the owned marker'] if marker in str(row.get('content'))]}
                    for row in body['messages']],
                "has_response_format": bool(body.get("response_format"))}) + "\n")
        if body.get("model") != model:
            return JSONResponse({"error": "Owned provider requires its configured model"}, status_code=400)
        calls += 1
        tools = [tool["function"] for tool in body.get("tools", [])]
        candidates = [tool for tool in tools if tool["name"].endswith("record_marker")]
        finished = body["messages"][-1]["role"] == "tool"
        message = {"role": "assistant", "content": "Owned provider response complete"}
        reason = "stop"
        selected = None
        arguments = {"marker": "owned-provider-effect"}
        if candidates and len(candidates) != 1:
            return JSONResponse({"error": "Owned provider requires one marker tool"}, status_code=400)
        # A finite model script consumes the actual cognition prompt and tool
        # schemas. Task completion still runs through the real governed tool.
        prompt = extract_raw_user_content(next((m['content'] for m in reversed(body['messages'])
            if m['role'] == 'user' and isinstance(m.get('content'), str)
            and not (m['content'].startswith('<operator_notice>\n')
                and m['content'].endswith('\n</operator_notice>'))), ''))
        incoming = re.search(r"\[A2A_TASK_SUBMITTED\] Peer `[^`]+` submitted native A2A task `([^`]+)`", prompt)
        if incoming:
            replies = [tool for tool in tools if tool['name'].endswith('respond_to_a2a_task')]
            if len(candidates) != 1 or len(replies) != 1:
                return JSONResponse({"error": "Owned A2A model requires the original marker and task response tools"}, status_code=400)
            called = {call['function']['name'] for m in body['messages']
                for call in m.get('tool_calls', [])}
            if candidates[0]['name'] not in called:
                selected = candidates[0]
            elif replies[0]['name'] not in called:
                selected = replies[0]
                arguments = {"task_id": incoming.group(1),
                    "content": "Owned permission evaluation complete", "state": "completed"}
            else:
                selected = None
        elif (prompt.strip() == 'Record the owned marker using the connected MCP tool.'
                and len(candidates) == 1 and not finished):
            selected = candidates[0]
        if selected is not None:
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": f"owned-provider-call-{calls}", "type": "function", "function": {
                    "name": selected["name"], "arguments": json.dumps(arguments),
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


def create_server(witness: Path, *, apps: bool = False, hold_tool_reply: bool = False,
                  operator_input: str | None = None, input_message: str = 'Private city?',
                  input_tool: str | None = None, app_html: str | None = None, input_rounds: int = 1,
                  input_schema: dict | None = None, input_content: dict | None = None,
                  input_barrier: tuple[asyncio.Event, asyncio.Event] | None = None):
    """One native effect witness for both permission and Apps campaigns."""
    extension = Apps()

    def record_marker(marker: str) -> str:
        if hold_tool_reply:
            # Publish barrier admission before the effect, so observing the
            # effect proves this same native invocation cannot return yet.
            with Path(str(witness) + ".held").open("a") as stream:
                stream.write(marker + "\n")
        with witness.open("a") as stream:
            stream.write(marker + "\n")
        if hold_tool_reply:
            # This owned failure-injection peer is retired by its real
            # container lifecycle. There is deliberately no successful reply.
            threading.Event().wait()
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

        extension.add_html_resource("ui://witness/app.html", app_html if app_html is not None
            else "<!doctype html><p>Owned native MCP App</p>")

    server = MCPServer("permission-boundary-witness", extensions=[extension] if apps else [])
    server.tool()(record_marker)
    if operator_input is not None:
        if type(input_rounds) is not int or not 1 <= input_rounds <= 4:
            raise ValueError('Owned MCP input rounds must be bounded')
        if operator_input not in {'form', 'url', 'mixed'}:
            raise ValueError('Unknown owned MCP input scenario')
        first_input_held = False
        async def operator_round(ctx, call_next):
            nonlocal first_input_held
            if ctx.method != 'tools/call' or (input_tool is not None and ctx.params.get('name') != input_tool):
                return await call_next(ctx)
            with Path(str(witness) + '.input-wire').open('a') as stream:
                stream.write(json.dumps(dict(ctx.params)) + '\n')
            responses = ctx.params.get('inputResponses')
            if not responses and input_barrier is not None and not first_input_held:
                first_input_held = True
                input_barrier[0].set()
                await input_barrier[1].wait()
            round_index = 0
            if responses:
                state = ctx.params['requestState']
                round_index = 1 if state == 'server-owned-state' else int(state.removeprefix('server-owned-state-'))
                assert 1 <= round_index <= input_rounds
                if 'private-url-key' in responses:
                    assert responses['private-url-key']['action'] == 'decline'
                form = responses.get('private-form-key')
                if form is None or form['action'] != 'accept':
                    return types.CallToolResult(is_error=True,
                        content=[types.TextContent(type='text', text='Operator input declined')])
                assert form['content'] == (input_content if input_content is not None else {'city': 'owned-private-answer'})
            if round_index < input_rounds:
                inputs = {}
                if operator_input in {'form', 'mixed'}:
                    inputs['private-form-key'] = types.ElicitRequest(params=types.ElicitRequestFormParams(
                        message=input_message, requested_schema=(input_schema if input_schema is not None
                            else {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']})))
                if operator_input in {'url', 'mixed'}:
                    inputs['private-url-key'] = types.ElicitRequest(params=types.ElicitRequestURLParams(
                        message='Private URL consent?', elicitation_id='owned-consent',
                        url='https://issuer.input.test/authorize'))
                return types.InputRequiredResult(input_requests=inputs, request_state=(
                    'server-owned-state' if round_index == 0 else f'server-owned-state-{round_index + 1}'))
            return await call_next(ctx)
        server.middleware.append(operator_round)
    return server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="streamable-http")
    parser.add_argument("--tls-cert")
    parser.add_argument("--tls-key")
    parser.add_argument("--witness")
    parser.add_argument("--ready")
    parser.add_argument("--provider-model")
    parser.add_argument("--provider-fault", choices=["wrong_choice"])
    parser.add_argument('--operator-input', choices=['form', 'url', 'mixed'])
    parser.add_argument("--drop-first-tool-reply", action="store_true")
    parser.add_argument("--linger-after-stdio", action="store_true")
    parser.add_argument("--hold-tool-reply", action="store_true")
    parser.add_argument("--require-profile-mode")
    args = parser.parse_args()
    if args.transport == "stdio":
        if not args.witness:
            parser.error("Stdio mode requires an explicit witness path")
        if any((args.tls_cert, args.tls_key, args.ready, args.provider_model,
                args.provider_fault, args.drop_first_tool_reply)):
            parser.error("Stdio mode cannot use HTTP/TLS or provider options")
        if args.require_profile_mode is not None and os.environ.get("OWNED_PROFILE_MODE") != args.require_profile_mode:
            parser.error("Native profile did not supply the required configured mode")
        if args.linger_after_stdio:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        create_server(Path(args.witness), hold_tool_reply=args.hold_tool_reply,
            operator_input=args.operator_input).run(transport="stdio")
        if args.linger_after_stdio and Path(args.witness).exists():
            while True:
                signal.pause()
        return
    if args.hold_tool_reply:
        parser.error("Held tool reply requires stdio transport")
    if args.linger_after_stdio:
        parser.error("Lingering disconnect requires stdio mode")
    if args.require_profile_mode is not None:
        parser.error("Configured mode requires stdio transport")
    tls = any((args.tls_cert, args.tls_key, args.witness, args.ready))
    if tls and not all((args.tls_cert, args.tls_key, args.witness, args.ready)):
        parser.error("TLS mode requires certificate, key, witness and ready paths")
    if args.provider_model and not tls:
        parser.error("The owned provider requires TLS mode")
    witness = Path(args.witness) if tls else Path("/tmp/permission-witness")
    server = create_server(witness, operator_input=args.operator_input)

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
