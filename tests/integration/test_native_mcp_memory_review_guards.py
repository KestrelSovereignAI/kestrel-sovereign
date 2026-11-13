"""Reproduce whole-review findings on real native memory and HTTP owners."""

import asyncio
from contextlib import ExitStack

import pytest
from kestrel_sdk.hooks.base import Hook, HookEvent, HookOutput
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.server.mcpserver import Context

from kestrel_sovereign.agent.invocation import bind_async_invocation
from kestrel_sovereign.auth import (
    CallerContext,
    caller_context_binding_scope,
    caller_context_lifetime,
    capture_caller_context_binding,
    current_caller_context,
)
from kestrel_sovereign.mcp_server.policy import MCPServerPolicy
from kestrel_sovereign.mcp_server.publication import NativeMCPPublication
from kestrel_sovereign.mcp_server.tool_caller import bind_native_mcp_tool_caller
from tests.integration.test_native_mcp_memory_tools import observe_native_call
from tests.integration.test_native_mcp_publication import KEY
from tests.integration.test_native_mcp_publication import (
    assistant_agent as _assistant_agent,
)
from tests.integration.test_native_mcp_publication import publication as _publication
from tests.integration.test_native_mcp_tool_context import issue_native_tokens

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]
assistant_agent = _assistant_agent
publication = _publication


@pytest.mark.parametrize("assistant_agent", [{"features": ["MemoryFeature"]}], indirect=True)
async def test_native_memory_bounds_the_complete_emitted_result(publication):
    owner, http = publication
    private = "owned-double-serialized-memory-" + "x" * 40000
    await owner.agent.privacy_agent.add_conversation("user", private, session_id="owned-large-result")
    owner.publish()
    token = (await issue_native_tokens(owner, http))["access_token"]
    result, failure, responses = await observe_native_call(owner, token, "recall_recent",
        {"limit": 5, "operation_id": "owned-full-result-budget", "conversation_id": "owned-client"})
    assert all(len(body) <= owner.policy.max_result_bytes for _status, body in responses), (
        "Native memory emitted a result larger than configured max_result_bytes")
    assert any(status == 500 for status, _body in responses), (
        "Native memory began delivery before complete result budget refusal")
    assert result is None and failure is not None
    assert all(private.encode() not in body for _status, body in responses)


@pytest.mark.parametrize("assistant_agent", [{"features": ["MemoryFeature"]}], indirect=True)
async def test_retired_legacy_http_revokes_original_governed_hook_caller(publication):
    owner, http = publication
    hook_entered, release_hook, hook_ended = asyncio.Event(), asyncio.Event(), asyncio.Event()
    release_child = asyncio.Event()
    requests, children = [], []
    task_key = "owned.native.memory.original_http_task"

    class OriginalHTTPTask:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            if scope["type"] == "http":
                scope[task_key] = asyncio.current_task()
            await self.app(scope, receive, send)

    class OriginalGovernedHook(Hook):
        def __init__(self):
            super().__init__("owned-native-original-caller", [HookEvent.PRE_TOOL_USE],
                matcher="^recall_recent$", priority=1, awaits_user_input=True)

        async def execute(self, input):
            assert current_caller_context() is not None
            assert len(owner._tool_callers) == 1
            requests.append(next(iter(owner._tool_callers)))

            async def child():
                await release_child.wait()
                return current_caller_context()

            children.append(asyncio.create_task(child()))
            hook_entered.set()
            try:
                await release_hook.wait()
            finally:
                hook_ended.set()
            return HookOutput.allow()

    hook = OriginalGovernedHook()
    await owner.agent.hooks_manager.register_async(hook)
    owner.app.add_middleware(OriginalHTTPTask)
    owner.publish()
    token = (await issue_native_tokens(owner, http))["access_token"]
    http.headers.update({"authorization": "Bearer " + token})

    async def call():
        async with Client(streamable_http_client(owner.resource, http_client=http), mode="legacy") as client:
            return await client.call_tool("recall_recent", {"limit": 5,
                "operation_id": "owned-original-http-cancellation", "conversation_id": "owned-client"})

    original = asyncio.create_task(call())
    entered = asyncio.create_task(hook_entered.wait())
    try:
        await asyncio.wait({original, entered}, timeout=5, return_when=asyncio.FIRST_COMPLETED)
        assert hook_entered.is_set(), "Actual legacy MCP governed hook was not entered"
        request = requests[0]
        original_http = request.scope[task_key]
        assert original_http is not asyncio.current_task() and original_http is not original
        original_http.cancel()
        await asyncio.wait({original_http}, timeout=5)
        original_retired = original_http.done() and not request.active and not owner._tool_callers
        retained_hook = not hook_ended.is_set()
        release_child.set()
        copied = await asyncio.gather(*children)
    finally:
        release_child.set()
        release_hook.set()
        entered.cancel()
        if not original.done():
            original.cancel()
        await asyncio.gather(original, entered, *children, return_exceptions=True)
        owner.agent.hooks_manager.unregister(hook)
    assert original_retired, "Original native HTTP owner did not physically retire"
    assert retained_hook, "Probe did not retain the actual governed hook after HTTP retirement"
    assert copied == [None], "Legacy governed memory hook retained OAuth caller authority after original HTTP retirement"


@pytest.mark.parametrize("binding_kind", ["active", "retired", "none", "conflict"])
async def test_actual_invocation_borrows_only_its_original_caller_binding(assistant_agent, binding_kind):
    caller = CallerContext.authenticated("owned-original-invocation-caller")
    entered = []

    @bind_async_invocation("operation_id")
    async def original(agent, *, operation_id, caller_binding, caller=None):
        entered.append((capture_caller_context_binding(), current_caller_context()))
        return "original-invocation-result"

    with ExitStack() as original_lifetime:
        binding = original_lifetime.enter_context(caller_context_lifetime(caller))
        if binding_kind == "retired":
            original_lifetime.close()
        with caller_context_binding_scope(binding):
            if binding_kind == "conflict":
                with pytest.raises(TypeError, match="one original caller binding or caller"):
                    await original(assistant_agent, operation_id="owned-conflicting-bindings",
                        caller_binding=binding, caller=caller)
                assert not entered, "Conflicting caller representations entered canonical invocation"
                return
            result = await original(assistant_agent, operation_id="owned-borrowed-caller",
                caller_binding=None if binding_kind == "none" else binding)
            assert result == "original-invocation-result"
            if binding_kind == "active":
                assert entered == [(binding, caller)] and binding.active, (
                    "Canonical invocation replaced or retired its borrowed caller lifetime")
            else:
                assert entered == [(None, None)], "Canonical invocation reactivated or inherited absent caller authority"
    assert not binding.active


async def test_native_private_delivery_counts_actual_protocol_envelope_bytes(publication):
    previous, http = publication
    private = "owned-native-wire-budget-" + "x" * 512
    owner = previous
    produced = []

    async def original_result(context: Context) -> dict[str, str]:
        async with bind_native_mcp_tool_caller(owner, context, "memory:read", disclose=True):
            produced.append(context.request_context.request.scope)
            return {"memory": private}

    owner.runtime.server.add_tool(original_result, name="probe_original_private_wire")
    owner.publish()
    token = (await issue_native_tokens(owner, http))["access_token"]
    result, failure, responses = await observe_native_call(owner, token, "probe_original_private_wire", {})
    assert result is not None and not result.is_error and failure is None
    emitted = [body for status, body in responses if status == 200 and private.encode() in body]
    assert len(emitted) == 1
    configuration = previous.policy.model_dump()
    configuration["max_result_bytes"] = len(emitted[0]) - 1
    policy = MCPServerPolicy.model_validate(configuration)
    await previous.close()
    owner = await NativeMCPPublication.prepare(previous.app, previous.agent, previous.routing_key,
        policy, sovereign_key=lambda: KEY)
    try:
        owner.runtime.server.add_tool(original_result, name="probe_original_private_wire")
        owner.publish()
        token = (await issue_native_tokens(owner, http))["access_token"]
        result, failure, responses = await observe_native_call(owner, token, "probe_original_private_wire", {})
        assert len(produced) == 2, "Original SDK private callback was not reached under both host byte policies"
        assert all(len(body) <= policy.max_result_bytes for _status, body in responses), (
            "Native private delivery omitted actual protocol envelope bytes from its delivery budget")
        assert result is None and failure is not None, "Native private delivery emitted an over-budget protocol envelope"
        assert all(private.encode() not in body for _status, body in responses)
    finally:
        await owner.close()
