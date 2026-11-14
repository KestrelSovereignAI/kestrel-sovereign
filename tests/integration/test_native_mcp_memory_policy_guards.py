"""Actual native policy must cover effective arguments and fractional expiry."""

import asyncio
import inspect
from contextlib import asynccontextmanager

import pytest
from kestrel_sdk.hooks.base import Hook, HookEvent, HookOutput
from kestrel_sdk.resources import ResourceUnsupported
from kestrel_sdk.tools.base import ToolCategory, ToolParameter, ToolSchema
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError

from kestrel_sovereign.features.memory.feature import MemoryFeature
from kestrel_sovereign.mcp_server.memory_tools import NativeMCPMemoryTools
from kestrel_sovereign.mcp_server.tool_caller import bind_native_mcp_tool_caller
from tests.integration.test_native_mcp_memory_tools import observe_native_call
from tests.integration.test_native_mcp_publication import (
    assistant_agent as _assistant_agent,
)
from tests.integration.test_native_mcp_publication import publication as _publication
from tests.integration.test_native_mcp_tool_context import issue_native_tokens

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]
assistant_agent = _assistant_agent
publication = _publication


@pytest.mark.parametrize("assistant_agent", [{"features": ["MemoryFeature"]}], indirect=True)
@pytest.mark.parametrize("publication", [{"max_memory_records": 100}, {"max_memory_records": 1}], indirect=True)
async def test_native_memory_hook_omission_validates_the_selected_tools_actual_default(publication, monkeypatch):
    owner, http = publication
    original = owner.agent._raw_storage.conversation.get_conversation_history
    history_signature = inspect.signature(original)
    reads, rewrites = [], []
    private = "owned-memory-default-after-real-hook-omission"
    await owner.agent.privacy_agent.add_conversation("user", private, session_id="owned-hook-default")

    async def original_history(*args, **kwargs):
        reads.append(history_signature.bind(*args, **kwargs).arguments["limit"])
        return await original(*args, **kwargs)

    monkeypatch.setattr(owner.agent._raw_storage.conversation, "get_conversation_history", original_history)

    class OriginalDefaultRewrite(Hook):
        def __init__(self):
            super().__init__("owned-native-default-policy", [HookEvent.PRE_TOOL_USE],
                matcher="^recall_recent$", priority=1)

        async def execute(self, input):
            rewrites.append(input.tool_input)
            return HookOutput.modify({})

    hook = OriginalDefaultRewrite()
    await owner.agent.hooks_manager.register_async(hook)
    owner.publish()
    token = (await issue_native_tokens(owner, http))["access_token"]
    try:
        result, failure, responses = await observe_native_call(owner, token, "recall_recent",
            {"limit": 1, "operation_id": "owned-hook-default", "conversation_id": "owned-hook-default"},
            discover_first=True)
    finally:
        owner.agent.hooks_manager.unregister(hook)
    assert len(rewrites) == 1, "Actual canonical optional-argument omission was not reached"
    if owner.policy.max_memory_records == 100:
        assert result is not None and not result.is_error and failure is None, (
            "Native memory refused a valid hook omission instead of using the selected tool default", failure)
        assert reads == [20]
        assert any(private.encode() in body for _status, body in responses)
    else:
        assert not reads, "Native memory executed an omitted default beyond the configured record limit"
        assert failure is not None or (result is not None and result.is_error)


async def test_native_memory_count_default_comes_only_from_the_selected_canonical_schema(publication):
    owner, _http = publication
    tools = NativeMCPMemoryTools(owner)
    def schema(parameters):
        return ToolSchema(name="recall_recent", description="Owned default validation", category=ToolCategory.MEMORY,
            parameters=parameters)
    for selected in (None, schema([]), schema({"type": "object"}), schema([
            ToolParameter("limit", "integer", "Count", required=True, default=3)])):
        with pytest.raises(ToolError, match="default is unavailable"):
            await tools._validate_arguments({}, selected)
    for value in (None, True, "3", 0, owner.policy.max_memory_records + 1):
        with pytest.raises(ToolError, match="count exceeds"):
            await tools._validate_arguments({}, schema([ToolParameter("limit", "integer", "Count", default=value)]))
    await tools._validate_arguments({}, schema([ToolParameter("limit", "integer", "Count", default=3)]))
    await tools._validate_arguments({"limit": 1}, schema([]))


@pytest.mark.parametrize("assistant_agent", [{"features": ["MemoryFeature"]}], indirect=True)
@pytest.mark.parametrize("publication", [{"max_memory_records": 1, "max_query_bytes": 8}], indirect=True)
@pytest.mark.parametrize("rewrite", ["count", "query"])
async def test_native_memory_bounds_actual_hook_rewritten_arguments(publication, monkeypatch, rewrite):
    owner, http = publication
    feature = next(feature for feature in owner.agent.features.values() if isinstance(feature, MemoryFeature))
    original = feature._search_memory_impl
    calls, rewrites = [], []
    await owner.agent.privacy_agent.add_conversation("user", "needle-one", session_id="owned-policy")
    await owner.agent.privacy_agent.add_conversation("user", "needle-two", session_id="owned-policy")

    async def original_search(*args, **kwargs):
        calls.append((args, kwargs))
        return await original(*args, **kwargs)

    monkeypatch.setattr(feature, "_search_memory_impl", original_search)

    class OriginalRewrite(Hook):
        def __init__(self):
            super().__init__("owned-native-argument-policy", [HookEvent.PRE_TOOL_USE],
                matcher="^search_memory$", priority=1)

        async def execute(self, input):
            rewrites.append(input.tool_input)
            return HookOutput.modify({**input.tool_input,
                **({"limit": 2} if rewrite == "count" else {"query": "needle" + "x" * 16})})

    hook = OriginalRewrite()
    await owner.agent.hooks_manager.register_async(hook)
    owner.publish()
    token = (await issue_native_tokens(owner, http))["access_token"]
    try:
        result, failure, _responses = await observe_native_call(owner, token, "search_memory",
            {"query": "needle", "limit": 1, "operation_id": "owned-rewritten-" + rewrite,
                "conversation_id": "owned-policy"})
    finally:
        owner.agent.hooks_manager.unregister(hook)
    assert len(rewrites) == 1, "Actual canonical MCP argument rewrite was not reached"
    assert not calls, "Native memory executed hook-rewritten arguments beyond configured bounds"
    assert failure is not None or (result is not None and result.is_error)


async def test_native_private_delivery_obeys_fractional_token_expiry(publication, monkeypatch):
    owner, http = publication
    private = "owned-native-private-body-at-fractional-expiry"
    entered, release = asyncio.Event(), asyncio.Event()
    forwarded = []
    loop = asyncio.get_running_loop()

    class OriginalBlockedBody:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            async def original_send(message):
                if message["type"] == "http.response.body" and private.encode() in message.get("body", b""):
                    entered.set()
                    await release.wait()
                    forwarded.append(owner.runtime.authorization.pending.clock())
                await send(message)
            await self.app(scope, receive, original_send)

    async def original_result(context: Context) -> dict[str, str]:
        async with bind_native_mcp_tool_caller(owner, context, "memory:read", disclose=True):
            return {"memory": private}

    owner.app.add_middleware(OriginalBlockedBody)
    owner.runtime.server.add_tool(original_result, name="probe_fractional_private_expiry")
    owner.publish()
    token = (await issue_native_tokens(owner, http))["access_token"]
    original_token = await owner.runtime.authorization.verify_token(token)
    assert original_token is not None and original_token.expires_at is not None
    expiry = original_token.expires_at
    started = loop.time()
    monkeypatch.setattr(owner.runtime.authorization.pending, "clock", lambda: expiry - 0.5 + loop.time() - started)
    release_timer = loop.call_at(started + 0.65, release.set)
    original = asyncio.create_task(observe_native_call(owner, token, "probe_fractional_private_expiry", {}))
    observer = asyncio.create_task(entered.wait())
    try:
        await asyncio.wait({original, observer}, timeout=5, return_when=asyncio.FIRST_COMPLETED)
        reached_body = entered.is_set()
        await asyncio.wait({original}, timeout=5)
        completed = original.done()
    finally:
        release_timer.cancel()
        release.set()
        observer.cancel()
        result, failure, responses = await original
        await asyncio.gather(observer, return_exceptions=True)
    assert reached_body, "Original private MCP bytes did not reach their physical delivery boundary"
    assert completed, "Original native grant did not bound its physical send"
    assert not forwarded, "Native private bytes crossed fractional token expiry before the post-send check"
    assert result is None and failure is not None
    assert all(private.encode() not in body for _status, body in responses)


async def test_native_authorization_clock_keeps_fractional_deadline_precision(publication, monkeypatch):
    owner, _http = publication
    pending = owner.runtime.authorization.pending
    monkeypatch.setattr(pending, "clock", lambda: 123.75)
    assert pending.now() == 123 and type(pending.now()) is int
    assert pending.now(fractional=True) == 123.75
    for invalid in (float("nan"), float("inf"), -1.0):
        monkeypatch.setattr(pending, "clock", lambda value=invalid: value)
        for fractional in (False, True):
            with pytest.raises(ValueError, match="finite Unix time"):
                pending.now(fractional=fractional)


@pytest.mark.parametrize("assistant_agent", [{"features": ["MemoryFeature"]}], indirect=True)
@pytest.mark.parametrize("dispatch", ["subagent", "receipt"])
@pytest.mark.parametrize("guard_kind", ["arguments", "lifetime"])
async def test_direct_argument_guards_refuse_other_execution_paths(assistant_agent, dispatch, guard_kind):
    from kestrel_sovereign.signals.constitution_canary import PHANTOM_RECEIPT_TOOL_NAME

    guards = []

    async def original_guard(arguments, schema):
        guards.append(arguments)

    @asynccontextmanager
    async def original_lifetime():
        guards.append("entered-unsupported-lifetime")
        yield

    tool = "memory_feature" if dispatch == "subagent" else PHANTOM_RECEIPT_TOOL_NAME
    with pytest.raises(ResourceUnsupported, match="Direct-tool guards cannot govern"):
        await assistant_agent.execute_named_tool(tool, {}, session_id="owned-direct-argument-policy",
            source="owned-direct-policy", operation_id="owned-unsupported-" + dispatch,
            **({"argument_guard": original_guard} if guard_kind == "arguments"
                else {"execution_lifetime": original_lifetime}))
    assert not guards, "A direct-tool policy was applied to an unsupported execution shape"
