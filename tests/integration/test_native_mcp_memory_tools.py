"""Real native MCP clients reach actual governed MemoryFeature execution."""

import asyncio
from uuid import uuid4

import httpx2
import pytest
from kestrel_sdk.hooks.base import Hook, HookEvent, HookOutput
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPError

from kestrel_sovereign.auth import AuthMethod, current_caller_context
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
async def test_native_memory_tools_use_actual_cold_core_feature_and_durable_admission(publication):
    owner, http = publication
    agent = owner.agent
    content = "owned-native-memory-" + uuid4().hex
    row = await agent.privacy_agent.add_conversation("user", content, session_id="owned-core-conversation")
    assert type(row) is int and row > 0
    owner.publish()
    token = (await issue_native_tokens(owner, http))["access_token"]
    http.headers.update({"authorization": "Bearer " + token, "x-api-key": KEY})
    async with Client(streamable_http_client(owner.resource, http_client=http), mode="2026-07-28") as client:
        tools = await client.list_tools()
        assert {tool.name for tool in tools.tools} == {"search_memory", "recall_recent"}
        for name, arguments in [
            ("search_memory", {"query": content, "session_id": "owned-core-conversation"}),
            ("recall_recent", {}),
        ]:
            result = await client.call_tool(name, {**arguments, "limit": 5,
                "operation_id": uuid4().hex, "conversation_id": "owned-core-conversation"})
            assert not result.is_error, result
            assert result.structured_content is not None, "Native memory tool omitted its official structured result"
            assert result.structured_content["status"] == "ok", result
            assert content in result.model_dump_json(), "Actual MCP memory call did not reach original Core storage"
    claims = await agent._raw_storage.db.fetchall("SELECT state FROM tool_effect_claims")
    assert len(claims) >= 2 and all(row[0] == "RETURNED_OK" for row in claims), (
        "Native MCP memory bypassed canonical durable tool admission")


async def observe_native_call(owner, token, name, arguments, *, discover_first=False):
    responses = []

    async def observe(response):
        responses.append((response.status_code, await response.aread()))

    result = failure = None
    try:
        async with (
            httpx2.AsyncClient(transport=httpx2.ASGITransport(app=owner.app, raise_app_exceptions=False),
                base_url=owner.policy.console_origin, headers={"authorization": "Bearer " + token},
                event_hooks={"response": [observe]}) as http,
            Client(streamable_http_client(owner.resource, http_client=http), mode="2026-07-28") as client,
        ):
            if discover_first:
                await client.list_tools()
            result = await client.call_tool(name, arguments)
    except (httpx2.HTTPError, MCPError, ExceptionGroup) as error:
        failure = error
    return result, failure, responses


@pytest.mark.parametrize("assistant_agent", [{"features": ["MemoryFeature"]}], indirect=True)
@pytest.mark.parametrize("decision", ["rewrite", "deny"])
async def test_native_memory_honors_actual_hooks_and_principal_session(publication, decision):
    owner, http = publication
    content = "owned-hook-selected-memory-" + uuid4().hex
    await owner.agent.privacy_agent.add_conversation("user", content, session_id="owned-source-session")
    events = []

    class OriginalHook(Hook):
        def __init__(self):
            super().__init__("owned-native-memory-governance", [HookEvent.PRE_TOOL_USE, HookEvent.POST_TOOL_USE],
                matcher="^search_memory$", priority=1)

        async def execute(self, input):
            caller = current_caller_context()
            assert caller is not None and caller.auth_method is AuthMethod.MCP_OAUTH and not caller.is_sovereign
            events.append((input.hook_event_name, input.session_id, caller.identity))
            if input.hook_event_name == HookEvent.PRE_TOOL_USE.value:
                if decision == "deny":
                    return HookOutput.deny("Owned native memory policy refusal")
                return HookOutput.modify({**input.tool_input, "query": content})
            return HookOutput.allow()

    hook = OriginalHook()
    await owner.agent.hooks_manager.register_async(hook)
    owner.publish()
    token = (await issue_native_tokens(owner, http))["access_token"]
    http.headers.update({"authorization": "Bearer " + token})
    try:
        async with Client(streamable_http_client(owner.resource, http_client=http), mode="2026-07-28") as client:
            result = await client.call_tool("search_memory", {"query": "owned-nonmatching-original-query",
                "limit": 5, "operation_id": uuid4().hex, "conversation_id": "owned-source-session"})
    finally:
        owner.agent.hooks_manager.unregister(hook)
    assert events, "Native memory bypassed the original governed hook dispatch"
    assert all(session.startswith("mcp:") and session != "owned-source-session"
        for _event, session, _caller in events), "Native memory trusted a caller-supplied Core audit session"
    if decision == "deny":
        assert result.is_error, "Native memory reported canonical tool denial as MCP execution success"
        assert result.structured_content["success"] is False and content not in result.model_dump_json(), (
            "Native memory bypassed canonical PRE_TOOL_USE denial")
        assert [event for event, _session, _caller in events] == [HookEvent.PRE_TOOL_USE.value]
    else:
        assert result.structured_content["status"] == "ok" and content in result.model_dump_json(), (
            "Native memory ignored the actual hook-rewritten search input")
        assert [event for event, _session, _caller in events] == [
            HookEvent.PRE_TOOL_USE.value, HookEvent.POST_TOOL_USE.value]


@pytest.mark.parametrize("assistant_agent", [{"features": ["MemoryFeature"]}], indirect=True)
async def test_native_memory_retry_after_refresh_retains_original_durable_intention(publication):
    owner, http = publication
    owner.publish()
    tokens = await issue_native_tokens(owner, http)
    arguments = {"limit": 5, "operation_id": uuid4().hex, "conversation_id": "owned-retry-session"}
    first, failure, _responses = await observe_native_call(owner, tokens["access_token"], "recall_recent", arguments)
    assert failure is None and first is not None and first.structured_content["status"] == "ok"
    before = await owner.agent._raw_storage.db.fetchall("SELECT * FROM tool_effect_claims ORDER BY claim_id")
    refreshed = await http.post(owner.resource + "/oauth/token", data={"grant_type": "refresh_token",
        "refresh_token": tokens["refresh_token"], "client_id": "owned-provider",
        "client_secret": "owned-secret", "resource": owner.resource})
    assert refreshed.status_code == 200
    second, failure, responses = await observe_native_call(owner, refreshed.json()["access_token"], "recall_recent", arguments)
    assert failure is not None and second is None and any(status == 500 for status, _body in responses), (
        "OAuth refresh silently readmitted the same native memory intention")
    assert await owner.agent._raw_storage.db.fetchall("SELECT * FROM tool_effect_claims ORDER BY claim_id") == before, (
        "Retried native MCP memory bypassed its original durable admission")


@pytest.mark.parametrize("assistant_agent", [{"features": ["MemoryFeature"]}], indirect=True)
@pytest.mark.parametrize("invalid", ["count", "query", "operation", "conversation"])
async def test_native_memory_input_limits_refuse_before_canonical_admission(publication, invalid):
    owner, http = publication
    owner.publish()
    token = (await issue_native_tokens(owner, http))["access_token"]
    arguments = {"query": "owned-query", "limit": 5, "operation_id": uuid4().hex, "conversation_id": "owned-client"}
    arguments.update({
        "count": {"limit": owner.policy.max_memory_records + 1},
        "query": {"query": "x" * (owner.policy.max_query_bytes + 1)},
        "operation": {"operation_id": ""}, "conversation": {"conversation_id": ""},
    }[invalid])
    result, failure, _responses = await observe_native_call(owner, token, "search_memory", arguments)
    assert failure is None and result is not None and result.is_error, (
        "Native memory input escaped its configured admission bounds")
    assert not await owner.agent._raw_storage.db.fetchall("SELECT state FROM tool_effect_claims"), (
        "Invalid native memory input entered canonical tool execution")


@pytest.mark.parametrize("assistant_agent", [{"features": ["MemoryFeature"]}], indirect=True)
@pytest.mark.parametrize("publication", [{"max_result_bytes": 256}], indirect=True)
async def test_native_memory_result_byte_limit_withholds_actual_private_body(publication):
    owner, http = publication
    private = "owned-oversize-private-memory-" + "x" * 2048
    await owner.agent.privacy_agent.add_conversation("user", private, session_id="owned-large-memory")
    owner.publish()
    token = (await issue_native_tokens(owner, http))["access_token"]
    result, failure, responses = await observe_native_call(owner, token, "recall_recent",
        {"limit": 5, "operation_id": uuid4().hex, "conversation_id": "owned-client"})
    assert result is None and failure is not None and any(status == 500 for status, _body in responses), (
        "Native memory result exceeded its configured byte limit")
    assert all(private.encode() not in body for _status, body in responses), (
        "Oversize private native memory escaped through the original error response")


@pytest.mark.parametrize("assistant_agent", [{"features": ["MemoryFeature"]}], indirect=True)
@pytest.mark.parametrize("publication", [{"operation_seconds": 0.1}], indirect=True)
async def test_native_memory_budget_covers_original_governed_execution(publication):
    owner, http = publication
    entered, released, ended = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class BlockOriginalHook(Hook):
        def __init__(self):
            super().__init__("owned-original-memory-execution", [HookEvent.PRE_TOOL_USE],
                matcher="^recall_recent$", priority=1, awaits_user_input=True)

        async def execute(self, input):
            entered.set()
            try:
                await released.wait()
            finally:
                ended.set()
            return HookOutput.allow()

    hook = BlockOriginalHook()
    await owner.agent.hooks_manager.register_async(hook)
    owner.publish()
    token = (await issue_native_tokens(owner, http))["access_token"]
    original = asyncio.create_task(observe_native_call(owner, token, "recall_recent",
        {"limit": 5, "operation_id": uuid4().hex, "conversation_id": "owned-budget-session"}))
    try:
        await asyncio.wait({original}, timeout=5)
        completed = original.done()
        original_hook_retired = entered.is_set() and ended.is_set()
        caller_retired = not owner._tool_callers
        claims = await owner.agent._raw_storage.db.fetchall("SELECT state FROM tool_effect_claims")
    finally:
        released.set()
        result, failure, responses = await original
        owner.agent.hooks_manager.unregister(hook)
    assert entered.is_set(), "Original governed memory hook did not execute"
    assert completed, "Native memory ignored its configured execution deadline"
    assert original_hook_retired and caller_retired, "Expired native execution retained original hook or caller authority"
    assert not claims, "Expired native memory executed after its original governance deadline"
    assert result is None and failure is not None and any(status == 500 for status, _body in responses)
