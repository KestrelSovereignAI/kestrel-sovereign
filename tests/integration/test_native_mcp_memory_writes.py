"""Native MCP writes use the original governed MemoryAgencyFeature."""

import asyncio
from uuid import uuid4

import pytest
from kestrel_sdk.hooks.base import Hook, HookEvent, HookOutput
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

from tests.integration.test_native_mcp_memory_tools import observe_native_call
from tests.integration.test_native_mcp_publication import (
    assistant_agent as _assistant_agent,
)
from tests.integration.test_native_mcp_publication import publication as _publication
from tests.integration.test_native_mcp_tool_context import issue_native_tokens

assistant_agent = _assistant_agent
publication = _publication

pytestmark = [
    pytest.mark.asyncio, pytest.mark.integration,
    pytest.mark.parametrize("assistant_agent", [{"features": ["MemoryAgencyFeature"]}], indirect=True),
    pytest.mark.parametrize("publication", [{"clients": [{
        "client_id": "owned-provider", "client_name": "Owned provider",
        "redirect_uris": ["https://provider.example/callback"], "client_secret": "owned-secret",
        "scopes": ["memory:read", "memory:write"],
    }, {
        "client_id": "owned-second-provider", "client_name": "Owned second provider",
        "redirect_uris": ["https://provider.example/callback"], "client_secret": "owned-secret",
        "scopes": ["memory:read", "memory:write"],
    }]}], indirect=True),
]


def fact_arguments(**changes):
    return {"subject": "user", "predicate": "preferred_deploy_region", "value": "us-central1",
        "operation_id": uuid4().hex, "conversation_id": "owned-fact-conversation", **changes}


async def test_native_fact_write_reaches_canonical_semantic_storage(publication):
    owner, http = publication
    owner.publish()
    tokens = await issue_native_tokens(owner, http, scope="memory:write")
    http.headers.update({"authorization": "Bearer " + tokens["access_token"]})
    async with Client(streamable_http_client(owner.resource, http_client=http), mode="2026-07-28") as client:
        tools = await client.list_tools()
        assert "save_fact" in {tool.name for tool in tools.tools}, (
            "Native MCP does not expose the existing governed save_fact tool")
        fact_tool = next(tool for tool in tools.tools if tool.name == "save_fact")
        assert fact_tool.annotations.destructive_hint is True
        result = await client.call_tool("save_fact", fact_arguments())
        assert not result.is_error, result
        assert result.structured_content["status"] == "ok", result
        data = result.structured_content["data"]
        assert data["saved"] and data["assertion_id"] and data["revision_id"]
    assertion = await owner.agent.privacy_agent.storage.get_assertion(data["assertion_id"])
    assert assertion is not None and assertion.object.value == "us-central1"
    sources = await owner.agent.storage.list_assertion_sources(data["assertion_id"])
    assert sources and all(source.actor.startswith("mcp:") and source.actor != owner.agent.did for source in sources), (
        "Native fact provenance attributed the external caller to Kestrel")
    assert all(source.source_kind == "http_request" and source.locator.startswith(owner.resource) for source in sources)
    assert tokens["access_token"] not in repr(sources)
    claims = await owner.agent._raw_storage.db.fetchall("SELECT state FROM tool_effect_claims")
    assert claims and all(row[0] == "RETURNED_OK" for row in claims), (
        "Native fact write bypassed the canonical durable tool journal")


async def test_read_consent_never_authorizes_a_canonical_fact_write(publication):
    owner, http = publication
    owner.publish()
    token = (await issue_native_tokens(owner, http))["access_token"]
    result, failure, _responses = await observe_native_call(owner, token, "save_fact", fact_arguments())
    assert failure is not None or result.is_error, "Read-only native grant executed save_fact"
    assert not await owner.agent._raw_storage.db.fetchall("SELECT state FROM tool_effect_claims"), (
        "Read-only MCP consent admitted a canonical write intention")


@pytest.mark.parametrize("decision", ["deny", "rewrite"])
async def test_native_fact_write_honors_original_governed_hooks(publication, decision):
    owner, http = publication
    seen = []

    class OriginalPolicy(Hook):
        def __init__(self):
            super().__init__("owned-native-write-policy", [HookEvent.PRE_TOOL_USE], matcher="^save_fact$", priority=1)

        async def execute(self, input):
            seen.append(input.tool_input)
            if decision == "deny":
                return HookOutput.deny("Owned canonical write refusal")
            return HookOutput.modify({**input.tool_input, "value": "europe-west4"})

    hook = OriginalPolicy()
    await owner.agent.hooks_manager.register_async(hook)
    owner.publish()
    token = (await issue_native_tokens(owner, http, scope="memory:write"))["access_token"]
    try:
        result, failure, _responses = await observe_native_call(owner, token, "save_fact", fact_arguments())
    finally:
        owner.agent.hooks_manager.unregister(hook)
    assert seen and failure is None
    if decision == "deny":
        assert result.is_error and result.structured_content["success"] is False
    else:
        assert not result.is_error and result.structured_content["data"]["value"] == "europe-west4"
        assertion = await owner.agent.storage.get_assertion(result.structured_content["data"]["assertion_id"])
        assert assertion.object.value == "europe-west4"


async def test_native_fact_unsupported_terms_are_canonical_refusals(publication):
    owner, http = publication
    owner.publish()
    token = (await issue_native_tokens(owner, http, scope="memory:write"))["access_token"]
    result, failure, _responses = await observe_native_call(owner, token, "save_fact",
        fact_arguments(predicate="unregistered-arbitrary-predicate"))
    assert failure is None and result.is_error
    assert "unsupported predicate" in result.model_dump_json()
    claims = await owner.agent._raw_storage.db.fetchall("SELECT state FROM tool_effect_claims")
    assert claims and all(row[0] == "RETURNED_ERROR" for row in claims)


async def test_revocation_during_original_hook_prevents_native_fact_effect(publication, monkeypatch):
    owner, http = publication
    entered, release = asyncio.Event(), asyncio.Event()
    writes = []
    original = owner.agent.storage.save_explicit_fact

    async def observe_write(**arguments):
        writes.append(arguments)
        return await original(**arguments)

    monkeypatch.setattr(owner.agent.storage, "save_explicit_fact", observe_write)

    class OriginalPolicy(Hook):
        def __init__(self):
            super().__init__("owned-native-write-pending-policy", [HookEvent.PRE_TOOL_USE], matcher="^save_fact$", priority=1)

        async def execute(self, input):
            entered.set()
            await release.wait()
            return HookOutput.allow()

    hook = OriginalPolicy()
    await owner.agent.hooks_manager.register_async(hook)
    owner.publish()
    token = (await issue_native_tokens(owner, http, scope="memory:write"))["access_token"]
    task = asyncio.create_task(observe_native_call(owner, token, "save_fact", fact_arguments(), discover_first=True))
    try:
        async with asyncio.timeout(15):
            await entered.wait()
            response = await http.post(owner.resource + "/oauth/revoke", data={
                "token": token, "client_id": "owned-provider", "client_secret": "owned-secret",
                "resource": owner.resource, "token_type_hint": "access_token"})
            assert response.status_code == 200, response.text
            release.set()
            result, failure, _responses = await task
        assert not writes, "Revoked MCP consent still reached the canonical fact writer"
        assert failure is not None or result.is_error
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        owner.agent.hooks_manager.unregister(hook)


async def test_expired_original_grant_rolls_back_canonical_fact_and_receipt(publication, monkeypatch):
    owner, http = publication
    owner.publish()
    token = (await issue_native_tokens(owner, http, scope="memory:write"))["access_token"]
    original = owner.agent.storage.save_explicit_fact
    receipts = []
    pending = owner.runtime.authorization.pending
    now = pending.now(fractional=True)

    async def expire_after_original_write(**arguments):
        receipt = await original(**arguments)
        receipts.append(receipt)
        monkeypatch.setattr(pending, "clock", lambda: now + owner.policy.access_token_seconds + 1)
        return receipt

    monkeypatch.setattr(owner.agent.storage, "save_explicit_fact", expire_after_original_write)
    result, failure, _responses = await observe_native_call(owner, token, "save_fact", fact_arguments(), discover_first=True)
    assert receipts and receipts[0].saved, "The original canonical fact mutation was not reached"
    assert failure is not None or result.is_error
    assert await owner.agent.storage.get_assertion(receipts[0].assertion_id) is None, (
        "An expired MCP write committed its provisional canonical fact")


async def test_native_fact_bounds_actual_post_hook_write_arguments(publication, monkeypatch):
    owner, http = publication
    writes = []
    original = owner.agent.storage.save_explicit_fact

    async def observe_write(**arguments):
        writes.append(arguments)
        return await original(**arguments)

    monkeypatch.setattr(owner.agent.storage, "save_explicit_fact", observe_write)

    class OversizedRewrite(Hook):
        def __init__(self):
            super().__init__("owned-oversize-native-write", [HookEvent.PRE_TOOL_USE], matcher="^save_fact$", priority=1)

        async def execute(self, input):
            return HookOutput.modify({**input.tool_input, "value": "x" * owner.policy.max_request_bytes})

    hook = OversizedRewrite()
    await owner.agent.hooks_manager.register_async(hook)
    owner.publish()
    token = (await issue_native_tokens(owner, http, scope="memory:write"))["access_token"]
    try:
        result, failure, _responses = await observe_native_call(owner, token, "save_fact", fact_arguments())
    finally:
        owner.agent.hooks_manager.unregister(hook)
    assert failure is not None or result.is_error
    assert not writes, "Native MCP executed a hook-rewritten write beyond configured bounds"
    assert not await owner.agent._raw_storage.db.fetchall("SELECT state FROM tool_effect_claims")


async def test_native_fact_never_switches_original_sdk_request_after_hooks(publication, monkeypatch):
    from starlette.requests import Request

    from kestrel_sovereign.mcp_server.memory_tools import NativeMCPMemoryTools

    owner, http = publication
    contexts, writes = [], []
    original_invoke = NativeMCPMemoryTools._invoke
    original_write = owner.agent.storage.save_explicit_fact

    async def observe_original(self, context, *args, **kwargs):
        contexts.append(context)
        return await original_invoke(self, context, *args, **kwargs)

    async def observe_write(**arguments):
        writes.append(arguments)
        return await original_write(**arguments)

    monkeypatch.setattr(NativeMCPMemoryTools, "_invoke", observe_original)
    monkeypatch.setattr(owner.agent.storage, "save_explicit_fact", observe_write)

    class ReplaceRequest(Hook):
        def __init__(self):
            super().__init__("owned-sdk-request-substitution", [HookEvent.PRE_TOOL_USE], matcher="^save_fact$", priority=1)

        async def execute(self, input):
            context = contexts[-1]
            context.request_context.request = Request(dict(context.request_context.request.scope))
            return HookOutput.allow()

    hook = ReplaceRequest()
    await owner.agent.hooks_manager.register_async(hook)
    owner.publish()
    token = (await issue_native_tokens(owner, http, scope="memory:write"))["access_token"]
    try:
        result, failure, _responses = await observe_native_call(owner, token, "save_fact", fact_arguments())
    finally:
        owner.agent.hooks_manager.unregister(hook)
    assert contexts, "The actual native SDK callback was not observed"
    assert not writes, "A substituted native SDK request reached the canonical fact writer"
    assert failure is not None or result.is_error


async def test_caught_canonical_fact_conflict_rolls_back_its_partial_mutation(publication, monkeypatch):
    from kestrel_sovereign.knowledge import AssertionStatus
    from tests.utils.semantic_mutation_faults import immutable_revision_conflict

    owner, http = publication
    owner.publish()
    token = (await issue_native_tokens(owner, http, scope="memory:write"))["access_token"]
    first, failure, _responses = await observe_native_call(owner, token, "save_fact", fact_arguments())
    assert failure is None and not first.is_error
    receipt = first.structured_content["data"]
    store = owner.agent._raw_storage._assertion_store()
    with immutable_revision_conflict(monkeypatch, store, object_value="europe-west4") as conflicts:
        result, failure, _responses = await observe_native_call(owner, token, "save_fact",
            fact_arguments(value="europe-west4"))
    assert conflicts, "The real canonical replacement revision boundary was not reached"
    assert failure is None and result.is_error
    held = await owner.agent.storage.get_assertion(receipt["assertion_id"], include_inactive=True)
    assert held.status is AssertionStatus.ACTIVE and held.revision_id == receipt["revision_id"], (
        "A caught semantic conflict committed its partial predecessor mutation")


async def test_distinct_native_principals_may_reuse_client_fact_operation_ids(publication):
    owner, http = publication
    owner.publish()
    arguments = fact_arguments(operation_id="owned-independent-principal-operation")
    receipts = []
    for client_id in ["owned-provider", "owned-second-provider"]:
        token = (await issue_native_tokens(owner, http, client_id=client_id, scope="memory:write"))["access_token"]
        result, failure, _responses = await observe_native_call(owner, token, "save_fact", arguments)
        assert failure is None and not result.is_error, (
            "Independent native MCP principals collided in canonical fact identity", result, failure)
        receipts.append(result.structured_content["data"])
    assert len({receipt["operation_id"] for receipt in receipts}) == 2
    sources = await owner.agent.storage.list_assertion_sources(receipts[-1]["assertion_id"])
    assert len({source.actor for source in sources}) == 2, (
        "Canonical semantic provenance merged independent native MCP principals")
