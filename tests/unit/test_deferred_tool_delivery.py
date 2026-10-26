"""Real dispatch and post-tool hooks gate private render delivery."""

import asyncio
from dataclasses import dataclass

import pytest
import pytest_asyncio

from kestrel_sdk.hooks.base import Hook, HookEvent, HookOutput
from kestrel_sdk.llm.response import ToolCall
from kestrel_sdk.tools import DeferredToolResult, ToolCategory, ToolResult, ToolSchema
from kestrel_sdk.features.base import Feature as SDKFeature, tool as feature_tool
from kestrel_sovereign.a2a.stores.unified.observability_store import ObservabilityStore
from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
from kestrel_sovereign.agent.parts import drain_parts, emit_part, part_collector
from kestrel_sovereign.agent.tool_registry import ToolRegistryMixin
from kestrel_sovereign.hooks.manager import HooksManager
from kestrel_sovereign.features.base import Feature as CoreFeature
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend


class CompletionHost(OrchestratorEngineMixin, ToolRegistryMixin):
    def __init__(self, store):
        self.did = "did:test:deferred-delivery"
        self.features = {}
        self._direct_tools, self._tool_to_feature, self._explored_features = {}, {}, {}
        self._direct_tool_defs, self._pinned_features = [], set()
        self.observability_store = store
        self.hooks_manager = HooksManager()


class PostPolicy(Hook):
    def __init__(self):
        super().__init__("deferred-post-policy", [HookEvent.POST_TOOL_USE])
        self.started, self.release = asyncio.Event(), asyncio.Event()
        self.redact = False
        self.output = HookOutput.allow()

    async def execute(self, value):
        self.started.set()
        await self.release.wait()
        if self.redact:
            value.tool_response["data"]["value"] = "redacted"
        emit_part("post_policy", {})
        return self.output


class DeferredTool:
    name = "deferred_result"
    schema = ToolSchema(name=name, description="Owned delivery test",
        category=ToolCategory.UTILITY, parameters=[])

    def __init__(self):
        self.executions = []
        self.publications = []

    async def execute(self, **arguments):
        self.executions.append(arguments)

        def publish():
            self.publications.append("published-once")
            return [{"type": "private_view", "data": {"reference": "opaque-reference"}}]

        return DeferredToolResult(ToolResult.ok("Complete", data={"value": "original"},
            parts=[{"type": "original_part", "data": {}}]).to_dict(), publish=publish)


@dataclass
class Boundary:
    host: CompletionHost
    tool: DeferredTool
    binding: object
    policy: PostPolicy

    async def invoke(self, path):
        if path == "named":
            return await self.host.execute_named_tool(self.binding.name, {},
                session_id="deferred-owned-session", source="owned-test")
        if path == "bound":
            return await self.binding.invoke({}, session_id="deferred-owned-session", source="owned-test")
        return await self.host._dispatch_tool_call(
            ToolCall(id="owned-call", name=self.binding.name, arguments={}),
            features_by_tool_name={}, known_tools=self.host._known_tool_names(),
            messages=[], iteration=0, user_message=None, session_id="deferred-owned-session",
        )


@pytest_asyncio.fixture(params=["handle", "sdk-decorator", "core-decorator"])
async def boundary(tmp_path, request):
    db = SQLiteBackend(str(tmp_path / "deferred-audit.db"))
    await db.connect()
    store = ObservabilityStore(db)
    await store.initialize()
    host = CompletionHost(store)
    tool, bindings, policy = DeferredTool(), [], PostPolicy()
    offered = tool
    if request.param != "handle":
        base = SDKFeature if request.param == "sdk-decorator" else CoreFeature

        class Source(base):
            tool_description = "Owned deferred feature fixture"

            async def initialize(self):
                pass

            @feature_tool(name="deferred_result", description="Owned deferred result", category=ToolCategory.UTILITY)
            async def deferred_result(self):
                return await tool.execute()

        source = Source(None)
        offered = next(item for item in source.get_tools() if item.name == tool.name)
    host.hooks_manager.register(policy)
    host.register_dynamic_tools("mcp:deferred-test", [offered],
        on_registered=lambda tool, binding: bindings.append(binding))
    try:
        yield Boundary(host, tool, bindings[0], policy)
    finally:
        policy.release.set()
        await store.close()


@pytest.mark.parametrize("path", ["chat", "named", "bound"])
@pytest.mark.parametrize("redact", [False, True])
async def test_private_delivery_waits_for_actual_post_policy_and_honors_redaction(boundary, path, redact):
    peer = boundary
    peer.policy.redact = redact
    with part_collector():
        call = asyncio.create_task(peer.invoke(path))
        try:
            await asyncio.wait_for(peer.policy.started.wait(), 5)
            assert peer.tool.executions == [{}] and not peer.tool.publications
            assert [part["type"] for part in drain_parts()] == ["original_part"]
            peer.policy.release.set()
            result = await call
            assert type(result) is dict
            assert result["data"]["value"] == ("redacted" if redact else "original")
            assert peer.tool.publications == ([] if redact else ["published-once"])
            assert [part["type"] for part in drain_parts()] == (
                ["post_policy"] if redact else ["post_policy", "private_view"])
            assert [part["type"] for part in result["parts"]] == (
                ["original_part"] if redact else ["original_part", "private_view"])
        finally:
            peer.policy.release.set()
            if not call.done():
                call.cancel()
            await asyncio.gather(call, return_exceptions=True)


@pytest.mark.parametrize("path", ["chat", "named", "bound"])
async def test_pre_tool_refusal_never_executes_or_publishes_a_deferred_result(boundary, path):
    class Refuse(Hook):
        def __init__(self):
            super().__init__("deferred-pre-refusal", [HookEvent.PRE_TOOL_USE])

        async def execute(self, value):
            return HookOutput.deny("Refused by original host policy")

    boundary.host.hooks_manager.register(Refuse())
    result = await boundary.invoke(path)
    assert "Refused by original host policy" in str(result)
    assert not boundary.tool.executions and not boundary.tool.publications


@pytest.mark.parametrize("path", ["chat", "named", "bound"])
@pytest.mark.parametrize("decision", ["deny", "ask", "stop"])
async def test_decision_only_post_policy_withholds_private_parts_without_replaying_the_tool(boundary, path, decision):
    peer = boundary
    peer.policy.output = {
        "deny": HookOutput.deny("Private delivery denied"),
        "ask": HookOutput.ask("not-an-automatic-retry", "Private delivery requires approval"),
        "stop": HookOutput(continue_execution=False, stop_reason="Private delivery stopped"),
    }[decision]
    peer.policy.release.set()
    with part_collector():
        result = await peer.invoke(path)
        assert peer.tool.executions == [{}]
        assert result["data"]["value"] == "original"
        assert peer.tool.publications == [], "A decision-only post-tool block released private render data"
        assert [part["type"] for part in result["parts"]] == ["original_part"]
        assert [part["type"] for part in drain_parts()] == ["original_part", "post_policy"]


@pytest.mark.parametrize("path", ["chat", "named", "bound"])
async def test_cancellation_while_post_policy_is_pending_never_publishes(boundary, path):
    peer = boundary
    with part_collector():
        call = asyncio.create_task(peer.invoke(path))
        try:
            await asyncio.wait_for(peer.policy.started.wait(), 5)
            call.cancel()
            with pytest.raises(asyncio.CancelledError):
                await call
            assert call.cancelled() and peer.tool.executions == [{}]
            assert not peer.tool.publications
            assert [part["type"] for part in drain_parts()] == ["original_part"]
        finally:
            peer.policy.release.set()
            await asyncio.gather(call, return_exceptions=True)
