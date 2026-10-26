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
from kestrel_sovereign.execution_custody import ExecutionAuthorityError
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
        self.replace = False
        self.replace_with_none = False
        self.redact_message = False
        self.clear_parts = False
        self.observe_result = None
        self.failure = None
        self.output = HookOutput.allow()

    async def execute(self, value):
        self.started.set()
        if self.observe_result is not None:
            self.observe_result(value.tool_response)
        if self.failure == "raise":
            raise RuntimeError("private-policy-sentinel")
        if self.failure == "timeout":
            await asyncio.Event().wait()
        if self.failure == "invalid":
            return None
        await self.release.wait()
        if self.redact:
            value.tool_response["data"]["value"] = "redacted"
        if self.replace:
            value.tool_response = ToolResult.ok("Replacement").to_dict()
        if self.replace_with_none:
            value.tool_response = None
        if self.redact_message:
            value.tool_response["confirmation"] = "Reviewed replacement text"
        if self.clear_parts:
            value.tool_response["parts"] = None
        emit_part("post_policy", {})
        return self.output


class DeferredTool:
    name = "deferred_result"
    schema = ToolSchema(name=name, description="Owned delivery test",
        category=ToolCategory.UTILITY, parameters=[])

    def __init__(self):
        self.executions = []
        self.publications = []
        self.publication_error = None
        self.publication_parts = None
        self.mutate_before_error = False
        self.part_form = "original"
        self.active_result = None

    async def execute(self, **arguments):
        self.executions.append(arguments)

        def publish():
            self.publications.append("published-once")
            if self.publication_error is not None:
                if self.mutate_before_error:
                    self.active_result["data"]["value"] = "private-mutated-factory-sentinel"
                raise self.publication_error
            if self.publication_parts is not None:
                return self.publication_parts
            return [{"type": "private_view", "data": {"reference": "opaque-reference"}}]

        envelope = ToolResult.ok("Complete", data={"value": "original"}).to_dict()
        if self.part_form == "original":
            envelope["parts"] = [{"type": "original_part", "data": {}}]
        elif self.part_form == "null":
            envelope["parts"] = None
        elif self.part_form == "empty":
            envelope["parts"] = []
        self.last_result = DeferredToolResult(envelope, publish=publish)
        return self.last_result


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
    policy.observe_result = lambda result: setattr(tool, "active_result", result)
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

        source = Source(host)
        assert source.agent.supports_deferred_tool_results is True
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


@pytest.mark.parametrize("path", ["chat", "named", "bound"])
@pytest.mark.parametrize("failure", ["raise", "timeout", "invalid"])
@pytest.mark.parametrize("enforcing", [False, True])
async def test_incomplete_post_policy_never_publishes_private_parts(boundary, path, failure, enforcing):
    peer = boundary
    peer.policy.failure = failure
    peer.policy.fail_closed = enforcing
    peer.policy.timeout = 0.01
    peer.policy.release.set()
    with part_collector():
        result = await peer.invoke(path)
        assert peer.tool.executions == [{}]
        assert not peer.tool.publications, "Incomplete policy was treated as permission for private delivery"
        assert result["data"] == {"value": "original"}
        assert [part["type"] for part in result["parts"]] == ["original_part"]
        assert [part["type"] for part in drain_parts()] == ["original_part"]


@pytest.mark.parametrize("path", ["chat", "named", "bound"])
async def test_replaced_post_hook_response_withholds_private_parts(boundary, path):
    peer = boundary
    peer.policy.replace = True
    peer.policy.release.set()
    with part_collector():
        result = await peer.invoke(path)
        assert peer.tool.executions == [{}] and not peer.tool.publications
        assert result == ToolResult.ok("Replacement").to_dict()
        assert [part["type"] for part in drain_parts()] == ["original_part", "post_policy"]


@pytest.mark.parametrize("path", ["chat", "named", "bound"])
@pytest.mark.parametrize("error", [RuntimeError, KeyError, ValueError])
async def test_publication_failure_reports_ui_error_without_failing_or_replaying_completed_tool(boundary, path, error, caplog):
    peer = boundary
    peer.tool.publication_error = error("private-factory-sentinel")
    peer.policy.release.set()
    with part_collector():
        result = await peer.invoke(path)
        assert result["status"] == "ok" and result["data"] == {"value": "original"}
        assert peer.tool.executions == [{}] and peer.tool.publications == ["published-once"]
        parts = drain_parts()
        assert [part["type"] for part in parts] == ["original_part", "post_policy", "notice"]
        notice = parts[-1]["data"]
        assert notice["level"] == "error" and "Do not run it again" in notice["body"]
        assert result["parts"][-1] == parts[-1]
        assert "private-factory-sentinel" not in str(result) + str(parts) + caplog.text


@pytest.mark.parametrize("path", ["chat", "named", "bound"])
async def test_deferred_parts_use_the_canonical_envelope_sanitizer(boundary, path):
    peer = boundary
    peer.policy.release.set()
    peer.tool.publication_parts = [{"type": "private_view", "data": {},
        "id": "a" * 256, "untrusted_extra": "private-extra-sentinel"}]
    with part_collector():
        result = await peer.invoke(path)
        published = drain_parts()[-1]
        assert result["parts"][-1] == published
        assert len(published["id"]) == 128
        assert "private-extra-sentinel" not in str(result)


@pytest.mark.parametrize("path", ["chat", "named", "bound"])
@pytest.mark.parametrize("error", [asyncio.CancelledError, ExecutionAuthorityError])
async def test_publication_preserves_execution_control_errors(boundary, path, error):
    peer = boundary
    peer.tool.publication_error = error("Original execution was withdrawn")
    peer.policy.release.set()
    with part_collector():
        with pytest.raises(error):
            await peer.invoke(path)
        assert peer.tool.executions == [{}] and peer.tool.publications == ["published-once"]
        assert [part["type"] for part in drain_parts()] == ["original_part", "post_policy"]


@pytest.mark.parametrize("path", ["chat", "named", "bound"])
async def test_failed_publication_cannot_rewrite_the_reviewed_tool_envelope(boundary, path):
    peer = boundary
    peer.tool.publication_error = RuntimeError("private-factory-sentinel")
    peer.tool.mutate_before_error = True
    peer.policy.release.set()
    with part_collector():
        result = await peer.invoke(path)
        assert result["data"] == {"value": "original"}
        assert "private-mutated-factory-sentinel" not in str(result) + str(drain_parts())


@pytest.mark.parametrize("path", ["chat", "named", "bound"])
async def test_non_dictionary_policy_replacement_never_restores_prior_data(boundary, path):
    from kestrel_sdk.resources import ResourceOutcomeUnknown
    peer = boundary
    peer.policy.replace_with_none = True
    peer.policy.release.set()
    with part_collector():
        if path == "chat":
            result = await peer.invoke(path)
            assert "data" not in result and result["success"] is False
        else:
            with pytest.raises((TypeError, ResourceOutcomeUnknown)):
                await peer.invoke(path)
        assert peer.tool.executions == [{}] and not peer.tool.publications
        assert [part["type"] for part in drain_parts()] == ["original_part", "post_policy"]
        with pytest.raises(RuntimeError, match="already finalized or transferred"):
            peer.tool.active_result.finalize(peer.tool.active_result, allowed=True)


@pytest.mark.parametrize("path", ["chat", "named", "bound"])
async def test_top_level_post_policy_redaction_withholds_private_delivery(boundary, path):
    peer = boundary
    peer.policy.redact_message = True
    peer.policy.release.set()
    with part_collector():
        result = await peer.invoke(path)
        assert result["confirmation"] == "Reviewed replacement text"
        assert peer.tool.executions == [{}] and not peer.tool.publications
        assert [part["type"] for part in drain_parts()] == ["original_part", "post_policy"]


@pytest.mark.parametrize("path", ["chat", "named", "bound"])
@pytest.mark.parametrize("part_form", ["missing", "null", "empty"])
@pytest.mark.parametrize("completion", ["deny", "empty", "failure"])
async def test_optional_parts_do_not_relabel_completed_execution(boundary, path, part_form, completion):
    peer = boundary
    peer.tool.part_form = part_form
    peer.policy.release.set()
    if completion == "deny":
        peer.policy.output = HookOutput.deny("Private delivery denied")
    elif completion == "empty":
        peer.tool.publication_parts = []
    else:
        peer.tool.publication_error = RuntimeError("private-factory-sentinel")
    with part_collector():
        result = await peer.invoke(path)
        assert result["status"] == "ok" and result["data"] == {"value": "original"}
        assert peer.tool.executions == [{}]
        assert peer.tool.publications == ([] if completion == "deny" else ["published-once"])
        parts = drain_parts()
        assert [part["type"] for part in parts] == (
            ["post_policy", "notice"] if completion == "failure" else ["post_policy"])
        if completion == "failure":
            assert result["parts"] == [parts[-1]]
        else:
            assert not result.get("parts")


@pytest.mark.parametrize("path", ["chat", "named", "bound"])
async def test_policy_can_remove_original_parts_with_null(boundary, path):
    peer = boundary
    peer.policy.clear_parts = True
    peer.policy.release.set()
    with part_collector():
        result = await peer.invoke(path)
        assert result["status"] == "ok" and result["data"] == {"value": "original"}
        assert peer.tool.executions == [{}] and not peer.tool.publications
        assert "parts" not in result
        assert [part["type"] for part in drain_parts()] == ["original_part", "post_policy"]
