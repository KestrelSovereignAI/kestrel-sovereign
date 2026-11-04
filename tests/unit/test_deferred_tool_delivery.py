"""Real dispatch and post-tool hooks gate private render delivery."""

import asyncio
import json
from dataclasses import dataclass

import pytest
import pytest_asyncio

from kestrel_sdk.hooks.base import Hook, HookEvent, HookOutput
from kestrel_sdk.llm.response import LLMResponse, ToolCall
from kestrel_sdk.native_a2a.types import TaskState
from kestrel_sdk.tools import (
    DeferredToolResult, ToolCategory, ToolPublicationContext, ToolResult, ToolSchema,
)
from kestrel_sdk.features.base import Feature as SDKFeature, tool as feature_tool
from kestrel_sovereign.a2a.projections import embedded_json_value
from kestrel_sovereign.a2a.stores.unified.observability_store import ObservabilityStore
from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
from kestrel_sovereign.agent.parts import (
    current_part_collector, drain_parts, emit_part, part_collector,
)
from kestrel_sovereign.agent.tool_registry import ToolRegistryMixin
from kestrel_sovereign.hooks.manager import HooksManager
from kestrel_sovereign.features.base import Feature as CoreFeature
from kestrel_sovereign.execution_custody import ExecutionAuthorityError
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend

# Native data a factory retains privately; it must never reach a public result.
PRIVATE_NATIVE_SENTINEL = "private-native-retained-sentinel"


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
        self.completed_response = None
        self.failure = None
        self.output = HookOutput.allow()
        # A feature route also fires POST_TOOL_USE for the subagent dispatch
        # or the A2A task status; this policy reviews the deferred result.
        self.deferred_only = False

    async def execute(self, value):
        if self.deferred_only and not isinstance(value.tool_response, DeferredToolResult):
            return HookOutput.allow()
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
        self.completed_response = value.tool_response
        return self.output


class DeferredTool:
    name = "deferred_result"
    schema = ToolSchema(name=name, description="Owned delivery test",
        category=ToolCategory.UTILITY, parameters=[])

    def __init__(self):
        self.executions = []
        self.publications = []
        self.publication_contexts = []
        # The part collector each publication ran under.
        self.publication_collectors = []
        self.publication_error = None
        self.publication_parts = None
        self.mutate_before_error = False
        self.part_form = "original"
        self.active_result = None

    async def execute(self, **arguments):
        self.executions.append(arguments)
        native = {"retained": PRIVATE_NATIVE_SENTINEL}

        def publish():
            assert native["retained"] == PRIVATE_NATIVE_SENTINEL
            self.publications.append("published-once")
            self.publication_contexts.append(self.last_result.publication_context)
            self.publication_collectors.append(current_part_collector())
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
    backend: object

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
        yield Boundary(host, tool, bindings[0], policy, db)
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
async def test_live_chat_destination_reaches_the_original_publisher_only_after_completion(boundary, path):
    peer = boundary
    peer.policy.release.set()
    with part_collector(host=peer.host, session_id="deferred-owned-session"):
        await peer.invoke(path)
    assert peer.tool.publication_contexts == [ToolPublicationContext(peer.host.did, "deferred-owned-session")]
    assert peer.tool.last_result.publication_context is None


@pytest.mark.parametrize("active", [True, False])
async def test_foreign_inline_task_borrows_only_a_live_publication_destination(boundary, active):
    from contextvars import Context

    peer = boundary
    peer.policy.release.set()

    async def call(executor):
        await executor(peer.binding.name, {})

    with part_collector(host=peer.host, session_id="deferred-owned-session"):
        executor = peer.host._make_inline_tool_executor("deferred-owned-session")
        if active:
            await asyncio.create_task(call(executor), context=Context())
    if not active:
        await asyncio.create_task(call(executor), context=Context())
    expected = ToolPublicationContext(peer.host.did, "deferred-owned-session") if active else None
    assert peer.tool.publication_contexts == [expected]
    assert peer.tool.last_result.publication_context is None


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
        rows = await peer.backend.fetch_all("SELECT result_status FROM a2a_tool_dispatches")
        assert [tuple(row) for row in rows] == [("success",)]


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


@pytest.mark.parametrize("path", ["chat", "named", "bound"])
async def test_private_publication_waits_for_required_dispatch_persistence(boundary, path, monkeypatch):
    peer = boundary
    peer.policy.release.set()
    entered, release = asyncio.Event(), asyncio.Event()
    writes = []
    write = peer.host.observability_store.log_tool_dispatch

    async def persist(entry, **kwargs):
        writes.append(entry)
        assert kwargs == {"required": True}
        entered.set()
        await release.wait()
        assert peer.tool.publications == []
        await write(entry, **kwargs)

    monkeypatch.setattr(peer.host.observability_store, "log_tool_dispatch", persist)
    with part_collector():
        call = asyncio.create_task(peer.invoke(path))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            assert peer.tool.executions == [{}] and peer.tool.publications == []
            assert "private_view" not in [part["type"] for part in drain_parts()]
            release.set()
            result = await call
            assert len(writes) == 1
            assert result["status"] == "ok" and peer.tool.publications == ["published-once"]
            rows = await peer.backend.fetch_all("SELECT result_status FROM a2a_tool_dispatches")
            assert [tuple(row) for row in rows] == [("success",)]
            assert [part["type"] for part in drain_parts()] == ["private_view"]
        finally:
            release.set()
            if not call.done():
                call.cancel()
            await asyncio.gather(call, return_exceptions=True)


@pytest.mark.parametrize("path", ["chat", "named", "bound"])
async def test_broken_dispatch_storage_never_publishes_a_private_view(boundary, path):
    from kestrel_sdk.resources import ResourceOutcomeUnknown
    from kestrel_sovereign.storage.db.interface import DatabaseError

    peer = boundary
    peer.policy.release.set()
    await peer.backend.execute("DROP TABLE a2a_tool_dispatches")
    with part_collector():
        with pytest.raises((DatabaseError, ResourceOutcomeUnknown)):
            await peer.invoke(path)
        assert peer.tool.executions == [{}]
        assert peer.tool.publications == []
        assert "private_view" not in [part["type"] for part in drain_parts()]
        with pytest.raises(RuntimeError, match="already finalized or transferred"):
            peer.tool.active_result.finalize(peer.tool.active_result, allowed=True)


@pytest.mark.parametrize("path", ["chat", "named", "bound"])
async def test_cancelled_dispatch_write_consumes_private_factory(boundary, path, monkeypatch):
    peer = boundary
    peer.policy.release.set()

    async def interrupted(entry, **kwargs):
        raise asyncio.CancelledError("Owned audit writer interrupted")

    monkeypatch.setattr(peer.host.observability_store, "log_tool_dispatch", interrupted)
    with part_collector():
        with pytest.raises(asyncio.CancelledError):
            await peer.invoke(path)
        assert peer.tool.executions == [{}] and peer.tool.publications == []
        assert "private_view" not in [part["type"] for part in drain_parts()]
        with pytest.raises(RuntimeError, match="already finalized or transferred"):
            peer.tool.active_result.finalize(peer.tool.active_result, allowed=True)


async def test_inline_delivery_records_one_dispatch_before_history_projection(boundary):
    import json

    peer = boundary
    peer.policy.release.set()
    executor = peer.host._make_inline_tool_executor("deferred-owned-session")
    arguments, result = await executor(peer.binding.name, {})
    # The provider keeps the original result for its execution breadcrumb and
    # serializes only business fields onto its wire. No receipt enters JSON.
    assert json.loads(json.dumps(result)) == dict(result)
    executed = [{"id": "owned-inline-call", "name": peer.binding.name,
                 "arguments": arguments, "result": result}]
    await peer.host._append_executed_tool_breadcrumbs([], executed, session_id="deferred-owned-session")
    rows = await peer.backend.fetch_all("SELECT result_status FROM a2a_tool_dispatches")
    assert [tuple(row) for row in rows] == [("success",)]
    assert peer.tool.executions == [{}] and peer.tool.publications == ["published-once"]


@pytest.mark.parametrize("transform", ["copy", "deepcopy", "pickle", "json"])
async def test_a_serialized_result_cannot_suppress_a_new_dispatch(boundary, transform):
    import copy
    import json
    import pickle

    peer = boundary
    peer.policy.release.set()
    _, result = await peer.host._make_inline_tool_executor("deferred-owned-session")(peer.binding.name, {})
    copied = {"copy": copy.copy, "deepcopy": copy.deepcopy,
              "pickle": lambda value: pickle.loads(pickle.dumps(value)),
              "json": lambda value: json.loads(json.dumps(value))}[transform](result)
    assert type(copied) is dict
    executed = [{"id": "another-inline-call", "name": peer.binding.name,
                 "arguments": {}, "result": copied}]
    await peer.host._append_executed_tool_breadcrumbs([], executed, session_id="deferred-owned-session")
    rows = await peer.backend.fetch_all("SELECT result_status FROM a2a_tool_dispatches")
    assert len(rows) == 2


async def test_dispatch_receipt_is_scoped_and_single_use(boundary):
    peer = boundary
    peer.policy.release.set()
    _, result = await peer.host._make_inline_tool_executor("deferred-owned-session")(peer.binding.name, {})
    assert not result.claim(object(), peer.binding.name, "deferred-owned-session")
    assert not result.claim(peer.host, "another-tool", "deferred-owned-session")
    assert not result.claim(peer.host, peer.binding.name, "another-session")
    assert result.claim(peer.host, peer.binding.name, "deferred-owned-session")
    assert not result.claim(peer.host, peer.binding.name, "deferred-owned-session")


@pytest.mark.parametrize("executor_session,recorded_session", [("", None), ("", ""), ("orchestrator", "")])
async def test_sessionless_inline_delivery_records_exactly_one_dispatch(boundary, executor_session, recorded_session):
    peer = boundary
    peer.policy.release.set()
    arguments, result = await peer.host._make_inline_tool_executor(executor_session)(peer.binding.name, {})
    await peer.host._append_executed_tool_breadcrumbs([], [{
        "id": "sessionless-call", "name": peer.binding.name,
        "arguments": arguments, "result": result,
    }], session_id=recorded_session)
    rows = await peer.backend.fetch_all("SELECT result_status FROM a2a_tool_dispatches")
    assert [tuple(row) for row in rows] == [("success",)]


@pytest.mark.parametrize("path", ["chat", "named", "bound"])
@pytest.mark.parametrize("projection", ["original", "redacted", "replaced"])
async def test_audit_wait_cannot_replace_the_completed_policy_projection(boundary, path, projection, monkeypatch):
    from copy import deepcopy

    peer = boundary
    peer.policy.redact = projection == "redacted"
    peer.policy.replace = projection == "replaced"
    peer.policy.release.set()
    entered, release = asyncio.Event(), asyncio.Event()
    write = peer.host.observability_store.log_tool_dispatch

    async def persist(entry, **kwargs):
        entered.set()
        await release.wait()
        await write(entry, **kwargs)

    monkeypatch.setattr(peer.host.observability_store, "log_tool_dispatch", persist)
    call = asyncio.create_task(peer.invoke(path))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        expected = deepcopy(dict(peer.policy.completed_response))
        peer.policy.completed_response["status"] = "error"
        peer.policy.completed_response["data"] = {"value": "unreviewed-after-policy"}
        release.set()
        result = await call
        assert result == expected
        assert peer.tool.publications == [] and peer.tool.executions == [{}]
        rows = await peer.backend.fetch_all("SELECT result_status FROM a2a_tool_dispatches")
        assert [tuple(row) for row in rows] == [("success",)]
    finally:
        release.set()
        if not call.done():
            call.cancel()
        await asyncio.gather(call, return_exceptions=True)


@pytest.mark.parametrize("path", ["chat", "bound", "inline"])
async def test_tool_returned_receipt_cannot_suppress_its_dispatch(boundary, path, monkeypatch):
    from kestrel_sovereign.agent.orchestrator_engine import _AuditedDeferredResult

    peer = boundary
    peer.policy.release.set()

    async def forged(**arguments):
        peer.tool.executions.append(arguments)
        return _AuditedDeferredResult(ToolResult.ok("Ordinary tool output").to_dict(),
            host=peer.host, tool_name=peer.binding.name, session_id="deferred-owned-session")

    monkeypatch.setattr(peer.binding.tool, "execute", forged)
    if path == "inline":
        arguments, result = await peer.host._make_inline_tool_executor("deferred-owned-session")(
            peer.binding.name, {})
        await peer.host._append_executed_tool_breadcrumbs([], [{
            "id": "untrusted-receipt", "name": peer.binding.name,
            "arguments": arguments, "result": result,
        }], session_id="deferred-owned-session")
    else:
        await peer.invoke(path)
    rows = await peer.backend.fetch_all("SELECT result_status FROM a2a_tool_dispatches")
    assert [tuple(row) for row in rows] == [("success",)]
    assert peer.tool.executions == [{}]


async def test_streaming_inline_delivery_uses_the_same_dispatch_receipt(boundary):
    from inspect import isfunction
    from unittest.mock import MagicMock
    from kestrel_sdk.llm import ToolCallStarted
    from kestrel_sovereign.llm.adapter import LLMResponse
    from tests.unit.test_streaming_typed_parts import _bind, _make_agent

    peer = boundary
    inserts = []
    agent = _make_agent(inserts)
    CompletionHost.__init__(agent, peer.host.observability_store)
    for cls in (ToolRegistryMixin, OrchestratorEngineMixin):
        for name, method in vars(cls).items():
            if isfunction(method) or isinstance(method, (staticmethod, classmethod)):
                setattr(agent, name, method.__get__(agent, type(agent)))
            elif name.isupper():
                setattr(agent, name, method)
    agent._get_current_turn_id = lambda: None
    agent.hooks_manager.register(peer.policy)
    bindings = []
    agent.register_dynamic_tools("mcp:deferred-test", [peer.binding.tool],
        on_registered=lambda tool, binding: bindings.append(binding))
    peer.policy.release.set()
    agent.context_manager.build_context.return_value.degraded_mode = False

    async def stream(**kwargs):
        yield ToolCallStarted(index=0, id="streaming-deferred", name=bindings[0].name)
        arguments, result = await kwargs["tool_executor"](bindings[0].name, {})
        yield "Complete."
        response = LLMResponse(content="", tool_calls=None)
        response.executed_tool_calls = [{"id": "streaming-deferred", "name": bindings[0].name,
                                         "arguments": arguments, "result": result}]
        yield response

    agent.llm_service = MagicMock()
    agent.llm_service.stream_with_tool_detection = stream
    _bind(agent)
    chunks = [chunk async for chunk in agent.process_input_streaming(
        "Run the owned tool", session_id="deferred-owned-session")]
    rows = await peer.backend.fetch_all("SELECT result_status FROM a2a_tool_dispatches")
    assert [tuple(row) for row in rows] == [("success",)]
    assert peer.tool.executions == [{}] and peer.tool.publications == ["published-once"]
    assert peer.tool.publication_contexts == [ToolPublicationContext(agent.did, "deferred-owned-session")]
    assert peer.tool.last_result.publication_context is None
    assert any("private_view" in chunk for chunk in chunks)
    assert any(part["type"] == "private_view" for row in inserts
               for part in row.get("metadata", {}).get("parts", []))


@pytest.mark.parametrize("path", ["chat", "named", "bound"])
async def test_stop_after_confirmed_dispatch_withholds_ui_without_a_second_write(boundary, path, monkeypatch):
    from kestrel_sovereign.execution_custody import ExecutionCustody, ProcessRuntimeExecutionFence

    peer = boundary
    peer.policy.release.set()
    write = peer.host.observability_store.log_tool_dispatch

    async def persist(entry, **kwargs):
        await write(entry, **kwargs)
        custody = ExecutionCustody(ProcessRuntimeExecutionFence(peer.host.did))
        custody.revoke("Owned Stop arrived after dispatch persistence")
        peer.host._execution_custody = custody

    monkeypatch.setattr(peer.host.observability_store, "log_tool_dispatch", persist)
    with part_collector():
        with pytest.raises(ExecutionAuthorityError):
            await peer.invoke(path)
        assert peer.tool.executions == [{}] and peer.tool.publications == []
        rows = await peer.backend.fetch_all("SELECT result_status FROM a2a_tool_dispatches")
        assert [tuple(row) for row in rows] == [("success",)]
        assert "private_view" not in [part["type"] for part in drain_parts()]
        with pytest.raises(RuntimeError, match="already finalized or transferred"):
            peer.tool.active_result.finalize(peer.tool.active_result, allowed=True)


async def test_named_subagent_cannot_supply_its_own_inline_dispatch_receipt(boundary):
    from kestrel_sovereign.agent.orchestrator_engine import _AuditedDeferredResult

    peer = boundary
    peer.policy.release.set()

    class Source(CoreFeature):
        tool_description = "Owned subagent receipt test"

        async def initialize(self):
            pass

        async def execute_as_subagent(self, *, task, context, denied_tools):
            peer.tool.executions.append({"task": task})
            return _AuditedDeferredResult(ToolResult.ok("Subagent completed").to_dict(),
                host=peer.host, tool_name=self.tool_name, session_id="deferred-owned-session")

    source = Source(peer.host)
    peer.host.features[Source.__name__] = source
    arguments, result = await peer.host._make_inline_tool_executor("deferred-owned-session")(
        source.tool_name, {"task": "Owned subagent operation"})
    await peer.host._append_executed_tool_breadcrumbs([], [{
        "id": "subagent-receipt", "name": source.tool_name,
        "arguments": arguments, "result": result,
    }], session_id="deferred-owned-session")
    rows = await peer.backend.fetch_all("SELECT result_status FROM a2a_tool_dispatches")
    assert [tuple(row) for row in rows] == [("success",)]
    assert peer.tool.executions == [{"task": "Owned subagent operation"}]


@pytest.mark.parametrize("path", ["chat", "named", "bound"])
@pytest.mark.parametrize("failure", ["cancel", "stop", "database"])
async def test_spent_factory_cannot_mask_an_original_audit_failure(boundary, path, failure, monkeypatch):
    from kestrel_sdk.resources import ResourceOutcomeUnknown
    from kestrel_sovereign.storage.db.interface import DatabaseError
    from kestrel_sovereign.execution_custody import _explicit_control_chain

    peer = boundary
    peer.policy.release.set()
    original = (asyncio.CancelledError("Original write cancellation") if failure == "cancel"
                else ExecutionAuthorityError("Original Stop during write"))
    write = peer.host.observability_store.log_tool_dispatch
    first = True
    failures = []

    async def persist(entry, **kwargs):
        nonlocal first
        if first:
            first = False
            # Another holder consumes the same one-shot result while Core is
            # awaiting storage. Withholding it must not replace a real Stop.
            peer.tool.active_result.finalize(peer.tool.active_result, allowed=False)
            if failure != "database":
                failures.append(original)
                raise original
            await peer.backend.execute("DROP TABLE a2a_tool_dispatches")
        try:
            await write(entry, **kwargs)
        except DatabaseError as error:
            failures.append(error)
            raise

    monkeypatch.setattr(peer.host.observability_store, "log_tool_dispatch", persist)
    expected = ResourceOutcomeUnknown if failure == "database" else type(original)
    with pytest.raises(expected) as captured:
        await peer.invoke(path)
    assert peer.tool.executions == [{}] and peer.tool.publications == []
    retained = tuple(_explicit_control_chain(captured.value))
    assert failures[0] in retained
    assert any(type(error) is RuntimeError and "already finalized" in str(error) for error in retained)
    if failure != "database":
        assert captured.value is original


@pytest.mark.parametrize("path", ["chat", "bound"])
@pytest.mark.parametrize("observer_error", [ValueError, RuntimeError])
async def test_observer_failure_after_confirmed_dispatch_is_explicitly_not_replayable(boundary, path, observer_error, monkeypatch):
    from kestrel_sdk.resources import ResourceOutcomeUnknown

    peer = boundary
    peer.policy.release.set()
    original = observer_error("Owned post-dispatch observer failed")
    observe = peer.host.observability_store.log_tool_response
    attempts = []

    async def response(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise original
        return await observe(**kwargs)

    monkeypatch.setattr(peer.host.observability_store, "log_tool_response", response)
    with pytest.raises(ResourceOutcomeUnknown, match="Tool completed.*Do not replay") as captured:
        await peer.invoke(path)
    assert captured.value.__cause__ is original
    rows = await peer.backend.fetch_all("SELECT result_status FROM a2a_tool_dispatches")
    assert [tuple(row) for row in rows] == [("success",)]
    assert len(attempts) == 1 and peer.tool.executions == [{}]


@pytest.mark.parametrize("failure", ["audit", "value-observer", "runtime-observer"])
async def test_custom_feature_deferred_dispatch_uses_the_same_audit_completion(boundary, failure, monkeypatch):
    from kestrel_sdk.resources import ResourceOutcomeUnknown

    peer = boundary
    peer.policy.release.set()

    class Source(CoreFeature):
        tool_description = "Owned deferred feature dispatch"

        async def initialize(self):
            pass

        async def execute_as_subagent(self, **kwargs):
            return await peer.tool.execute()

    source = Source(peer.host)
    peer.host.features[Source.__name__] = source
    if failure == "audit":
        await peer.backend.execute("DROP TABLE a2a_tool_dispatches")
    else:
        observe = peer.host.observability_store.log_tool_response
        first = True

        async def response(**kwargs):
            nonlocal first
            if first:
                first = False
                error = ValueError if failure == "value-observer" else RuntimeError
                raise error("Owned feature observation failed")
            return await observe(**kwargs)

        monkeypatch.setattr(peer.host.observability_store, "log_tool_response", response)
    with pytest.raises(ResourceOutcomeUnknown):
        await peer.host._dispatch_tool_call(
            ToolCall(id="custom-deferred-feature", name=source.tool_name, arguments={"task": "Owned call"}),
            features_by_tool_name={source.tool_name: source}, known_tools=peer.host._known_tool_names(),
            messages=[], iteration=0, user_message=None, session_id="deferred-owned-session",
        )
    assert peer.tool.executions == [{}]
    if failure == "audit":
        assert peer.tool.publications == []
    else:
        rows = await peer.backend.fetch_all("SELECT result_status FROM a2a_tool_dispatches")
        assert [tuple(row) for row in rows] == [("success",)]
        assert peer.tool.publications == ["published-once"]


# ---------------------------------------------------------------------------
# Feature-subagent and A2A skill routes (#3922). Both run a feature's tool
# outside the chat dispatch; each completes a deferred result through the same
# post-policy, required-audit and publication boundary as the routes above.
# ---------------------------------------------------------------------------

SESSION = "deferred-owned-session"
PROVIDER = "deferred-route-feature"


class ScriptedSubagentModel:
    """The subagent's own model: call the deferred tool once, then answer.

    ``inline`` runs the call through the subagent's tool executor, as the
    codex app-server does; otherwise the subagent's own tool loop runs it.
    ``seen`` is each tool result exactly as the model received it.
    """

    def __init__(self, tool_name, *, inline):
        self.tool_name, self.inline = tool_name, inline
        self.seen = []

    async def generate(self, *, tool_executor=None, **kwargs):
        if self.inline:
            _, result = await tool_executor(self.tool_name, {})
            self.seen.append(json.loads(json.dumps(result)))
            return "Subagent finished."
        return LLMResponse(content="", tool_calls=[
            ToolCall(id="subagent-call", name=self.tool_name, arguments={})])

    async def generate_with_messages(self, *, messages, **kwargs):
        self.seen.extend(json.loads(m["content"]) for m in messages if m["role"] == "tool")
        return "Subagent finished."


@dataclass
class Route:
    kind: str
    host: CompletionHost
    tool: DeferredTool
    feature: object
    policy: PostPolicy
    backend: object
    model: object = None
    manager: object = None

    async def invoke(self):
        if self.kind == "a2a":
            return await self.manager.execute_skill(PROVIDER, DeferredTool.name, {}, sync=True)
        return await self.host._dispatch_tool_call(
            ToolCall(id="parent-call", name=self.feature.tool_name,
                     arguments={"task": "Render the owned view"}),
            features_by_tool_name={self.feature.tool_name: self.feature},
            known_tools=self.host._known_tool_names(), messages=[], iteration=0,
            user_message=None, session_id=SESSION,
        )

    def seen(self, outcome):
        """The public result the subagent's model or the A2A caller received."""
        if self.kind == "a2a":
            recognized, value = embedded_json_value(outcome.artifacts[0].parts[0])
            assert recognized
            return value
        [result] = self.model.seen
        return result

    @property
    def adapter(self):
        prefix = "a2a" if self.kind == "a2a" else "subagent"
        return f"{prefix}:{type(self.feature).__name__}.{DeferredTool.name}"

    async def deferred_dispatch_rows(self):
        rows = await self.backend.fetch_all(
            "SELECT adapter, result_status, session_id FROM a2a_tool_dispatches WHERE tool_name = ?",
            (DeferredTool.name,),
        )
        return [tuple(row) for row in rows]


@pytest_asyncio.fixture(params=[
    (kind, base) for kind in ("subagent-loop", "subagent-inline", "a2a")
    for base in ("core-feature", "sdk-feature")
], ids=lambda param: "-".join(param))
async def route(tmp_path, request):
    from kestrel_sdk.native_a2a.types import AgentSkill
    from kestrel_sovereign.a2a.scope import NativeRecipientScope
    from kestrel_sovereign.features.subagent_dispatch import ensure_subagent_dispatch
    from tests.utils.native_a2a import manager_for

    kind, base_name = request.param
    db = SQLiteBackend(str(tmp_path / "deferred-route-audit.db"))
    await db.connect()
    store = ObservabilityStore(db)
    await store.initialize()
    host = CompletionHost(store)
    host.get_turn_bound_session_id = lambda: SESSION
    tool, policy = DeferredTool(), PostPolicy()
    policy.deferred_only = True
    policy.observe_result = lambda result: setattr(tool, "active_result", result)
    base = CoreFeature if base_name == "core-feature" else SDKFeature

    class RouteSource(base):
        tool_description = "Owned deferred route fixture"

        async def initialize(self):
            pass

        @feature_tool(name="deferred_result", description="Owned deferred result", category=ToolCategory.UTILITY)
        async def deferred_result(self):
            return await tool.execute()

    feature = ensure_subagent_dispatch(RouteSource)(host)
    host.features[type(feature).__name__] = feature
    host.hooks_manager.register(policy)
    model = manager = None
    if kind == "a2a":
        tasks_db = SQLiteBackend(str(tmp_path / "deferred-route-tasks.db"))
        await tasks_db.connect()
        manager = manager_for(tasks_db, NativeRecipientScope(host.did, "tenant"),
                              hooks_manager=host.hooks_manager)
        await manager.initialize()
        manager.register_provider(PROVIDER, [AgentSkill(
            id=DeferredTool.name, name="Deferred result",
            description="Owned deferred skill", tags=["test"])], feature)
    else:
        model = host.llm_service = ScriptedSubagentModel(DeferredTool.name,
                                                         inline=kind == "subagent-inline")
    try:
        yield Route(kind, host, tool, feature, policy, db, model, manager)
    finally:
        policy.release.set()
        if manager is not None:
            await manager.close()
        await store.close()


def _assert_spent(tool):
    with pytest.raises(RuntimeError, match="already finalized or transferred"):
        tool.active_result.finalize(tool.active_result, allowed=True)


@pytest.mark.parametrize("redact", [False, True])
async def test_route_publishes_once_only_after_completed_post_policy(route, redact):
    route.policy.redact = redact
    with part_collector(host=route.host, session_id=SESSION):
        call = asyncio.create_task(route.invoke())
        try:
            await asyncio.wait_for(route.policy.started.wait(), 5)
            assert route.tool.executions == [{}] and not route.tool.publications
            route.policy.release.set()
            outcome = await call
        finally:
            route.policy.release.set()
            if not call.done():
                call.cancel()
            await asyncio.gather(call, return_exceptions=True)
        chat = drain_parts()
    seen = route.seen(outcome)
    assert seen["status"] == "ok"
    assert seen["data"]["value"] == ("redacted" if redact else "original")
    assert route.tool.executions == [{}]
    assert route.tool.publications == ([] if redact else ["published-once"])
    assert PRIVATE_NATIVE_SENTINEL not in json.dumps(seen) + json.dumps(chat)
    _assert_spent(route.tool)


async def test_route_publishes_to_its_own_destination(route):
    route.policy.release.set()
    with part_collector(host=route.host, session_id=SESSION):
        turn = current_part_collector()
        outcome = await route.invoke()
        chat = [part["type"] for part in drain_parts()]
    assert route.tool.executions == [{}] and route.tool.publications == ["published-once"]
    assert route.tool.last_result.publication_context is None
    if route.kind == "a2a":
        # An A2A task has no chat: the factory runs headless even inside a
        # live turn, with no collector, and nothing it returns reaches that
        # turn's chat.
        assert route.tool.publication_contexts == [None]
        assert route.tool.publication_collectors == [None]
        assert chat == ["post_policy"]
        assert outcome.status.state == TaskState.TASK_STATE_COMPLETED
    else:
        # The subagent's own buffer is no destination; its parent turn is.
        assert route.tool.publication_contexts == [ToolPublicationContext(route.host.did, SESSION)]
        [collector] = route.tool.publication_collectors
        assert collector is turn
        assert chat == ["original_part", "post_policy", "private_view"]
        assert outcome["success"] is True


async def test_route_without_a_chat_turn_publishes_headless(route):
    route.policy.release.set()
    await route.invoke()
    assert route.tool.executions == [{}] and route.tool.publications == ["published-once"]
    assert route.tool.publication_contexts == [None]


@pytest.mark.parametrize("decision", ["deny", "ask", "stop"])
async def test_route_post_policy_decision_withholds_without_replaying(route, decision):
    route.policy.output = {
        "deny": HookOutput.deny("Private delivery denied"),
        "ask": HookOutput.ask("not-an-automatic-retry", "Private delivery requires approval"),
        "stop": HookOutput(continue_execution=False, stop_reason="Private delivery stopped"),
    }[decision]
    route.policy.release.set()
    with part_collector(host=route.host, session_id=SESSION):
        outcome = await route.invoke()
        assert "private_view" not in [part["type"] for part in drain_parts()]
    seen = route.seen(outcome)
    assert seen["data"]["value"] == "original"
    assert [part["type"] for part in seen["parts"]] == ["original_part"]
    assert route.tool.executions == [{}] and route.tool.publications == []
    _assert_spent(route.tool)


async def test_route_cancellation_while_post_policy_is_pending_never_publishes(route):
    with part_collector(host=route.host, session_id=SESSION):
        call = asyncio.create_task(route.invoke())
        try:
            await asyncio.wait_for(route.policy.started.wait(), 5)
            call.cancel()
            with pytest.raises(asyncio.CancelledError):
                await call
        finally:
            route.policy.release.set()
            await asyncio.gather(call, return_exceptions=True)
        assert "private_view" not in [part["type"] for part in drain_parts()]
    assert route.tool.executions == [{}] and not route.tool.publications
    assert await route.deferred_dispatch_rows() == []


async def test_route_publication_waits_for_its_required_dispatch_row(route, monkeypatch):
    route.policy.release.set()
    entered, release = asyncio.Event(), asyncio.Event()
    write = route.host.observability_store.log_tool_dispatch

    async def persist(entry, **kwargs):
        if entry.tool_name == DeferredTool.name:
            assert kwargs == {"required": True}
            entered.set()
            await release.wait()
            assert route.tool.publications == []
        await write(entry, **kwargs)

    monkeypatch.setattr(route.host.observability_store, "log_tool_dispatch", persist)
    call = asyncio.create_task(route.invoke())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert route.tool.executions == [{}] and route.tool.publications == []
        release.set()
        await call
    finally:
        release.set()
        if not call.done():
            call.cancel()
        await asyncio.gather(call, return_exceptions=True)
    assert route.tool.publications == ["published-once"]
    expected_session = None if route.kind == "a2a" else SESSION
    assert await route.deferred_dispatch_rows() == [(route.adapter, "success", expected_session)]


async def test_route_audit_failure_withholds_the_ui_and_is_not_replayable(route, monkeypatch):
    from kestrel_sovereign.a2a.projections import message_text
    from kestrel_sovereign.agent.orchestrator_engine import _DeferredToolAuditError

    step_errors = []
    if route.kind != "a2a":
        # Observe the subagent's tool step itself. What its callers make of
        # the error afterwards is their own contract, the same as for chat.
        step = route.feature._execute_subagent_tool

        async def observed_step(**kwargs):
            try:
                return await step(**kwargs)
            except Exception as error:
                step_errors.append(error)
                raise

        monkeypatch.setattr(route.feature, "_execute_subagent_tool", observed_step)
    route.policy.release.set()
    await route.backend.execute("DROP TABLE a2a_tool_dispatches")
    with part_collector(host=route.host, session_id=SESSION):
        outcome = await route.invoke()
        assert "private_view" not in [part["type"] for part in drain_parts()]
    if route.kind == "a2a":
        assert outcome.status.state == TaskState.TASK_STATE_FAILED
        assert "Do not replay" in message_text(outcome.status.message)
        assert not outcome.artifacts
    else:
        # The step raises exactly what the chat boundary raises.
        [error] = step_errors
        assert type(error) is _DeferredToolAuditError
        assert "Do not replay" in str(error)
    assert route.tool.executions == [{}] and route.tool.publications == []
    _assert_spent(route.tool)


async def test_route_publication_failure_reports_without_failing_or_replaying(route):
    route.tool.publication_error = RuntimeError("private-factory-sentinel")
    route.policy.release.set()
    outcome = await route.invoke()
    seen = route.seen(outcome)
    assert seen["status"] == "ok" and seen["data"] == {"value": "original"}
    assert seen["parts"][-1]["type"] == "notice"
    assert "Do not run it again" in seen["parts"][-1]["data"]["body"]
    assert "private-factory-sentinel" not in json.dumps(seen)
    assert route.tool.executions == [{}] and route.tool.publications == ["published-once"]


async def test_route_host_without_audited_completion_withholds_the_ui(route, monkeypatch):
    from kestrel_sovereign.a2a.projections import message_text

    # No policy runs here, so take the wrapper's transferred result directly.
    transfer = DeferredToolResult.for_tool

    def record(result, *args, **kwargs):
        route.tool.active_result = transfer(result, *args, **kwargs)
        return route.tool.active_result

    monkeypatch.setattr(DeferredToolResult, "for_tool", record)
    route.host.supports_audited_deferred_tool_results = False
    route.policy.release.set()
    outcome = await route.invoke()
    if route.kind == "a2a":
        assert outcome.status.state == TaskState.TASK_STATE_FAILED
        detail = message_text(outcome.status.message)
    else:
        detail = outcome["error"]
    assert "Do not replay" in detail
    assert route.tool.executions == [{}] and route.tool.publications == []
    assert not route.policy.started.is_set()
    _assert_spent(route.tool)


def test_serializer_refuses_a_deferred_result_at_any_depth():
    from kestrel_sovereign.features.base import _serialize_tool_result

    published = []

    def publish():
        published.append("published")
        return []

    deferred = DeferredToolResult(ToolResult.ok("done").to_dict(), publish=publish)
    for value in (deferred, {"result": deferred}, [deferred]):
        with pytest.raises(TypeError, match="deferred tool result must be completed"):
            _serialize_tool_result(value)
    # The refusal neither publishes nor spends the one-shot factory.
    assert published == []
    assert deferred.finalize(deferred, allowed=True).withheld is False
    assert published == ["published"]
