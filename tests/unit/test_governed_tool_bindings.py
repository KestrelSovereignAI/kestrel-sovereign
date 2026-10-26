"""External tool bindings use the real dispatcher, hooks, and durable audit."""

import asyncio
import json
from copy import deepcopy

import pytest
import pytest_asyncio
from kestrel_sdk.hooks.base import Hook, HookEvent, HookOutput
from kestrel_sdk.resources import ResourceConflict, ResourceOutcomeUnknown, ResourceUnsupported
from kestrel_sdk.tools.base import ToolCategory, ToolParameter, ToolSchema

from kestrel_sovereign.a2a.stores.unified.observability_store import ObservabilityStore
from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
from kestrel_sovereign.agent.tool_registry import ToolRegistryMixin
from kestrel_sovereign.hooks.manager import HooksManager
from kestrel_sovereign.features.security.approval_queue import ApprovalQueue, ApprovalStatus
from kestrel_sovereign.features.security.hooks import SecurityHook
from kestrel_sovereign.features.security.permissions import PermissionLevel, PermissionStore
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from kestrel_sovereign.agent import orchestrator_engine
from kestrel_sovereign.security.input_guardrails import MAX_TOOL_ARG_LENGTH


class _Tool:
    def __init__(self, name="view.update", *, visible=False):
        self.name = name
        self.schema = ToolSchema(
            name, "Update this view", ToolCategory.UTILITY,
            parameters=[ToolParameter("count", "integer", "Count", required=True)],
            model_visible=visible,
        )
        self.calls = []

    async def execute(self, **args):
        self.calls.append(args)
        return {"success": True, "count": args["count"]}


class _Policy(Hook):
    def __init__(self, output, *, before=None):
        super().__init__("app-policy", [HookEvent.PRE_TOOL_USE], awaits_user_input=True)
        self.output, self.before, self.inputs = output, before, []

    async def execute(self, value):
        self.inputs.append(value)
        if self.before is not None:
            await self.before()
        return self.output


class _Agent(OrchestratorEngineMixin, ToolRegistryMixin):
    def __init__(self, store):
        self.did = "did:test:apps-owner"
        self.features = {}
        self._direct_tools, self._tool_to_feature, self._explored_features = {}, {}, {}
        self._direct_tool_defs, self._pinned_features = [], set()
        self.hooks_manager, self.observability_store = HooksManager(), store


class _Store(ObservabilityStore):
    """Real store with a scheduling point after the admission write."""

    on_admitted = None
    on_dispatch = None
    fail_success_response_once = False

    async def log_tool_call(self, *args, **kwargs):
        result = await super().log_tool_call(*args, **kwargs)
        if self.on_admitted is not None:
            await self.on_admitted()
        return result

    async def log_tool_response(self, **kwargs):
        if self.fail_success_response_once and kwargs["success"]:
            self.fail_success_response_once = False
            # Exercise an actual backend QueryError after the tool's effect.
            # The next ordinary response write succeeds on the same database.
            await self._backend.execute("UPDATE deliberately_missing_table SET value = 1")
        return await super().log_tool_response(**kwargs)

    async def log_tool_dispatch(self, entry, *, required=False):
        if self.on_dispatch is not None:
            await self.on_dispatch()
        return await super().log_tool_dispatch(entry, required=required)


@pytest_asyncio.fixture
async def host(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "app-dispatches.db"))
    await backend.connect()
    store = _Store(backend)
    await store.initialize()
    try:
        yield _Agent(store), backend
    finally:
        await store.close()


def _mount(agent, tool, owner="mcp:views"):
    bindings = []
    count = agent.register_dynamic_tools(
        owner, [tool], on_registered=lambda mounted, binding: bindings.append((mounted, binding)),
    )
    assert count == 1
    assert bindings[0][0] is tool
    return bindings[0][1]


@pytest.mark.asyncio
async def test_app_only_tool_has_governed_registration_without_model_projection(host):
    agent, backend = host
    app = _Tool()
    binding = _mount(agent, app)
    visible = _Tool("other", visible=True)
    _mount(agent, visible, owner="mcp:other")
    assert binding.is_current
    assert binding.name == "view_update"
    assert set(agent._direct_tools) == {"view_update", "other"}
    for schemas in [agent._build_all_tools(), agent.build_progressive_tool_schemas()]:
        assert [item["function"]["name"] for item in schemas] == ["other"]
    policy = _Policy(HookOutput.modify({"count": 7}))
    agent.hooks_manager.register(policy)
    result = await binding.invoke({"count": 3}, session_id="mcp-app:view-1", source="mcp-app")
    assert result == {"success": True, "count": 7}
    assert app.calls == [{"count": 7}]
    assert policy.inputs[0].feature_name == "mcp:views"
    row = await backend.fetch_one(
        "SELECT agent_did, session_id, tool_name, args_redacted, result_status "
        "FROM a2a_tool_dispatches",
    )
    assert tuple(row[:3]) == (agent.did, "mcp-app:view-1", binding.name)
    assert json.loads(row[3]) == {"count": 7}
    assert row[4] == "success"


@pytest.mark.asyncio
@pytest.mark.parametrize("output", [HookOutput.deny("host denied"), HookOutput.ask("approval-1")])
async def test_bound_call_does_not_bypass_deny_or_pending_approval(host, output):
    agent, backend = host
    tool = _Tool()
    binding = _mount(agent, tool)
    agent.hooks_manager.register(_Policy(output))
    result = await binding.invoke({"count": 3}, session_id="mcp-app:view-1")
    assert result["success"] is False
    assert tool.calls == []
    rows = await backend.fetch_all("SELECT result_status FROM a2a_tool_dispatches")
    assert len(rows) == 1
    assert rows[0][0] != "success"


@pytest.mark.asyncio
async def test_same_tool_object_cannot_revive_an_unregistered_binding(host):
    agent, _ = host
    tool = _Tool()
    old = _mount(agent, tool)
    assert agent.unregister_dynamic_tools("mcp:views") == 1
    new = _mount(agent, tool)
    assert old.is_current is False
    assert new.is_current is True
    with pytest.raises(ResourceConflict):
        await old.invoke({"count": 3}, session_id="mcp-app:view-1")
    assert tool.calls == []


@pytest.mark.asyncio
async def test_replacement_during_approval_cannot_execute_either_registration(host):
    agent, backend = host
    original, replacement = _Tool(), _Tool()
    binding = _mount(agent, original)
    entered, resume = asyncio.Event(), asyncio.Event()

    async def wait_for_approval():
        entered.set()
        await resume.wait()

    agent.hooks_manager.register(_Policy(HookOutput.allow(), before=wait_for_approval))
    call = asyncio.create_task(binding.invoke({"count": 3}, session_id="mcp-app:view-1"))
    try:
        await entered.wait()
        agent.unregister_dynamic_tools("mcp:views")
        new = _mount(agent, replacement)
        assert new.name == binding.name
        resume.set()
        with pytest.raises(ResourceConflict):
            await call
        assert original.calls == replacement.calls == []
        row = await backend.fetch_one("SELECT error_class FROM a2a_tool_dispatches")
        assert row[0] == "ResourceConflict"
    finally:
        resume.set()
        if not call.done():
            call.cancel()
        await asyncio.gather(call, return_exceptions=True)


@pytest.mark.asyncio
async def test_failed_structured_audit_is_unknown_and_never_reexecutes(host):
    agent, backend = host
    tool = _Tool()
    binding = _mount(agent, tool)
    await backend.execute("DROP TABLE a2a_tool_dispatches")
    with pytest.raises(ResourceOutcomeUnknown, match="audit"):
        await binding.invoke({"count": 3}, session_id="mcp-app:view-1")
    assert tool.calls == [{"count": 3}]


@pytest.mark.asyncio
async def test_replacement_while_admission_is_awaited_cannot_redirect_the_call(host):
    agent, backend = host
    original, replacement = _Tool(), _Tool()
    binding = _mount(agent, original)

    async def replace():
        agent.unregister_dynamic_tools("mcp:views")
        _mount(agent, replacement, owner="mcp:replacement")

    agent.observability_store.on_admitted = replace
    policy = _Policy(HookOutput.allow())
    agent.hooks_manager.register(policy)
    with pytest.raises(ResourceConflict):
        await binding.invoke({"count": 3}, session_id="mcp-app:view-1")
    assert original.calls == replacement.calls == []
    assert policy.inputs[0].feature_name == "mcp:views"
    row = await backend.fetch_one("SELECT adapter, error_class FROM a2a_tool_dispatches")
    assert tuple(row) == ("mcp:views.view_update", "ResourceConflict")


@pytest.mark.asyncio
async def test_hook_modified_arguments_are_validated_before_execution(host):
    agent, backend = host
    tool = _Tool()
    binding = _mount(agent, tool)
    agent.hooks_manager.register(_Policy(HookOutput.modify({"misspelled": 7})))
    result = await binding.invoke({"count": 3}, session_id="mcp-app:view-1")
    assert result["success"] is False
    assert "misspelled" in result["error"]
    assert tool.calls == []
    row = await backend.fetch_one("SELECT args_redacted FROM a2a_tool_dispatches")
    assert json.loads(row[0]) == {"misspelled": 7}


@pytest.mark.asyncio
async def test_business_resource_conflict_is_not_misreported_as_stale_binding(host):
    class ConflictingTool(_Tool):
        async def execute(self, **args):
            raise ResourceConflict("Domain revision changed")

    agent, _ = host
    binding = _mount(agent, ConflictingTool())
    result = await binding.invoke({"count": 3}, session_id="mcp-app:view-1")
    assert result == {"success": False, "error": "Domain revision changed"}
    assert binding.is_current


@pytest.mark.asyncio
async def test_collision_uses_actual_name_and_original_permission_owner(host):
    agent, _ = host
    first, second = _Tool(), _Tool()
    _mount(agent, first)
    binding = _mount(agent, second, owner="mcp:second")
    policy = _Policy(HookOutput.allow())
    agent.hooks_manager.register(policy)
    await binding.invoke({"count": 4}, session_id="mcp-app:view-1")
    assert binding.name == "view_update_2"
    assert first.calls == []
    assert second.calls == [{"count": 4}]
    assert policy.inputs[0].feature_name == "mcp:second"
    assert policy.inputs[0].tool_name == binding.name


@pytest.mark.asyncio
async def test_evicted_mount_is_not_offered_to_callback(host):
    agent, _ = host
    agent.MAX_DIRECT_TOOLS = 1
    agent.register_dynamic_tools("pinned", [_Tool("pinned")], pin=True)
    offered = []
    assert agent.register_dynamic_tools(
        "overflow", [_Tool()], on_registered=lambda *pair: offered.append(pair),
    ) == 0
    assert offered == []


@pytest.mark.asyncio
async def test_lru_eviction_expires_binding(host):
    agent, _ = host
    agent.MAX_DIRECT_TOOLS = 1
    old = _mount(agent, _Tool())
    current = _mount(agent, _Tool("new"), owner="mcp:new")
    assert not old.is_current
    assert current.is_current
    with pytest.raises(ResourceConflict):
        await old.invoke({"count": 3}, session_id="mcp-app:view-1")


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_store", [False, True])
async def test_missing_audit_writer_refuses_before_any_effect(host, monkeypatch, missing_store):
    agent, _ = host
    tool = _Tool()
    binding = _mount(agent, tool)
    if missing_store:
        monkeypatch.setattr(agent, "observability_store", None)
    else:
        monkeypatch.setattr(agent.observability_store, "log_tool_dispatch", None)
    with pytest.raises(ResourceUnsupported, match="audit"):
        await binding.invoke({"count": 3}, session_id="mcp-app:view-1")
    assert tool.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments,session_id,source,error", [
    ([], "mcp-app:view-1", "external", TypeError),
    ({"count": 3}, "", "external", ValueError),
    ({"count": 3}, None, "external", ValueError),
    ({"count": 3}, "mcp-app:view-1", "", ValueError),
    ({"count": 3}, "mcp-app:view-1", None, ValueError),
])
async def test_binding_requires_valid_invocation_context(host, arguments, session_id, source, error):
    agent, _ = host
    tool = _Tool()
    binding = _mount(agent, tool)
    with pytest.raises(error):
        await binding.invoke(arguments, session_id=session_id, source=source)
    assert tool.calls == []


@pytest.mark.asyncio
async def test_invalid_callback_does_not_mutate_registry(host):
    agent, _ = host
    with pytest.raises(TypeError, match="on_registered"):
        agent.register_dynamic_tools("mcp:views", [_Tool()], on_registered=7)
    assert agent._direct_tools == agent._explored_features == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["allow", "deny", "approve", "reject", "replace", "cancel"])
async def test_binding_uses_native_permission_store_and_approval_queue(host, tmp_path, decision):
    agent, backend = host
    tool = _Tool()
    binding = _mount(agent, tool)
    permissions = PermissionStore(str(tmp_path / "permissions.db"))
    await permissions.initialize()
    requests = asyncio.Queue()

    async def added(request):
        await requests.put(request)

    approvals = ApprovalQueue(on_request_added=added, permission_store=permissions, agent=agent)
    agent.hooks_manager.register(SecurityHook(permissions, approvals))
    if decision in {"allow", "deny"}:
        await permissions.set_permission(
            "mcp:views", binding.name,
            PermissionLevel.ALLOW if decision == "allow" else PermissionLevel.DENY,
        )
    call = asyncio.create_task(binding.invoke({"count": 3}, session_id="mcp-app:view-1"))
    try:
        if decision not in {"allow", "deny"}:
            request = await requests.get()
            assert request.feature_name == "mcp:views"
            assert request.tool_name == binding.name
            assert not call.done()
            assert tool.calls == []
            if decision == "replace":
                agent.unregister_dynamic_tools("mcp:views")
                _mount(agent, tool)
            if decision == "cancel":
                call.cancel()
            else:
                await approvals.submit_decision(request.id, decision != "reject", "once")
        if decision == "replace":
            with pytest.raises(ResourceConflict):
                await call
        elif decision == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await call
            assert request.status == ApprovalStatus.AWAITER_GONE
            late = await approvals.submit_decision(request.id, True, "once")
            assert late.awaiter_gone and late.persisted
            assert approvals.pending_count == 0
            row = await backend.fetch_one("SELECT result_status, error_class FROM a2a_tool_dispatches")
            assert tuple(row) == ("error", "CancelledError")
        else:
            result = await call
            assert result["success"] is (decision in {"allow", "approve"})
        expected = [{"count": 3}] if decision in {"allow", "approve"} else []
        assert tool.calls == expected
        assert await permissions.get_audit_log()
        if decision != "cancel":
            rows = await backend.fetch_all("SELECT result_status FROM a2a_tool_dispatches")
            assert len(rows) == 1
        if decision == "approve":
            # A once-only approval cannot authorize a second app request.
            call = asyncio.create_task(binding.invoke({"count": 4}, session_id="mcp-app:view-1"))
            request = await requests.get()
            assert not call.done()
            assert tool.calls == expected
            await approvals.submit_decision(request.id, False, "once")
            assert (await call)["success"] is False
            assert tool.calls == expected
    finally:
        approvals.cancel_all()
        if not call.done():
            call.cancel()
        await asyncio.gather(call, return_exceptions=True)


@pytest.mark.asyncio
async def test_registering_owner_again_expires_prior_batch(host):
    agent, _ = host
    old = _mount(agent, _Tool())
    new = _mount(agent, _Tool("new"))
    assert not old.is_current
    assert new.is_current


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["target", "owner"])
async def test_binding_checks_canonical_registry_not_just_batch_presence(host, changed):
    agent, _ = host
    original, replacement = _Tool(), _Tool()
    binding = _mount(agent, original)
    # A retained owner batch cannot authorize a handle the host has reassigned.
    if changed == "target":
        agent._direct_tools[binding.name] = replacement
    else:
        agent._tool_to_feature[binding.name] = "mcp:another-owner"
    assert not binding.is_current
    with pytest.raises(ResourceConflict):
        await binding.invoke({"count": 3}, session_id="mcp-app:view-1")
    assert original.calls == replacement.calls == []


@pytest.mark.asyncio
async def test_bound_arguments_are_an_independent_snapshot_while_approval_waits(host):
    agent, _ = host
    tool = _Tool()
    binding = _mount(agent, tool)
    entered, resume = asyncio.Event(), asyncio.Event()

    async def pending():
        entered.set()
        await resume.wait()

    agent.hooks_manager.register(_Policy(HookOutput.allow(), before=pending))
    arguments = {"count": {"nested": [3]}}
    call = asyncio.create_task(binding.invoke(arguments, session_id="mcp-app:view-1"))
    try:
        await entered.wait()
        arguments["count"]["nested"][0] = 9
        resume.set()
        await call
        assert tool.calls == [{"count": {"nested": [3]}}]
    finally:
        resume.set()
        if not call.done():
            call.cancel()
        await asyncio.gather(call, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("nested", [False, True])
async def test_oversize_argument_returns_an_explicit_refusal_without_execution(host, nested):
    agent, backend = host
    tool = _Tool()
    binding = _mount(agent, tool)
    value = "x" * (MAX_TOOL_ARG_LENGTH + 1)
    if nested:
        value = {"nested": value}
    result = await binding.invoke({"count": value}, session_id="mcp-app:view-1")
    assert isinstance(result, dict) and result["success"] is False
    assert "validation" in result["error"].lower()
    assert tool.calls == []
    row = await backend.fetch_one("SELECT error_class FROM a2a_tool_dispatches")
    assert row[0] == "ToolValidationError"


@pytest.mark.asyncio
async def test_validation_refusal_retains_original_owner_across_rejection_await(host, monkeypatch):
    agent, backend = host
    tool = _Tool()
    binding = _mount(agent, tool)
    original_reject = orchestrator_engine.record_tool_rejection

    async def replace_after_rejection(*args, **kwargs):
        await original_reject(*args, **kwargs)
        agent.unregister_dynamic_tools("mcp:views")
        _mount(agent, _Tool(), owner="mcp:replacement")

    monkeypatch.setattr(orchestrator_engine, "record_tool_rejection", replace_after_rejection)
    await binding.invoke({"count": "x" * (MAX_TOOL_ARG_LENGTH + 1)}, session_id="mcp-app:view-1")
    assert tool.calls == []
    row = await backend.fetch_one("SELECT adapter FROM a2a_tool_dispatches")
    assert row[0] == "mcp:views.view_update"


@pytest.mark.asyncio
async def test_transient_response_audit_failure_after_effect_is_unknown(host):
    agent, backend = host
    tool = _Tool()
    binding = _mount(agent, tool)
    agent.observability_store.fail_success_response_once = True
    with pytest.raises(ResourceOutcomeUnknown) as captured:
        await binding.invoke({"count": 3}, session_id="mcp-app:view-1")
    assert tool.calls == [{"count": 3}]
    row = await backend.fetch_one("SELECT result_status, error_class FROM a2a_tool_dispatches")
    assert row is not None, repr(captured.value.__cause__)
    assert tuple(row) == ("error", "ResourceOutcomeUnknown")


@pytest.mark.asyncio
async def test_cancelled_tool_attempt_has_durable_unknown_outcome_before_unwinding(host):
    entered = asyncio.Event()

    class InterruptedTool(_Tool):
        async def execute(self, **args):
            self.calls.append(args)
            entered.set()
            await asyncio.Future()

    agent, backend = host
    tool = InterruptedTool()
    binding = _mount(agent, tool)
    call = asyncio.create_task(binding.invoke({"count": 3}, session_id="mcp-app:view-1"))
    try:
        await entered.wait()
        call.cancel()
        with pytest.raises(asyncio.CancelledError) as captured:
            await call
        assert tool.calls == [{"count": 3}]
        row = await backend.fetch_one("SELECT result_status, error_class FROM a2a_tool_dispatches")
        assert row is not None, repr(captured.value.__cause__)
        assert tuple(row) == ("error", "ResourceOutcomeUnknown")
    finally:
        if not call.done():
            call.cancel()
        await asyncio.gather(call, return_exceptions=True)


@pytest.mark.asyncio
async def test_repeated_cancellation_joins_the_same_audit_without_reexecuting(host):
    started, writing, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class InterruptedTool(_Tool):
        async def execute(self, **args):
            self.calls.append(args)
            started.set()
            await asyncio.Future()

    async def wait_for_write():
        writing.set()
        await release.wait()

    agent, backend = host
    tool = InterruptedTool()
    binding = _mount(agent, tool)
    agent.observability_store.on_dispatch = wait_for_write
    call = asyncio.create_task(binding.invoke({"count": 3}, session_id="mcp-app:view-1"))
    try:
        await started.wait()
        call.cancel()
        await writing.wait()
        call.cancel()
        await asyncio.sleep(0)
        call.cancel()
        await asyncio.sleep(0)
        assert not call.done()
        assert tool.calls == [{"count": 3}]
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await call
        rows = await backend.fetch_all("SELECT error_class FROM a2a_tool_dispatches")
        assert [row[0] for row in rows] == ["ResourceOutcomeUnknown"]
    finally:
        release.set()
        if not call.done():
            call.cancel()
        await asyncio.gather(call, return_exceptions=True)


@pytest.mark.asyncio
async def test_post_hooks_and_audit_observe_the_actual_normalized_arguments(host):
    agent, backend = host
    tool = _Tool()
    binding = _mount(agent, tool)
    agent.hooks_manager.register(_Policy(HookOutput.modify({"count": 3, "context": "orchestration"})))
    post = _Policy(HookOutput.allow())
    post.events = [HookEvent.POST_TOOL_USE]
    agent.hooks_manager.register(post)
    await binding.invoke({"count": 1}, session_id="mcp-app:view-1")
    assert tool.calls == [{"count": 3}]
    assert post.inputs[0].tool_input == {"count": 3}
    row = await backend.fetch_one("SELECT args_redacted FROM a2a_tool_dispatches")
    assert json.loads(row[0]) == {"count": 3}


@pytest.mark.asyncio
async def test_hook_cannot_introduce_an_oversize_argument_after_initial_validation(host):
    agent, _ = host
    tool = _Tool()
    binding = _mount(agent, tool)
    agent.hooks_manager.register(_Policy(HookOutput.modify({"count": "x" * (MAX_TOOL_ARG_LENGTH + 1)})))
    result = await binding.invoke({"count": 1}, session_id="mcp-app:view-1")
    assert result["success"] is False
    assert "maximum length" in result["error"]
    assert tool.calls == []


@pytest.mark.asyncio
async def test_tool_mutation_cannot_rewrite_the_record_of_approved_arguments(host):
    class MutatingTool(_Tool):
        async def execute(self, **args):
            self.calls.append(deepcopy(args))
            args["count"]["nested"].append(9)
            return {"success": True}

    agent, backend = host
    tool = MutatingTool()
    tool.schema.parameters[0].type = "object"
    binding = _mount(agent, tool)
    await binding.invoke({"count": {"nested": [3]}}, session_id="mcp-app:view-1")
    assert tool.calls == [{"count": {"nested": [3]}}]
    row = await backend.fetch_one("SELECT args_redacted FROM a2a_tool_dispatches")
    assert json.loads(row[0]) == {"count": {"nested": [3]}}


@pytest.mark.asyncio
async def test_cancelled_audit_failure_preserves_cancellation_and_failed_evidence(host):
    from kestrel_sovereign.storage.db.interface import DatabaseError
    entered = asyncio.Event()

    class InterruptedTool(_Tool):
        async def execute(self, **args):
            self.calls.append(args)
            entered.set()
            await asyncio.Future()

    agent, backend = host
    tool = InterruptedTool()
    binding = _mount(agent, tool)
    call = asyncio.create_task(binding.invoke({"count": 3}, session_id="mcp-app:view-1"))
    try:
        await entered.wait()
        await backend.execute("DROP TABLE a2a_tool_dispatches")
        call.cancel()
        with pytest.raises(asyncio.CancelledError) as captured:
            await call
        assert isinstance(captured.value.__cause__, ResourceOutcomeUnknown)
        assert isinstance(captured.value.__cause__.__cause__, DatabaseError)
        assert tool.calls == [{"count": 3}]
        assert not any(task.get_name().startswith("bound-tool-invocation")
                       for task in asyncio.all_tasks() if not task.done())
    finally:
        if not call.done():
            call.cancel()
        await asyncio.gather(call, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError("no confirmed result"), ResourceOutcomeUnknown("unconfirmed effect")])
async def test_tool_attempt_without_confirmed_result_is_not_a_retryable_failure(host, error):
    class UnconfirmedTool(_Tool):
        async def execute(self, **args):
            self.calls.append(args)
            raise error

    agent, backend = host
    tool = UnconfirmedTool()
    binding = _mount(agent, tool)
    with pytest.raises(ResourceOutcomeUnknown) as captured:
        await binding.invoke({"count": 3}, session_id="mcp-app:view-1")
    assert captured.value.__cause__ is error
    assert tool.calls == [{"count": 3}]
    row = await backend.fetch_one("SELECT result_status, error_class FROM a2a_tool_dispatches")
    assert tuple(row) == ("error", "ResourceOutcomeUnknown")


@pytest.mark.asyncio
async def test_tool_control_exception_group_preserves_original_leaves_and_audit(host):
    interrupted = asyncio.CancelledError("transport interrupted")
    failed = ValueError("transport teardown failed")
    original = BaseExceptionGroup("tool stopped", [interrupted, failed])

    class InterruptedTool(_Tool):
        async def execute(self, **args):
            self.calls.append(args)
            raise original

    agent, backend = host
    tool = InterruptedTool()
    binding = _mount(agent, tool)
    with pytest.raises(BaseExceptionGroup) as captured:
        await binding.invoke({"count": 3}, session_id="mcp-app:view-1")
    assert captured.value is original
    assert captured.value.exceptions == (interrupted, failed)
    assert tool.calls == [{"count": 3}]
    row = await backend.fetch_one("SELECT result_status, error_class FROM a2a_tool_dispatches")
    assert tuple(row) == ("error", "ResourceOutcomeUnknown")


@pytest.mark.asyncio
async def test_unserializable_result_after_effect_is_unknown_and_audited_once(host):
    class InvalidResultTool(_Tool):
        async def execute(self, **args):
            self.calls.append(args)
            return {"success": True, ("invalid-json-key",): 1}

    agent, backend = host
    tool = InvalidResultTool()
    binding = _mount(agent, tool)
    with pytest.raises(ResourceOutcomeUnknown) as captured:
        await binding.invoke({"count": 3}, session_id="mcp-app:view-1")
    assert isinstance(captured.value.__cause__, TypeError)
    assert tool.calls == [{"count": 3}]
    rows = await backend.fetch_all("SELECT result_status, error_class FROM a2a_tool_dispatches")
    assert [tuple(row) for row in rows] == [("error", "ResourceOutcomeUnknown")]


@pytest.mark.asyncio
async def test_pre_execution_authority_failure_records_its_actual_class(host):
    from kestrel_sovereign.execution_custody import ExecutionAuthorityError
    error = ExecutionAuthorityError("original admission expired")

    async def expire():
        raise error

    agent, backend = host
    tool = _Tool()
    binding = _mount(agent, tool)
    agent.hooks_manager.register(_Policy(HookOutput.allow(), before=expire))
    with pytest.raises(ExecutionAuthorityError) as captured:
        await binding.invoke({"count": 3}, session_id="mcp-app:view-1")
    assert captured.value is error
    assert tool.calls == []
    row = await backend.fetch_one("SELECT result_status, error_class FROM a2a_tool_dispatches")
    assert tuple(row) == ("error", "ExecutionAuthorityError")
