"""External tool bindings use the real dispatcher, hooks, and durable audit."""

import asyncio
import json

import pytest
import pytest_asyncio
from kestrel_sdk.hooks.base import Hook, HookEvent, HookOutput
from kestrel_sdk.resources import ResourceConflict, ResourceOutcomeUnknown
from kestrel_sdk.tools.base import ToolCategory, ToolParameter, ToolSchema

from kestrel_sovereign.a2a.stores.unified.observability_store import ObservabilityStore
from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
from kestrel_sovereign.agent.tool_registry import ToolRegistryMixin
from kestrel_sovereign.hooks.manager import HooksManager
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend


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


@pytest_asyncio.fixture
async def host(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "app-dispatches.db"))
    await backend.connect()
    store = ObservabilityStore(backend)
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
