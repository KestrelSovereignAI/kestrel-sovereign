"""One tool call has one recorded outcome in both audit tables (#3924).

``a2a_observability.success`` and ``a2a_tool_dispatches.result_status`` are
written by different calls. A tool that returned a failed envelope used to be
"successful" in the first and "error" in the second. Both are now written from
the one status ``infer_tool_result_status`` computes for the call, on the
direct, bound and feature-dispatch paths.
"""

import pytest
import pytest_asyncio
from kestrel_sdk.hooks.base import Hook, HookEvent, HookOutput
from kestrel_sdk.resources import ResourceOutcomeUnknown
from kestrel_sdk.tools import DeferredToolResult
from kestrel_sdk.tools.base import ToolCategory, ToolParameter, ToolSchema
from kestrel_sdk.tools.result import ToolResult

from kestrel_sovereign.a2a.stores.unified.observability_store import (
    ObservabilityStore,
    infer_tool_result_status,
)
from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
from kestrel_sovereign.agent.tool_registry import ToolRegistryMixin
from kestrel_sovereign.hooks.manager import HooksManager
from kestrel_sovereign.llm.adapter import ToolCall
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend

DENIED = "denied"
RAISES = "raises"

# Each case: what the tool returns, and the status both tables must record.
OUTCOMES = {
    "ok_tool_result": (lambda: ToolResult.ok("Saved"), "success"),
    "ok_envelope": (lambda: ToolResult.ok("Saved").to_dict(), "success"),
    "failed_tool_result": (lambda: ToolResult.failed("store unavailable"), "error"),
    "failed_envelope": (
        lambda: {**ToolResult.failed("store unavailable").to_dict(), "success": False},
        "error",
    ),
    "failed_dict": (lambda: {"success": False, "error": "store unavailable"}, "error"),
    "partial_tool_result": (
        lambda: ToolResult.partial("Saved", "index degraded"), "error",
    ),
    # The in-tree wrapper's PARTIAL shape keeps ``success: True``.
    "partial_envelope": (
        lambda: {**ToolResult.partial("Saved", "index degraded").to_dict(), "success": True},
        "error",
    ),
    "status_error_dict": (lambda: {"status": "error", "detail": "no route"}, "error"),
    "permission_denied": (DENIED, "policy_denied"),
}


class _Tool:
    """A direct tool whose ``execute`` returns (or raises) a fixed outcome."""

    def __init__(self, outcome, name="record_fact"):
        self.name = name
        self.schema = ToolSchema(
            name, "Record a fact", ToolCategory.UTILITY,
            parameters=[ToolParameter("count", "integer", "Count", required=False)],
        )
        self.outcome = outcome
        self.calls = 0

    async def execute(self, **_args):
        self.calls += 1
        if self.outcome is RAISES:
            raise ValueError("broken on purpose")
        return self.outcome()


class _Feature:
    """A feature dispatched as a subagent, returning a fixed outcome."""

    tool_name = "fact_feature"

    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = 0

    async def execute_as_subagent(self, *, task, context, denied_tools, model_override):
        self.calls += 1
        if self.outcome is RAISES:
            raise ValueError("broken on purpose")
        return self.outcome()


class _Deny(Hook):
    def __init__(self):
        super().__init__("deny-policy", [HookEvent.PRE_TOOL_USE], awaits_user_input=True)

    async def execute(self, value):
        return HookOutput.deny("host denied")


class _Agent(OrchestratorEngineMixin, ToolRegistryMixin):
    def __init__(self, store):
        self.did = "did:test:outcome-audit"
        self.features = {}
        self._direct_tools, self._tool_to_feature, self._explored_features = {}, {}, {}
        self._direct_tool_defs, self._pinned_features = [], set()
        self.hooks_manager, self.observability_store = HooksManager(), store

    def _register_explored_feature_tools(self, feature):
        pass


@pytest_asyncio.fixture
async def host(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "outcome-audit.db"))
    await backend.connect()
    store = ObservabilityStore(backend)
    await store.initialize()
    try:
        yield _Agent(store), backend
    finally:
        await store.close()


async def _recorded(backend):
    observed = await backend.fetch_all(
        "SELECT success FROM a2a_observability WHERE event_type = 'tool_response'"
    )
    dispatched = await backend.fetch_all("SELECT result_status FROM a2a_tool_dispatches")
    return [bool(row[0]) for row in observed], [row[0] for row in dispatched]


async def _assert_one_outcome(backend, expected):
    observed, dispatched = await _recorded(backend)
    assert dispatched == [expected]
    assert observed == [expected == "success"]


async def _dispatch(agent, name, features_by_tool_name=None):
    # A feature subagent takes a task; a direct tool only its own parameters.
    arguments = {"task": "record it"} if features_by_tool_name else {"count": 3}
    return await agent._dispatch_tool_call(
        ToolCall(id="tc-1", name=name, arguments=arguments),
        features_by_tool_name=features_by_tool_name or {},
        known_tools={name},
        messages=[],
        iteration=0,
        user_message="record it",
        session_id="sess-outcome",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("case", list(OUTCOMES))
async def test_direct_tool_records_one_outcome(host, case):
    agent, backend = host
    produce, expected = OUTCOMES[case]
    tool = _Tool(None if produce is DENIED else produce)
    agent._direct_tools[tool.name] = tool
    agent._tool_to_feature[tool.name] = "FactFeature"
    if produce is DENIED:
        agent.hooks_manager.register(_Deny())

    await _dispatch(agent, tool.name)

    assert tool.calls == (0 if produce is DENIED else 1)
    await _assert_one_outcome(backend, expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", list(OUTCOMES))
async def test_bound_tool_records_one_outcome(host, case):
    agent, backend = host
    produce, expected = OUTCOMES[case]
    tool = _Tool(None if produce is DENIED else produce, name="view.update")
    bindings = []
    agent.register_dynamic_tools(
        "mcp:views", [tool], on_registered=lambda _tool, binding: bindings.append(binding),
    )
    if produce is DENIED:
        agent.hooks_manager.register(_Deny())

    await bindings[0].invoke({"count": 3}, session_id="mcp-app:view-1", source="mcp-app")

    assert tool.calls == (0 if produce is DENIED else 1)
    await _assert_one_outcome(backend, expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", list(OUTCOMES))
async def test_feature_dispatch_records_one_outcome(host, case):
    agent, backend = host
    produce, expected = OUTCOMES[case]
    feature = _Feature(None if produce is DENIED else produce)
    if produce is DENIED:
        agent.hooks_manager.register(_Deny())

    await _dispatch(agent, feature.tool_name, {feature.tool_name: feature})

    assert feature.calls == (0 if produce is DENIED else 1)
    await _assert_one_outcome(backend, expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["ok_envelope", "failed_envelope", "partial_envelope"])
async def test_deferred_result_records_one_outcome(host, case):
    """A deferred result's dispatch row is written before it returns."""
    agent, backend = host
    produce, expected = OUTCOMES[case]
    envelope = produce()
    tool = _Tool(lambda: DeferredToolResult(envelope, publish=lambda: []))
    agent._direct_tools[tool.name] = tool
    agent._tool_to_feature[tool.name] = "FactFeature"

    await _dispatch(agent, tool.name)

    await _assert_one_outcome(backend, expected)


@pytest.mark.asyncio
async def test_raising_direct_tool_still_records_failure(host):
    agent, backend = host
    tool = _Tool(RAISES)
    agent._direct_tools[tool.name] = tool
    agent._tool_to_feature[tool.name] = "FactFeature"

    result = await _dispatch(agent, tool.name)

    assert result == {"success": False, "error": "broken on purpose"}
    await _assert_one_outcome(backend, "error")


@pytest.mark.asyncio
async def test_raising_feature_still_records_failure(host):
    agent, backend = host
    feature = _Feature(RAISES)

    result = await _dispatch(agent, feature.tool_name, {feature.tool_name: feature})

    assert result == {"success": False, "error": "broken on purpose"}
    await _assert_one_outcome(backend, "error")


@pytest.mark.asyncio
async def test_raising_bound_tool_still_records_failure(host):
    agent, backend = host
    tool = _Tool(RAISES, name="view.update")
    bindings = []
    agent.register_dynamic_tools(
        "mcp:views", [tool], on_registered=lambda _tool, binding: bindings.append(binding),
    )

    with pytest.raises(ResourceOutcomeUnknown):
        await bindings[0].invoke({"count": 3}, session_id="mcp-app:view-1", source="mcp-app")

    await _assert_one_outcome(backend, "error")


def test_tool_result_status_reads_the_envelope_not_its_constructors():
    # ``ToolResult.failed`` is a classmethod: it must not mark every result.
    assert infer_tool_result_status(ToolResult.ok("Saved")) == "success"
    assert infer_tool_result_status(ToolResult.failed("no")) == "error"
    assert infer_tool_result_status(ToolResult.partial("Saved", "degraded")) == "error"


def test_failed_flag_attribute_still_marks_an_error():
    class _Outcome:
        failed = True

    class _Passed:
        failed = False

    assert infer_tool_result_status(_Outcome()) == "error"
    assert infer_tool_result_status(_Passed()) == "success"
