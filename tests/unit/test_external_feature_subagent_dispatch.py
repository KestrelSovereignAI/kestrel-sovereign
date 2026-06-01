"""External (SDK-base) features must be dispatchable like in-tree features.

Regression coverage for the bug where an installed external feature
(e.g. kestrel-feature-github) loaded and registered but was silently skipped by
the orchestrator because the runtime-coupled subagent-dispatch methods
(``execute_as_subagent`` et al.) live on the sovereign Feature base, not the
lean ``kestrel_sdk`` base that external packages subclass.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kestrel_sdk.features.base import Feature as SdkFeature, tool as sdk_tool
from kestrel_sdk.tools.base import ToolCategory

from kestrel_sovereign.features.base import Feature as SovereignFeature
from kestrel_sovereign.features.subagent_dispatch import (
    _DISPATCH_METHODS,
    ensure_subagent_dispatch,
)


class _ExternalFeature(SdkFeature):
    """Minimal feature built on the public SDK base, like a pip-installed one."""

    tool_name = "external_demo"

    @property
    def tool_description(self) -> str:
        return "A demo external feature"

    async def initialize(self) -> None:
        return None

    @sdk_tool(
        name="ping",
        description="Return pong",
        category=ToolCategory.DATA_ACCESS,
    )
    async def ping(self) -> str:
        return "pong"


def _orchestrator_gate(feature) -> bool:
    """Mirror agent.tool_registry._feature_supports_subagent_dispatch."""
    return callable(getattr(feature, "to_orchestrator_tool", None)) and callable(
        getattr(feature, "execute_as_subagent", None)
    )


def test_sdk_base_feature_lacks_dispatch_by_default():
    """Establish the bug precondition: the SDK base has no execute_as_subagent."""
    plain = _ExternalFeature(agent=None)
    assert not _orchestrator_gate(plain)


def test_injection_makes_external_feature_dispatchable():
    cls = ensure_subagent_dispatch(_ExternalFeature)
    feature = cls(agent=None)

    assert cls is not _ExternalFeature
    assert cls.__name__ == "_ExternalFeature"  # name preserved for logging/payloads
    # __module__ must be preserved: the ToolResult-contract enforcer keys its
    # migrated-module allowlist off type(feature).__module__. A synthesized
    # class reporting "abc" would silently skip enforcement for migrated
    # external features.
    assert cls.__module__ == _ExternalFeature.__module__
    assert cls.__qualname__ == _ExternalFeature.__qualname__
    assert _orchestrator_gate(feature)
    assert callable(feature.handle_task)
    # The feature's own tool is still intact and reachable.
    assert feature.to_orchestrator_tool()["function"]["name"] == "external_demo"
    assert {t.name for t in feature.get_tools()} == {"ping"}


def test_intree_feature_returned_unchanged():
    """Features already on the sovereign base must not be re-wrapped."""
    from kestrel_sovereign.features.cli.feature import CliFeature

    assert issubclass(CliFeature, SovereignFeature)
    assert ensure_subagent_dispatch(CliFeature) is CliFeature


def test_injection_does_not_clobber_feature_supplied_dispatch():
    sentinel = object()

    class _CustomDispatch(SdkFeature):
        tool_name = "custom_dispatch"

        async def initialize(self) -> None:
            return None

        async def execute_as_subagent(self, *a, **k):
            return sentinel

    cls = ensure_subagent_dispatch(_CustomDispatch)
    # It still needs handle_task etc., so it is wrapped...
    assert cls.execute_as_subagent is _CustomDispatch.execute_as_subagent


def test_dispatch_method_closure_is_present_on_sovereign_base():
    """Guard against drift: every injected name must exist on the sovereign base.

    If someone adds a new ``self._helper(...)`` call inside the dispatch loop
    without adding it here, the end-to-end test below breaks at runtime — this
    keeps the list honest at the type level too.
    """
    for name in _DISPATCH_METHODS:
        assert callable(getattr(SovereignFeature, name, None)), name


@pytest.mark.asyncio
async def test_external_feature_executes_as_subagent_end_to_end():
    """Exercise the borrowed cluster end-to-end on an external feature."""
    cls = ensure_subagent_dispatch(_ExternalFeature)
    fake_agent = SimpleNamespace(
        llm_service=SimpleNamespace(generate=AsyncMock(return_value="all done")),
        hooks_manager=None,
    )
    feature = cls(agent=fake_agent)

    result = await feature.execute_as_subagent(task="say something")

    assert result["success"] is True
    assert result["result"] == "all done"
    fake_agent.llm_service.generate.assert_awaited_once()
