"""External (SDK-base) features must be dispatchable like in-tree features.

Regression coverage for the bug where an installed external feature
(e.g. kestrel-feature-github) loaded and registered but was silently skipped by
the orchestrator because the runtime-coupled subagent-dispatch methods
(``execute_as_subagent`` et al.) live on the sovereign Feature base, not the
lean ``kestrel_sdk`` base that external packages subclass.
"""

from __future__ import annotations

import ast
import inspect
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


def test_subagent_dispatch_closure_is_complete():
    """The other direction: every ``self.x(...)`` the cluster makes is borrowable.

    ``test_dispatch_method_closure_is_present_on_sovereign_base`` only checks
    that listed names exist; it cannot notice a name that was never listed. That
    is the drift that actually happens — #2940 added a ``self._turn_session_id()``
    call inside ``execute_as_subagent`` and the omission surfaced only as an
    ``AttributeError`` swallowed into ``success: False`` at runtime. Walk the
    listed methods' source transitively and require every helper they call on
    ``self`` to be provided by the SDK base (external features already have it)
    or to be borrowed too.
    """
    pending = list(_DISPATCH_METHODS)
    checked: set[str] = set()
    unborrowable: dict[str, str] = {}
    while pending:
        name = pending.pop()
        if name in checked:
            continue
        checked.add(name)
        static = inspect.getattr_static(SovereignFeature, name, None)
        func = (
            static.__func__
            if isinstance(static, (staticmethod, classmethod))
            else getattr(SovereignFeature, name, None)
        )
        if func is None:
            continue
        for called in _self_call_names(func):
            if getattr(SdkFeature, called, None) is not None:
                continue  # the lean SDK base already provides it
            if called in _DISPATCH_METHODS:
                pending.append(called)
            elif getattr(SovereignFeature, called, None) is not None:
                unborrowable[called] = name

    assert not unborrowable, (
        "sovereign-only helpers called by the dispatch cluster but missing from "
        f"_DISPATCH_METHODS: {unborrowable}"
    )


def _self_call_names(func) -> set[str]:
    """Names referenced as ``self.<name>`` in ``func``'s own source.

    Calls *and* bare attribute reads. #3245 added a class attribute the
    cluster read as ``self._SUBAGENT_CONTEXT_FRACTION`` and the walk below,
    which then saw only ``self.<name>(...)`` calls, let it through; every
    external feature's subagent died on its first budget check (#3298). A
    borrowed method may only touch what the SDK base provides or what is
    borrowed with it, however it touches it.

    Parsed under a synthetic block header rather than dedented: these methods
    embed prompt literals whose continuation lines sit at a shallower indent
    than the ``def``, so ``textwrap.dedent`` finds no common prefix and slicing
    a fixed width off every line would eat real characters out of the literal.
    Indentation inside a string literal is just content to the tokenizer.
    """
    tree = ast.parse("if True:\n" + inspect.getsource(func))
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    }


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


@pytest.mark.asyncio
async def test_external_feature_legacy_no_argument_prompt_override_dispatches():
    """Published visual features still implement the old prompt override.

    Core must not fail the whole subagent before its tool can run merely
    because the injected dispatch loop gained a runtime-toolset argument.
    """

    class LegacyVisualPrompt(_ExternalFeature):
        def _get_subagent_prompt(self) -> str:
            return "legacy visual identity prompt"

    cls = ensure_subagent_dispatch(LegacyVisualPrompt)
    fake_agent = SimpleNamespace(
        llm_service=SimpleNamespace(generate=AsyncMock(return_value="all done")),
        hooks_manager=None,
    )
    feature = cls(agent=fake_agent)

    result = await feature.execute_as_subagent(task="make a selfie")

    assert result["success"] is True, result
    assert result["result"] == "all done"
    assert fake_agent.llm_service.generate.await_args.kwargs["system_prompt"] == (
        "legacy visual identity prompt"
    )


@pytest.mark.asyncio
async def test_legacy_prompt_does_not_advertise_a_policy_denied_tool():
    """A hard-coded old prompt must not contradict the executable palette."""

    class LegacyVisualPrompt(_ExternalFeature):
        @property
        def tool_description(self) -> str:
            return "Generate selfies and avatars"

        @sdk_tool(
            name="generate_selfie",
            description="Generate a selfie",
            category=ToolCategory.DATA_ACCESS,
        )
        async def generate_selfie(self) -> str:
            return "selfie"

        @sdk_tool(
            name="generate_avatar",
            description="Generate an avatar",
            category=ToolCategory.DATA_ACCESS,
        )
        async def generate_avatar(self) -> str:
            return "avatar"

        def _get_subagent_prompt(self) -> str:
            return "Available tools: generate_selfie, generate_avatar"

    cls = ensure_subagent_dispatch(LegacyVisualPrompt)
    fake_agent = SimpleNamespace(
        llm_service=SimpleNamespace(generate=AsyncMock(return_value="all done")),
        hooks_manager=None,
    )
    feature = cls(agent=fake_agent)

    result = await feature.execute_as_subagent(
        task="make an avatar", denied_tools={"ping", "generate_selfie"}
    )

    assert result["success"] is True, result
    prompt = fake_agent.llm_service.generate.await_args.kwargs["system_prompt"]
    assert "Available tools: generate_avatar" in prompt
    assert "ping" not in prompt
    assert "selfie" not in prompt.lower()
    assert "Markdown image" in prompt


@pytest.mark.asyncio
async def test_filtered_current_prompt_overrides_cannot_advertise_denied_tools():
    """Current positional and keyword-only overrides also yield to policy."""
    called = []

    class PositionalPrompt(_ExternalFeature):
        @property
        def tool_description(self) -> str:
            return "Use ping and generate_avatar"

        @sdk_tool(
            name="generate_avatar",
            description="Generate an avatar",
            category=ToolCategory.DATA_ACCESS,
        )
        async def generate_avatar(self) -> str:
            return "avatar"

        def _get_subagent_prompt(self, runtime_tools) -> str:
            called.append("positional")
            return "Use ping and generate_avatar"

    class KeywordPrompt(PositionalPrompt):
        def _get_subagent_prompt(self, *, runtime_tools) -> str:
            called.append("keyword")
            return "Use ping and generate_avatar"

    for prompt_class in (PositionalPrompt, KeywordPrompt):
        cls = ensure_subagent_dispatch(prompt_class)
        fake_agent = SimpleNamespace(
            llm_service=SimpleNamespace(generate=AsyncMock(return_value="all done")),
            hooks_manager=None,
        )
        result = await cls(agent=fake_agent).execute_as_subagent(
            task="make an avatar", denied_tools={"ping"}
        )
        assert result["success"] is True, result
        prompt = fake_agent.llm_service.generate.await_args.kwargs["system_prompt"]
        assert "Available tools: generate_avatar" in prompt
        assert "ping" not in prompt
        assert "Markdown image" in prompt

    assert called == []


@pytest.mark.asyncio
async def test_external_feature_keyword_only_runtime_prompt_receives_toolset():
    seen = []

    class KeywordPrompt(_ExternalFeature):
        def _get_subagent_prompt(self, *, runtime_tools) -> str:
            seen.append([tool.name for tool in runtime_tools])
            return "keyword-only prompt"

    cls = ensure_subagent_dispatch(KeywordPrompt)
    fake_agent = SimpleNamespace(
        llm_service=SimpleNamespace(generate=AsyncMock(return_value="all done")),
        hooks_manager=None,
    )

    result = await cls(agent=fake_agent).execute_as_subagent(task="ping")

    assert result["success"] is True, result
    assert seen == [["ping"]]
    assert fake_agent.llm_service.generate.await_args.kwargs["system_prompt"] == (
        "keyword-only prompt"
    )


@pytest.mark.asyncio
async def test_prompt_builder_internal_type_error_is_not_signature_fallback():
    class BrokenPrompt(_ExternalFeature):
        def _get_subagent_prompt(self, runtime_tools=None) -> str:
            raise TypeError("internal prompt builder defect")

    cls = ensure_subagent_dispatch(BrokenPrompt)
    fake_agent = SimpleNamespace(
        llm_service=SimpleNamespace(generate=AsyncMock(return_value="not reached")),
        hooks_manager=None,
    )

    result = await cls(agent=fake_agent).execute_as_subagent(task="ping")

    assert result["success"] is False
    assert "internal prompt builder defect" in result["error"]
    fake_agent.llm_service.generate.assert_not_awaited()


def test_external_feature_gets_a_subagent_context_budget():
    """The exact door #3298 opened: a borrowed budget check on an SDK feature.

    ``_subagent_context_budget`` is borrowed onto external features; anything
    it reads through ``self`` must come with it. A model whose context limit
    resolves yields an integer budget rather than an ``AttributeError``
    swallowed into ``success: False`` at runtime.
    """
    from unittest.mock import patch

    from kestrel_sovereign.features.base import SUBAGENT_CONTEXT_FRACTION

    cls = ensure_subagent_dispatch(_ExternalFeature)
    feature = cls.__new__(cls)
    feature.name = "external"
    counter = SimpleNamespace(resolved_context_limit=lambda: 200_000)

    # The limit is resolved through the token counter; pin it so the test
    # measures the borrowed budget arithmetic, not model discovery.
    with patch(
        "kestrel_sovereign.agent.token_counter.get_token_counter",
        return_value=counter,
    ):
        budget = feature._subagent_context_budget("any-model")

    assert budget == int(200_000 * SUBAGENT_CONTEXT_FRACTION)
    assert 0 < SUBAGENT_CONTEXT_FRACTION < 1
