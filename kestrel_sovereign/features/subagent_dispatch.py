"""Make the A2A subagent-dispatch pattern universal across all features.

The orchestrator drives every feature as a subagent: the LLM calls the feature
as one high-level tool, the feature runs its own LLM turn over its own ``@tool``
methods, and (after exploration) those methods are promoted to direct tools.
That machinery — ``execute_as_subagent`` and the loop it drives — is
*runtime-coupled*: it calls ``self.agent.llm_service``, enforces ``PRE_TOOL_USE``
hooks, and builds the codex inline tool-executor. So it lives on the sovereign
``kestrel_sovereign.features.base.Feature`` base, **not** on the lean,
runtime-agnostic ``kestrel_sdk.features.base.Feature`` base that external feature
packages subclass.

The consequence: an installed external feature (e.g. ``kestrel-feature-github``)
loads and registers, but the orchestrator's dispatch gate
(``_feature_supports_subagent_dispatch`` requires ``execute_as_subagent``) skips
it, so the LLM is never offered the feature's tool. It is present in the catalog
but uncallable.

This module closes that gap *without* bloating the SDK or duplicating the
loop: at feature discovery, external feature classes get the sovereign dispatch
methods mixed in. In-tree features (which already subclass the sovereign base)
are returned unchanged. Borrowed methods keep ``base.py`` as their
``__globals__``, so the module-level A2A imports they rely on resolve correctly
regardless of which class they are attached to.

The same subclass guards the feature's ``@tool`` methods. The SDK's @tool
wrapper returns every non-control error as an ordinary failed result, so an A2A
task could not tell that a skill raising ``ResourceOutcomeUnknown`` may already
have acted, and a workflow would run it again (#3946). The guard reports every
error the method raises to the running ``handle_task``, which keeps one
carrying an unknown outcome (itself, a cause or an exception-group member), and
re-raises it unchanged, so the wrapper's result, and everything a model sees,
stay the same.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Type

# The complete transitive closure of methods the subagent-dispatch loop needs
# that the lean SDK Feature base does NOT provide. (``to_orchestrator_tool``,
# ``get_agent_card``, and ``get_skill_for_command`` already exist on the SDK
# base, so external features have them.) Keep this in sync with the dispatch
# cluster in ``kestrel_sovereign/features/base.py``; the test
# ``test_subagent_dispatch_closure_is_complete`` guards it.
_DISPATCH_METHODS: tuple[str, ...] = (
    "execute_as_subagent",
    "handle_task",
    "_handle_feature_tool_calls",
    "_make_feature_inline_tool_executor",
    "_execute_subagent_tool",
    "_get_subagent_prompt",
    "_compose_subagent_runtime_tools",
    "_subagent_context_budget",
    "_subagent_context_overflow",
    "_subagent_wire_chars",
    "_turn_session_id",
    "_repair_subagent_premature_yield",
    "_get_tool_by_name",
    "_build_subagent_assistant_tool_history_msg",
    "_extract_response_reasoning_content",
    "_signals_unfinished_tool_work",
    "_append_missing_tool_call_repair",
)


def _outcome_reporting_tool(
    method: Callable[..., Any], report: Callable[[BaseException], None]
) -> Callable[..., Any]:
    """Wrap one @tool method so its unknown outcome reaches ``handle_task``.

    The wrapper keeps the method's name, signature, ``_tool_schema`` and kind
    (a coroutine function stays one, a plain function stays plain), so the
    SDK's ``get_tools`` advertises and coerces exactly as before and a direct
    caller gets what it always got. ``report`` keeps only an error that
    carries an unknown outcome, whether bare, chained or grouped (a method
    running its effects in an ``asyncio.TaskGroup`` raises an
    ``ExceptionGroup``). An error the SDK wrapper re-raises as control instead
    never yields a task to mark: ``handle_task`` propagates it.
    """

    async def reported(awaitable: Any) -> Any:
        try:
            return await awaitable
        except Exception as error:
            report(error)
            raise

    if inspect.iscoroutinefunction(method):

        @functools.wraps(method)
        async def reporting_coroutine(self: Any, *args: Any, **kwargs: Any) -> Any:
            return await reported(method(self, *args, **kwargs))

        return reporting_coroutine

    # The SDK wrapper awaits whatever the method returns, so a plain method
    # can still fail later, from the awaitable it returned.
    @functools.wraps(method)
    def reporting(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            result = method(self, *args, **kwargs)
        except Exception as error:
            report(error)
            raise
        return reported(result) if inspect.isawaitable(result) else result

    return reporting


def _outcome_reporting_tools(
    feature_class: Type, report: Callable[[BaseException], None]
) -> dict[str, Callable[..., Any]]:
    """The class's resolved @tool methods, each wrapped to report its outcome."""
    wrapped: dict[str, Callable[..., Any]] = {}
    resolved: set[str] = set()
    for klass in feature_class.__mro__:
        for name, value in vars(klass).items():
            if name in resolved:
                continue
            # The nearest definition is the one the instance resolves to; an
            # override without the decorator hides a decorated base method.
            resolved.add(name)
            if inspect.isfunction(value) and hasattr(value, "_tool_schema"):
                wrapped[name] = _outcome_reporting_tool(value, report)
    return wrapped


def ensure_subagent_dispatch(feature_class: Type) -> Type:
    """Return a feature class guaranteed to carry the subagent-dispatch methods.

    In-tree features subclass the sovereign ``Feature`` (which defines the whole
    cluster) and are returned unchanged. External features built on the lean
    ``kestrel_sdk`` Feature base are returned as a dynamically-created subclass
    that mixes the missing sovereign dispatch methods in. The subclass keeps the
    original ``__name__`` so ``type(self).__name__`` (used in tool logging and
    approval payloads) stays accurate. It also overrides each ``@tool`` method
    with one that reports a ``ResourceOutcomeUnknown`` to the sovereign
    ``handle_task`` before the SDK wrapper flattens it (#3946); in-tree features
    report from their own wrapper.

    Injection never overrides a method the feature already defines, so a feature
    is free to supply its own ``execute_as_subagent`` / ``handle_task``.
    """
    from kestrel_sovereign.features.base import Feature as _SovereignFeature
    from kestrel_sovereign.features.base import report_flattened_outcome_unknown

    if issubclass(feature_class, _SovereignFeature):
        return feature_class

    mixin_ns: dict[str, object] = {}
    for name in _DISPATCH_METHODS:
        if getattr(feature_class, name, None) is not None:
            # The feature (or its SDK base) already provides this — don't clobber.
            continue
        # Resolve via __dict__ along the MRO so staticmethod/classmethod
        # descriptors are preserved (two of these are @staticmethod).
        for klass in _SovereignFeature.__mro__:
            if name in klass.__dict__:
                mixin_ns[name] = klass.__dict__[name]
                break

    if not mixin_ns:
        return feature_class

    mixin = type("SubagentDispatchMixin", (), mixin_ns)
    # Carry over the original class identity. ``__module__`` in particular is
    # load-bearing: the ToolResult-contract enforcer keys its migrated-module
    # allowlist off ``type(feature).__module__`` (and ``type(self).__name__``
    # appears in tool logging / approval payloads). Without this the synthesized
    # class reports ``abc`` and contract enforcement is silently skipped for
    # migrated external features.
    return type(
        feature_class.__name__,
        (mixin, feature_class),
        {
            **_outcome_reporting_tools(
                feature_class, report_flattened_outcome_unknown
            ),
            "__module__": feature_class.__module__,
            "__qualname__": feature_class.__qualname__,
            "__doc__": feature_class.__doc__,
        },
    )
