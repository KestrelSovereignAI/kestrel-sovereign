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
"""

from __future__ import annotations

from typing import Type

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
    "_repair_subagent_premature_yield",
    "_get_tool_by_name",
    "_build_subagent_assistant_tool_history_msg",
    "_extract_response_reasoning_content",
    "_signals_unfinished_tool_work",
    "_append_missing_tool_call_repair",
)


def ensure_subagent_dispatch(feature_class: Type) -> Type:
    """Return a feature class guaranteed to carry the subagent-dispatch methods.

    In-tree features subclass the sovereign ``Feature`` (which defines the whole
    cluster) and are returned unchanged. External features built on the lean
    ``kestrel_sdk`` Feature base are returned as a dynamically-created subclass
    that mixes the missing sovereign dispatch methods in. The subclass keeps the
    original ``__name__`` so ``type(self).__name__`` (used in tool logging and
    approval payloads) stays accurate.

    Injection never overrides a method the feature already defines, so a feature
    is free to supply its own ``execute_as_subagent`` / ``handle_task``.
    """
    from kestrel_sovereign.features.base import Feature as _SovereignFeature

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
            "__module__": feature_class.__module__,
            "__qualname__": feature_class.__qualname__,
            "__doc__": feature_class.__doc__,
        },
    )
