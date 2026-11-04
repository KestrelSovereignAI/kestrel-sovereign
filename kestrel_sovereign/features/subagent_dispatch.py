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

An adopted feature keeps the SDK's own ``get_tools()`` wrapper, whose log
records carry a tool's exception message and part of a rejected argument.
Adoption also installs :class:`SdkToolLogSummary`, which gives those records
the content-free shape Core's own wrapper logs (#3919).
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Type

from kestrel_sdk.features import base as _sdk_feature_base

from kestrel_sovereign.logging_config import (
    content_log_summary,
    exception_frames,
    exception_log_summary,
)

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

    Adopting an external feature also installs :class:`SdkToolLogSummary`,
    because its tools still run through the SDK's ``get_tools()`` wrapper.
    """
    from kestrel_sovereign.features.base import Feature as _SovereignFeature

    if issubclass(feature_class, _SovereignFeature):
        return feature_class

    summarize_sdk_tool_logs()

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


# The SDK wrapper is ``DynamicTool.execute`` inside
# ``kestrel_sdk.features.base.Feature.get_tools``; it logs through that
# module's logger. The two record shapes it writes (SDK 0.46):
#   "Error executing tool {name}: {error}"                (f-string, no args)
#   "Rejected tool call — %s" % "{name}: parameter '{p}' expects type '{t}',
#       but received {repr(value)[:120]} ({type}), which is not a valid {t}"
_SDK_TOOL_WRAPPER_FUNCTION = "execute"
_SDK_TOOL_ERROR = re.compile(r"Error executing tool (?P<tool>[\w.\-]+): ")
_SDK_REJECTED_CALL = "Rejected tool call — %s"
_SDK_REJECTED_ARGUMENT = re.compile(
    r"(?P<head>[\w.\-]+: parameter '\w+' expects type '\w+'), "
    r"but received (?P<value>.*)",
    re.DOTALL,
)
_TOOL_ERROR_SUMMARY = "Error executing tool %s: %s"
_REJECTED_ARGUMENT_SUMMARY = "Rejected tool call — %s, but received %s"
_WITHHELD_SUMMARY = "SDK tool wrapper record withheld: %s"
_WITH_FRAMES = "{}\n%s"
# Every template this filter writes, so a record it already rewrote (a second
# install, a handler that filters again) passes unchanged.
_SUMMARIES = frozenset(
    template
    for summary in (_TOOL_ERROR_SUMMARY, _REJECTED_ARGUMENT_SUMMARY, _WITHHELD_SUMMARY)
    for template in (summary, _WITH_FRAMES.format(summary))
)


class SdkToolLogSummary(logging.Filter):
    """Keep tool content out of the SDK tool wrapper's log records (#3919).

    The SDK wrapper logs a tool's exception message, and up to 120 characters
    of an argument that failed its declared type. A tool's exception can quote
    what it read from the store, which keeps it encrypted, and the host log is
    plaintext at every level. Core's own wrapper logs the exception's class and
    :func:`content_log_summary` instead; this gives the SDK wrapper's records
    the same shape, so an extracted feature's tools log what an in-tree
    feature's tools log.

    Only log records change; what the wrapper returns to the model is the
    SDK's. A record from the wrapper that has neither known shape is withheld
    whole, its summary kept, rather than passed through: a reworded SDK record
    must not start leaking again.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.funcName != _SDK_TOOL_WRAPPER_FUNCTION or (
            isinstance(record.msg, str) and record.msg in _SUMMARIES
        ):
            return True
        # Read before the record changes: the wrapper logs from inside its
        # ``except``, so the tool's exception is the one being handled.
        error = sys.exception()
        record.msg, record.args = _summary(record, error)
        if record.exc_info:
            # Rendering ``exc_info`` ends with ``Type: message``. Keep where
            # it failed, not what it said.
            logged = record.exc_info[1]
            record.msg = _WITH_FRAMES.format(record.msg)
            record.args = (*record.args, exception_frames(logged) if logged else "")
            record.exc_info = None
            record.exc_text = None
        return True


def _summary(record: logging.LogRecord, error: BaseException | None):
    """The content-free ``(msg, args)`` for one SDK wrapper record."""
    if (
        record.msg == _SDK_REJECTED_CALL
        and isinstance(record.args, tuple)
        and len(record.args) == 1
    ):
        match = _SDK_REJECTED_ARGUMENT.fullmatch(str(record.args[0]))
        if match:
            return _REJECTED_ARGUMENT_SUMMARY, (
                match["head"], content_log_summary(match["value"]),
            )
    try:
        rendered = record.getMessage()
    except (TypeError, ValueError):
        # A record whose arguments do not fit its format; raising here would
        # escape into the wrapper's own ``except`` and fail the tool call.
        rendered = repr((record.msg, record.args))
    if error is not None:
        match = _SDK_TOOL_ERROR.match(rendered)
        if match and rendered[match.end():] == str(error):
            return _TOOL_ERROR_SUMMARY, (match["tool"], exception_log_summary(error))
    return _WITHHELD_SUMMARY, (content_log_summary(rendered),)


_SDK_TOOL_LOG_SUMMARY = SdkToolLogSummary()


def summarize_sdk_tool_logs() -> None:
    """Install :class:`SdkToolLogSummary` on the SDK feature logger, once.

    A logger's own filter sees every record that logger creates, whichever
    handlers (the host log, the launch log, a test's capture) receive it.
    """
    _sdk_feature_base.logger.addFilter(_SDK_TOOL_LOG_SUMMARY)
