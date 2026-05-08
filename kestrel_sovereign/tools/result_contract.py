"""Registry-time validator for the ToolResult return contract.

#1042 layer 4 (the ``ToolResult`` envelope) requires every ``@tool``
method to return :class:`kestrel_sdk.tools.result.ToolResult`. Six
extracted feature packages have already been migrated; this module
guards the migration as it rolls into the framework's 235 in-tree
tools (issue #1061).

The validator is **scoped** — only modules listed in
:data:`MIGRATED_FEATURE_MODULES` are checked. Pre-migration features
keep returning ``Dict[str, Any]`` and the validator stays silent for
them. As each new module migrates, its dotted name is appended to the
allowlist and registration immediately fails for any backslide.

The validator runs at @tool-discovery time inside
``ToolRegistryMixin._register_explored_feature_tools`` (and in unit
tests via :func:`assert_feature_returns_tool_result`).

Why static + runtime, not just one:

  - Static: catches forgot-to-update-annotation and forgot-to-return
    cases at registration. Annotations are the source of truth the
    LLM-facing system prompt assembly will read once narration-honesty
    layer 3 (PR-E piece 4) lands.
  - Runtime (deferred to PR-E piece 4): catches the case where a tool
    correctly annotates ``-> ToolResult`` but a code-path returns a
    bare dict. Out of scope for this pilot — added when the
    ResponseAuditHook narration check becomes deterministic via
    ToolCallStarted from #1048 Wave 5.
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import Any, List, Optional, get_type_hints

from kestrel_sdk.tools.result import ToolResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Allowlist of migrated feature modules
# ---------------------------------------------------------------------------
#
# Add a module's dotted name here only when every @tool method in that
# module returns ``ToolResult``. Every name in this set MUST pass the
# validator; a regression triggers a hard registration error.
#
# When extending the allowlist, also bump the kestrel-sovereign version
# (the doctored honesty contract layer).
MIGRATED_FEATURE_MODULES: frozenset[str] = frozenset({
    "kestrel_sovereign.features.tasks.feature",
    "kestrel_sovereign.features.memory.feature",
    "kestrel_sovereign.features.context.feature",
    "kestrel_sovereign.features.bootstrap.feature",
    "kestrel_sovereign.features.security.feature",
    "kestrel_sovereign.features.health.feature",
    "kestrel_sovereign.features.identity.feature",
    "kestrel_sovereign.features.council.feature",
    "kestrel_sovereign.features.memory_agency.feature",
    "kestrel_sovereign.features.save.feature",
    "kestrel_sovereign.features.scheduler.feature",
    "kestrel_sovereign.features.reflection.feature",
    "kestrel_sovereign.features.model.feature",
    "kestrel_sovereign.features.keys.feature",
    "kestrel_sovereign.features.strategic_memory.feature",
    "kestrel_sovereign.features.compute.feature",
    "kestrel_sovereign.features.github.feature",
    "kestrel_sovereign.features.talon.coordinator",
    "kestrel_sovereign.features.peers.feature",
})


# ---------------------------------------------------------------------------
# Behavior: hard-fail (default) vs. warn
# ---------------------------------------------------------------------------
#
# Default behavior is to RAISE on a violation: the framework refuses to
# register a feature whose @tool methods don't honor the contract. The
# alternative — log-and-continue — would let an honesty-violating tool
# silently ship.
#
# An escape hatch is provided for emergency rollback only: setting
# ``KESTREL_TOOL_RESULT_CONTRACT_WARN_ONLY=1`` downgrades violations to
# WARNING. This exists for the case where an upstream SDK change has
# broken the annotation walk but the feature is otherwise correct;
# users should fix the annotation, not pin the env var.

_WARN_ONLY_ENV = "KESTREL_TOOL_RESULT_CONTRACT_WARN_ONLY"


class ToolResultContractError(TypeError):
    """Raised when a migrated feature ships a non-ToolResult @tool."""


def _module_name_for(feature: Any) -> Optional[str]:
    """Return the dotted module name of a feature instance / class."""
    cls = feature if inspect.isclass(feature) else type(feature)
    return getattr(cls, "__module__", None)


def _is_tool_result_annotation(annotation: Any) -> bool:
    """Return True if ``annotation`` is exactly ``ToolResult``.

    We accept the runtime class only (no Optional[ToolResult], no
    Union[..., ToolResult]). The contract is that every code path
    returns a ToolResult — Optional means the tool can also return
    None, which is exactly what the contract forbids.
    """
    return annotation is ToolResult


def _iter_tool_methods(feature: Any):
    """Yield ``(name, method)`` pairs for every @tool method on a feature."""
    for name, method in inspect.getmembers(feature, predicate=inspect.ismethod):
        if hasattr(method, "_tool_schema"):
            yield name, method


def find_violations(feature: Any) -> List[str]:
    """Return a list of human-readable violation strings for a feature.

    Empty list = clean. Each violation string names the offending tool
    and the annotation we found, suitable for pasting into a log line
    or test failure message.

    The caller decides whether to raise, warn, or both — see
    :func:`enforce_tool_result_contract`.
    """
    violations: List[str] = []
    cls = feature if inspect.isclass(feature) else type(feature)
    try:
        # ``get_type_hints`` resolves forward refs and string annotations
        # against the method's defining module — the safest source.
        hints_by_method = {
            name: get_type_hints(method)
            for name, method in _iter_tool_methods(feature)
        }
    except Exception as e:
        # If we can't introspect at all, surface a single violation
        # rather than masking the migration regression.
        return [
            f"could not resolve type hints for {cls.__qualname__}: {e}"
        ]

    for name, _method in _iter_tool_methods(feature):
        hints = hints_by_method.get(name, {})
        return_ann = hints.get("return", inspect.Parameter.empty)
        if return_ann is inspect.Parameter.empty:
            violations.append(
                f"{cls.__qualname__}.{name}: missing return annotation "
                "(must be ToolResult per #1042 layer 4 / #1061)"
            )
            continue
        if not _is_tool_result_annotation(return_ann):
            violations.append(
                f"{cls.__qualname__}.{name}: return annotation is "
                f"{return_ann!r}, expected ToolResult "
                "(#1042 layer 4 / #1061)"
            )
    return violations


def is_migrated_module(module_name: Optional[str]) -> bool:
    """Return True if ``module_name`` is in the migrated allowlist."""
    return bool(module_name) and module_name in MIGRATED_FEATURE_MODULES


def enforce_tool_result_contract(feature: Any) -> None:
    """Validate ``feature`` against the ToolResult contract.

    No-op for features whose module is not on the migrated allowlist.
    For migrated features, raises :class:`ToolResultContractError`
    unless ``KESTREL_TOOL_RESULT_CONTRACT_WARN_ONLY=1`` is set, in
    which case violations are logged at WARNING.
    """
    module_name = _module_name_for(feature)
    if not is_migrated_module(module_name):
        return

    violations = find_violations(feature)
    if not violations:
        return

    msg = (
        f"ToolResult contract violations in {module_name}:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )

    if os.environ.get(_WARN_ONLY_ENV, "").strip() in ("1", "true", "yes"):
        logger.warning(msg)
        return

    raise ToolResultContractError(msg)


def assert_feature_returns_tool_result(feature: Any) -> None:
    """Test-friendly alias that always raises on violation.

    Use from unit tests to pin the contract regardless of the
    ``KESTREL_TOOL_RESULT_CONTRACT_WARN_ONLY`` env var (which exists
    for emergency rollback in production, not for tests).
    """
    violations = find_violations(feature)
    if violations:
        cls = feature if inspect.isclass(feature) else type(feature)
        raise ToolResultContractError(
            f"ToolResult contract violations in "
            f"{cls.__module__}.{cls.__qualname__}:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )
