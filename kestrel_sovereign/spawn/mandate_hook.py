"""Runtime enforcement of a SpawnMandate's ``restricted_tools`` (#2137).

A spawn mandate's ``additional_constraints`` are validated at spawn time
(``ScopedConstitution.validate_constraints``) and woven into the child's
anchored constitution (soft, system-prompt enforcement). This hook adds the
*hard* runtime guarantee the mandate implies: a child may not actually invoke a
tool its mandate lists under ``restricted_tools``. It denies at PRE_TOOL_USE,
before the tool executes, using the same blocking-decision path as every other
security hook.

Some tools must be narrowed at the *argument* level rather than denied
outright — e.g. ``workflow_run`` is allowed, but only to start one specific
workflow. ``restricted_tool_args`` expresses that: a per-tool allowlist of
argument values. Enforcement is a positive allowlist (deny unless the observed
argument value is explicitly permitted), so it fails closed on a missing or
unexpected value (#2321).
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from kestrel_sdk.hooks.base import Hook, HookEvent, HookInput, HookOutput


class MandateRestrictionHook(Hook):
    """PRE_TOOL_USE hook that hard-denies mandate-restricted tools.

    Registered on a spawned child whose mandate carries a non-empty
    ``restricted_tools`` list and/or ``restricted_tool_args`` map. Tool names
    are matched exactly against the dispatched ``tool_name``; argument
    allowlists are matched against ``tool_input``.
    """

    def __init__(
        self,
        restricted_tools: Iterable[str],
        *,
        restricted_tool_args: Mapping[str, Mapping[str, Iterable[Any]]] | None = None,
        priority: int = 10,
    ):
        # Low priority number = runs early, so the denial lands before slower
        # permission/audit hooks do their work.
        super().__init__(
            name="mandate_restriction",
            events=[HookEvent.PRE_TOOL_USE],
            priority=priority,
        )
        self._restricted = {str(t) for t in (restricted_tools or [])}
        # Normalize to {tool_name: {arg_name: {allowed_str_value, ...}}}. Values
        # are compared as strings so callers need not match the tool's exact
        # argument type.
        self._arg_allow: dict[str, dict[str, set[str]]] = {}
        for tool_name, arg_spec in (restricted_tool_args or {}).items():
            if not isinstance(arg_spec, Mapping):
                continue
            norm: dict[str, set[str]] = {}
            for arg_name, allowed in arg_spec.items():
                if isinstance(allowed, (list, tuple, set, frozenset)):
                    values = {str(v) for v in allowed}
                else:
                    values = {str(allowed)}
                norm[str(arg_name)] = values
            if norm:
                self._arg_allow[str(tool_name)] = norm

    @property
    def restricted_tools(self) -> frozenset[str]:
        return frozenset(self._restricted)

    @property
    def restricted_tool_args(self) -> dict[str, dict[str, frozenset[str]]]:
        return {
            tool: {arg: frozenset(vals) for arg, vals in spec.items()}
            for tool, spec in self._arg_allow.items()
        }

    async def execute(self, input: HookInput) -> HookOutput:
        if input.tool_name in self._restricted:
            return HookOutput.deny(
                f"Tool '{input.tool_name}' is restricted by this agent's spawn "
                f"mandate and may not be used."
            )
        arg_rules = self._arg_allow.get(input.tool_name)
        if arg_rules:
            tool_input = input.tool_input or {}
            for arg_name, allowed in arg_rules.items():
                # Positive allowlist: fail closed when the argument is absent or
                # its value is not explicitly permitted.
                value = str(tool_input.get(arg_name))
                if value not in allowed:
                    return HookOutput.deny(
                        f"Tool '{input.tool_name}' may only be used with "
                        f"'{arg_name}' in {sorted(allowed)} per this agent's "
                        f"spawn mandate (got {tool_input.get(arg_name)!r})."
                    )
        return HookOutput.allow()
