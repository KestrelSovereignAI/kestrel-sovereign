"""Runtime enforcement of a SpawnMandate's tool constraints (#2137).

A spawn mandate's ``additional_constraints`` are validated at spawn time
(``ScopedConstitution.validate_constraints``) and woven into the child's
anchored constitution (soft, system-prompt enforcement). This hook adds the
*hard* runtime guarantee the mandate implies: a child may not actually invoke a
tool outside its positive ``allowed_tools`` ceiling or listed under
``restricted_tools``. It denies at PRE_TOOL_USE, before the tool executes,
using the same blocking-decision path as every other security hook.

Some tools must be narrowed at the *argument* level rather than denied
outright — e.g. ``workflow_run`` is allowed, but only to start one specific
workflow. ``restricted_tool_args`` expresses that: a per-tool allowlist of
argument values. Enforcement is a positive allowlist (deny unless the observed
argument value is explicitly permitted), so it fails closed on a missing or
unexpected value (#2321).

A call that only continues another call takes that call's authority. Reading
a tool result the orchestrator clipped to a preview (``read_tool_output``)
returns characters the producing call already returned, so it is permitted
wherever the producing tool is, and a signed ceiling that predates the reader
needs no re-signing to read a permitted file whole (#3740).
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Optional

from kestrel_sdk.hooks.base import Hook, HookEvent, HookInput, HookOutput

#: Names the tool a call continues (see :class:`MandateRestrictionHook`).
ContinuedToolResolver = Callable[[str, Mapping[str, Any]], Optional[str]]


class MandateRestrictionHook(Hook):
    """PRE_TOOL_USE hook that hard-denies mandate-restricted tools.

    Registered on a spawned child whose mandate carries a non-empty
    ``allowed_tools`` ceiling, ``restricted_tools`` list, and/or
    ``restricted_tool_args`` map. Tool names are matched exactly against the
    dispatched ``tool_name``; argument allowlists are matched against
    ``tool_input``.

    ``continued_tool`` names, for a ``(tool_name, tool_input)`` call, the tool
    whose call it continues, or ``None``. A call outside the ``allowed_tools``
    ceiling is permitted when the tool it continues is inside it and not
    restricted. Listing the call's own tool in ``restricted_tools`` still
    denies it, and its argument allowlist still applies. Without a resolver no
    call is a continuation.
    """

    def __init__(
        self,
        restricted_tools: Iterable[str],
        *,
        allowed_tools: Iterable[str] | None = None,
        restricted_tool_args: Mapping[str, Mapping[str, Iterable[Any]]] | None = None,
        continued_tool: ContinuedToolResolver | None = None,
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
        # ``None`` means no positive ceiling. An explicit empty iterable means
        # no tool is permitted, which is a valid fail-closed restriction.
        self._allowed = (
            None if allowed_tools is None else {str(t) for t in allowed_tools}
        )
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
        self._continued_tool = continued_tool

    @property
    def restricted_tools(self) -> frozenset[str]:
        return frozenset(self._restricted)

    @property
    def allowed_tools(self) -> frozenset[str] | None:
        return None if self._allowed is None else frozenset(self._allowed)

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
        if (
            self._allowed is not None
            and input.tool_name not in self._allowed
            and not self._continues_permitted_tool(input)
        ):
            return HookOutput.deny(
                f"Tool '{input.tool_name}' is outside this agent's allowed-tools "
                "ceiling and may not be used."
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

    def _continues_permitted_tool(self, input: HookInput) -> bool:
        """True when the call only continues a call of a tool this ceiling permits."""
        if self._continued_tool is None or self._allowed is None:
            return False
        source = self._continued_tool(input.tool_name, input.tool_input or {})
        return (
            isinstance(source, str)
            and source != input.tool_name
            and source in self._allowed
            and source not in self._restricted
        )
