"""Runtime enforcement of a SpawnMandate's ``restricted_tools`` (#2137).

A spawn mandate's ``additional_constraints`` are validated at spawn time
(``ScopedConstitution.validate_constraints``) and woven into the child's
anchored constitution (soft, system-prompt enforcement). This hook adds the
*hard* runtime guarantee the mandate implies: a child may not actually invoke a
tool its mandate lists under ``restricted_tools``. It denies at PRE_TOOL_USE,
before the tool executes, using the same blocking-decision path as every other
security hook.
"""

from __future__ import annotations

from typing import Iterable

from kestrel_sdk.hooks.base import Hook, HookEvent, HookInput, HookOutput


class MandateRestrictionHook(Hook):
    """PRE_TOOL_USE hook that hard-denies mandate-restricted tools.

    Registered on a spawned child whose mandate carries a non-empty
    ``restricted_tools`` list. Tool names are matched exactly against the
    dispatched ``tool_name``.
    """

    def __init__(self, restricted_tools: Iterable[str], *, priority: int = 10):
        # Low priority number = runs early, so the denial lands before slower
        # permission/audit hooks do their work.
        super().__init__(
            name="mandate_restriction",
            events=[HookEvent.PRE_TOOL_USE],
            priority=priority,
        )
        self._restricted = {str(t) for t in (restricted_tools or [])}

    @property
    def restricted_tools(self) -> frozenset[str]:
        return frozenset(self._restricted)

    async def execute(self, input: HookInput) -> HookOutput:
        if input.tool_name in self._restricted:
            return HookOutput.deny(
                f"Tool '{input.tool_name}' is restricted by this agent's spawn "
                f"mandate and may not be used."
            )
        return HookOutput.allow()
