"""Drive component probes inside the real Agent request and journal lifetime."""

from uuid import uuid4

from tests.unit.test_tool_effect_invocation import governed
from kestrel_sovereign.auth import current_caller_context


async def drive_native(host, operation, *, invocation_id=None):
    """A fresh input owns one native lifetime; retirement errors remain errors.

    An explicit invocation ID exercises redelivery of that same original input.
    """
    @governed
    async def invoke(agent, user_input, *, invocation_id=None, caller=None):
        return await operation()

    return await invoke(host, "Owned native component input", invocation_id=uuid4().hex if invocation_id is None else invocation_id,
        caller=current_caller_context())
