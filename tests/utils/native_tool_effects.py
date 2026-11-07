"""Drive component probes inside the real Agent request and journal lifetime."""

from uuid import uuid4

from tests.unit.test_tool_effect_invocation import governed
from kestrel_sovereign.auth import current_caller_context


async def drive_native(host, operation):
    """A fresh input owns one native lifetime; retirement errors remain errors.

    Tests of redelivery use their original operation IDs explicitly instead.
    """
    @governed
    async def invoke(agent, user_input, *, invocation_id=None, caller=None):
        return await operation()

    return await invoke(host, "Owned native component input", invocation_id=uuid4().hex,
        caller=current_caller_context())
