"""Negative control: generic suppression keeps assignment conditional."""

from kestrel_sovereign.execution_custody import owned_native_http_client
from tests.static.owned_native_http_client_contracts import SuppressingClient, value


async def generic_assignment_remains_conditional(
    client: SuppressingClient, fail: bool,
) -> str:
    async with owned_native_http_client(client):
        if fail:
            raise RuntimeError("before generic assignment")
        result = await value()
    return result
