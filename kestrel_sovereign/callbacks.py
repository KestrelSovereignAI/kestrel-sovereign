"""Canonical return-value ownership contract for host/provider callbacks.

A fresh coroutine runs in its caller's existing context. A Task/Future already
has another execution context: awaiting it cannot retrofit the original caller's
custody or native observation. Refusal never cancels or closes borrowed work.
"""

import inspect

from kestrel_sovereign.execution_custody import ExecutionAuthorityError


class HostCallbackOwnershipError(ExecutionAuthorityError):
    """An opaque original awaitable remains unresolved retirement debt."""

    def __init__(self, original):
        super().__init__(
            "host callbacks must return a synchronous result or an unstarted coroutine, not a pre-existing awaitable"
        )
        self.original_awaitable = original
        self._kestrel_native_retirement_debts = ((original, None),)


def require_callback_result(value):
    """Accept a synchronous answer, not unresolved work masquerading as data."""
    if inspect.isawaitable(value):
        raise HostCallbackOwnershipError(value)
    return value


async def invoke_host_callback(operation, *args, **kwargs):
    """Drive one fresh coroutine in the original context; inspect its result."""
    value = operation(*args, **kwargs)
    if (
        inspect.iscoroutine(value)
        and inspect.getcoroutinestate(value) == inspect.CORO_CREATED
    ):
        value = await value
    return require_callback_result(value)
