"""The codex app-server's inline tool dispatch, without a codex process.

``CodexAdapter``'s ``item/tool/call`` handler runs the turn's tool executor on
a task its reader loop creates. The reader starts before the turn, so that task
holds none of the turn's context unless the executor re-presents it. An error
the executor raises reaches the model as an ordinary tool failure, except a
terminal one (execution control, an unconfirmed required audit), which ends
the turn: the turn owner raises it.
"""

from __future__ import annotations

import asyncio
from typing import Any

from kestrel_sovereign.execution_custody import (
    execution_observation_error, is_tool_execution_terminal_error,
)
from kestrel_sovereign._async_ownership import OwnedCallbackScope, await_owned_task, owned_task_error


def codex_inline_failure(error: Exception) -> dict[str, Any]:
    """What the codex inline handler returns to the model for a raised error.

    A terminal error is re-raised, as the turn owner raises it.
    """
    if is_tool_execution_terminal_error(error):
        raise error
    return {"success": False, "error": f"{error}"}


class OwnedCallbackReader:
    """One pre-turn reader; each callback has actual owned Task admission."""

    def __init__(self) -> None:
        self._calls: asyncio.Queue = asyncio.Queue()
        self._handlers: set[asyncio.Task] = set()
        self._reader: asyncio.Task | None = None
        self.baseline: dict = {}

    async def start(self) -> None:
        self._reader = asyncio.create_task(self._read())
        await asyncio.sleep(0)

    async def _read(self) -> None:
        import contextvars

        self.baseline = dict(contextvars.copy_context().items())
        while (call := await self._calls.get()) is not None:
            scope, executor, name, args, done = call
            if scope.closed:
                continue
            try:
                handler = scope.start(
                    lambda executor=executor, name=name, args=args: self._execute(executor, name, args),
                    name='owned-test-codex-callback')
                self._handlers.add(handler)
                handler.add_done_callback(lambda task, done=done: self._complete(task, done))
            except BaseException as error:
                # An accepted Task remains in its original scope even when a
                # later admission/publication step fails. Dispatch joins it.
                self._publish_reply(done, error=error)

    async def _execute(self, executor, name, args):
        return await executor(name, args)

    @staticmethod
    def _publish_reply(done, *, result=None, error=None):
        if done.done():
            return
        if error is not None:
            done.set_exception(error)
        else:
            done.set_result(result)

    def _complete(self, task, done):
        self._handlers.discard(task)
        result, error = None, None
        try:
            error = owned_task_error(task)
            if error is None:
                result = task.result()
        except BaseException as observed:
            error = observed
        # Completion, rather than callback entry, is the sole reply publisher:
        # a failed native start gate never enters the callback body.
        self._publish_reply(done, result=result, error=error)

    async def dispatch(self, executor, name: str = 'tool', args: dict | None = None) -> Any:
        scope = OwnedCallbackScope()
        done = asyncio.get_running_loop().create_future()
        result, failures = None, []
        try:
            await self._calls.put((scope, executor, name, args or {}, done))
            async with scope.wait():
                result = await done
        except BaseException as error:
            failures.append(error)
        finally:
            scope.close()
            for task in tuple(scope.tasks):
                outcome = await await_owned_task(task)
                self._handlers.discard(task)
                if outcome.error is not None:
                    failures.append(outcome.error)
                if outcome.cancellation is not None:
                    failures.append(outcome.cancellation)
        terminal = execution_observation_error(*failures)
        if terminal is not None:
            raise terminal
        return result

    async def stop(self) -> None:
        await self._calls.put(None)
        if self._reader is not None:
            await self._reader


class PreTurnReader(OwnedCallbackReader):
    """Apply the Codex model-reply policy on the canonical owned reader."""

    async def _execute(self, executor, name, args):
        try:
            _, result = await executor(name, args)
        except Exception as error:
            return codex_inline_failure(error)
        return result

    async def call(self, executor, name: str, args: dict | None = None) -> Any:
        return await self.dispatch(executor, name, args)
