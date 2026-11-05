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

from kestrel_sovereign.execution_custody import is_tool_execution_terminal_error


def codex_inline_failure(error: Exception) -> dict[str, Any]:
    """What the codex inline handler returns to the model for a raised error.

    A terminal error is re-raised, as the turn owner raises it.
    """
    if is_tool_execution_terminal_error(error):
        raise error
    return {"success": False, "error": f"{error}"}


class PreTurnReader:
    """The codex reader loop, started before the skill whose tools it runs.

    Each inline call runs on a task created from the reader's own context,
    which predates the skill's ``handle_task``.
    """

    def __init__(self) -> None:
        self._calls: asyncio.Queue = asyncio.Queue()
        self._handlers: set[asyncio.Task] = set()
        self._reader: asyncio.Task | None = None

    async def start(self) -> None:
        self._reader = asyncio.create_task(self._read())
        # Yield so the reader is running before the caller goes on.
        await asyncio.sleep(0)

    async def _read(self) -> None:
        while (call := await self._calls.get()) is not None:
            handler = asyncio.create_task(self._handle(*call))
            self._handlers.add(handler)
            handler.add_done_callback(self._handlers.discard)

    @staticmethod
    async def _handle(executor, name, args, done: asyncio.Future) -> None:
        try:
            _, result = await executor(name, args)
        except Exception as error:
            try:
                result = codex_inline_failure(error)
            except Exception as terminal:
                done.set_exception(terminal)
                return
        done.set_result(result)

    async def call(self, executor, name: str, args: dict | None = None) -> Any:
        """Run ``executor(name, args)`` on a reader task; the model's result."""
        done = asyncio.get_running_loop().create_future()
        await self._calls.put((executor, name, args or {}, done))
        return await done

    async def stop(self) -> None:
        await self._calls.put(None)
        if self._reader is not None:
            await self._reader
