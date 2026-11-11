"""Custody of an admitted private native Gateway process group and waiter."""
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from collections.abc import Callable
import asyncio
import os
import signal

from kestrel_sovereign._async_process import _process_tree_alive
from kestrel_sovereign._async_ownership import await_owned_operation, raise_owned_outcome


@dataclass
class OriginalGatewayGroup:
    """A private group receipt; its leader's exit cannot retire its children.

    Construct only from the original Popen or a birth/command/executable/session
    attested private admission. ``pid`` is the retained original private PGID,
    which the canonical POSIX group probe reads. A missing original leader may
    have no waiter only after actual native absence has been established.
    """

    pid: int
    wait: Callable[[float], object] | None
    exited: Callable[[], bool]
    _group_extinct: bool = field(default=False, init=False)

    def __post_init__(self):
        assert os.name == 'posix' and self.pid > 1 and self.pid != os.getpgrp()

    def group_alive(self):
        if not self._group_extinct and not _process_tree_alive(self):
            # A terminal receipt never adopts a later group reusing this ID.
            self._group_extinct = True
        return not self._group_extinct

    def retired(self):
        return not self.group_alive() and self.exited()

    async def retire(self):
        async def phase(process_signal):
            deadline = asyncio.get_running_loop().time() + 5
            joined = False

            async def join():
                nonlocal joined
                if self.wait is not None and not joined:
                    remaining = max(0.0, deadline - asyncio.get_running_loop().time())
                    raise_owned_outcome(await await_owned_operation(
                        asyncio.to_thread(self.wait, remaining)),
                        operation='original native Gateway waiter')
                    joined = True

            if self.group_alive():
                try:
                    os.killpg(self.pid, process_signal)
                except ProcessLookupError:
                    # Group disappearance is still not the original join.
                    self.group_alive()
                except PermissionError:
                    # macOS can report EPERM for an unreaped zombie with no
                    # signalable member. Join the exact original, then retry
                    # if the group still exists; genuine refusal still raises.
                    await join()
                    if self.group_alive():
                        os.killpg(self.pid, process_signal)
            await join()
            async with asyncio.timeout_at(deadline):
                while not self.retired():
                    await asyncio.sleep(0.01)

        async def observe_phase(process_signal):
            raise_owned_outcome(await await_owned_operation(phase(process_signal)),
                operation='original native Gateway group retirement')

        async def force_if_live():
            if not self.retired():
                await observe_phase(signal.SIGKILL)

        if self.retired():
            return
        async with AsyncExitStack() as retirement:
            retirement.push_async_callback(force_if_live)
            await observe_phase(signal.SIGTERM)
