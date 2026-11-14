"""Fixed installed worker launcher and process-lifetime withdrawal latch.

The launcher blocks withdrawal across exec; the canonical resource installs
its latch before heavy imports and then unmasks. No event-loop shutdown can
restore SIGTERM's default action after the original cleanup has finished.
"""
from __future__ import annotations

import signal
import sys

_RETIRE_SIGNALS = (signal.SIGINT, signal.SIGTERM)


def _block_withdrawal():
    return signal.pthread_sigmask(signal.SIG_BLOCK, _RETIRE_SIGNALS)

# Executing this real file avoids Core.__init__ before the first owned guard.
# Python startup before this instruction remains unproved, never a zero ACK.
if __name__ == "__main__" and "--initialize" not in sys.argv[1:]:
    _block_withdrawal()

import os
from pathlib import Path


def files(package):
    from importlib.resources import files as installed_files
    return installed_files(package)


class ProcessLifetime:
    """One process owns this latch; importing the module installs no handler."""

    def __init__(self):
        self.requested = False
        self._loop = None
        self._callback = None
        for signum in _RETIRE_SIGNALS:
            signal.signal(signum, self._request)
        # Pending launcher/exec signals are handled before resource admission.
        signal.pthread_sigmask(signal.SIG_UNBLOCK, _RETIRE_SIGNALS)

    def _request(self, _signum, _frame):
        if not self.requested:
            self.requested = True
            loop = self._loop
            if loop is not None:
                # The synchronous handler never mutates async resource state.
                loop.call_soon_threadsafe(self._notify)

    def _notify(self):
        callback = self._callback
        if callback is not None:
            callback()

    def bind(self, loop, callback):
        if self._loop is not None or self._callback is not None:
            raise RuntimeError("original_process_lifetime_already_bound")
        self._loop, self._callback = loop, callback
        if self.requested:
            loop.call_soon_threadsafe(self._notify)

    async def wait_for_bootstrap(self, path, retirement):
        """Retire a worker whose bootstrap was never admitted.

        This deadline covers waiting only. Reading or validating bootstrap,
        native startup and their errors remain the caller's responsibility.
        Caller cancellation must still pass through the original lifetime.
        """
        import asyncio

        deadline = asyncio.timeout(30)
        try:
            async with deadline:
                while not path.exists() and not self.requested and not retirement.is_set():
                    await asyncio.sleep(.05)
        except TimeoutError:
            if not deadline.expired():
                raise
            return False
        return not self.requested and not retirement.is_set()

    def detach(self):
        # Python finalization resets caught dispositions before native exit.
        # After the original main joins, keep withdrawal blocked across that
        # terminal interval; no accepting resource or callback remains alive.
        _block_withdrawal()
        self._loop = self._callback = None

    async def run(self, operation):
        original, result = None, None
        try:
            result = await operation
        except BaseException as error:
            original = error
        try:
            self.detach()
        except BaseException as closing:
            if original is not None:
                from kestrel_sovereign.execution_custody import execution_terminal_error
                raise execution_terminal_error(original, closing, retain_originals=True)
            raise
        if original is not None:
            raise original
        return result


def main() -> None:
    programs = {"browser": "two_domain_worker.py", "broker": "broker_worker.py"}
    arguments = sys.argv[1:]
    lane = (len(arguments) == 3 and arguments[0] == "broker"
            and arguments[1] == "--lane" and arguments[2] in {"normal", "priority"})
    if not lane and (len(arguments) not in (1, 2) or arguments[0] not in programs
            or len(arguments) == 2 and arguments[1] != "--initialize"):
        raise SystemExit("exact private worker role required")
    old_mask = (_block_withdrawal()
                if arguments[1:] != ["--initialize"] else None)
    try:
        resource = files("kestrel_sovereign.visual_computer").joinpath(
            "_worker_programs", programs[arguments[0]]
        )
        # Only the actual ordinary installed resource; no source fallback.
        program = Path(os.fspath(resource)).resolve(strict=True)
        if not program.is_file():
            raise SystemExit("installed private worker program absent")
        os.execv(sys.executable, [sys.executable, "-B", str(program), *arguments[1:]])
        raise RuntimeError("original_worker_exec_returned")
    except BaseException as original:
        # exec does not return. A library/unit caller retains its original mask
        # if lookup or exec fails; a real guarded process retains its block.
        if old_mask is not None:
            try:
                signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
            except BaseException as restoration:
                from kestrel_sovereign.execution_custody import execution_terminal_error
                raise execution_terminal_error(original, restoration, retain_originals=True)
        raise


if __name__ == "__main__":
    main()
