"""Bounded, read-only diagnostics for explicitly retained native test Tasks.

No global scan, task creation, cancellation, exception consumption or resource
takeover. Frames expose code locations only, never their argument values.
"""
import asyncio
import logging
import threading
from contextlib import contextmanager
from itertools import islice

from kestrel_sovereign.async_ownership import OwnedOperation
from kestrel_sovereign._async_ownership import OwnedAsyncIterator, OwnedSourceAdmission
from kestrel_sovereign.execution_custody import _ExecutionForwarder
from tests.utils.virtual_clock import HANG_GUARD_SECONDS


def original_task_tails(tasks):
    lines = []
    visited = set()
    inspected = set()

    def originals(value, nested, depth=0):
        if depth > 4 or id(value) in inspected or len(inspected) >= 64:
            return
        if isinstance(value, asyncio.Task):
            nested.append(value)
        elif isinstance(value, OwnedOperation):
            inspected.add(id(value))
            if value.task is not None:
                nested.append(value.task)
        elif isinstance(value, OwnedSourceAdmission):
            inspected.add(id(value))
            for receipt in (value.producer, value.watcher, value.retirement):
                if receipt is not None and isinstance(receipt.task, asyncio.Task):
                    nested.append(receipt.task)
        elif isinstance(value, OwnedAsyncIterator):
            inspected.add(id(value))
            originals(value._source_admission, nested, depth + 1)
        elif isinstance(value, _ExecutionForwarder):
            inspected.add(id(value))
            for original in (value._source_admission, value._iterator, value._close_operation):
                originals(original, nested, depth + 1)

    def describe(task):
        if task in visited or len(visited) >= 16:
            return
        visited.add(task)
        lines.append(f"original-task {task.get_name()} done={task.done()} cancelling={task.cancelling()}")
        if task.done():
            return
        operation = task.get_coro()
        nested = []
        for _ in range(48):
            frame = getattr(operation, "cr_frame", None) or getattr(operation, "ag_frame", None)
            if frame is not None:
                lines.append(f"  {frame.f_code.co_filename}:{frame.f_lineno}:{frame.f_code.co_name}")
                for value in islice(frame.f_locals.values(), 64):
                    originals(value, nested)
            following = getattr(operation, "cr_await", None) or getattr(operation, "ag_await", None)
            if following is None or following is operation:
                break
            operation = following
        for original in nested:
            describe(original)

    for task in tasks:
        if len(visited) >= 16:
            break
        describe(task)
    return tuple(lines)


@contextmanager
def original_task_tail_diagnostics(tasks, *, seconds=HANG_GUARD_SECONDS):
    """One real-time, read-only snapshot, independent of a test's loop clock.

    This is not a timeout, completion observer or cleanup policy. The original
    Task keeps running unchanged; pytest's existing guard still decides failure.
    The exact finite timer is cancelled and joined before the scope returns.
    """
    retained = tuple(islice(tasks, 16))
    loop = asyncio.get_running_loop()
    active = True

    def report():
        if active:
            logging.getLogger(__name__).warning(
                "Original native Task tails before test hang limit:\n%s",
                "\n".join(original_task_tails(retained)),
            )

    timer = threading.Timer(seconds, loop.call_soon_threadsafe, args=(report,))
    timer.name = "original-native-task-tail-diagnostic"
    timer.start()
    try:
        yield timer
    finally:
        active = False
        timer.cancel()
        timer.join(1)
        if timer.is_alive():
            raise RuntimeError("original native Task diagnostic timer did not retire")
