"""Bounded, read-only diagnostics for explicitly retained native test Tasks.

No global scan, task creation, cancellation, exception consumption or resource
takeover. Frames expose code locations only, never their argument values.
"""
import asyncio
from itertools import islice

from kestrel_sovereign.async_ownership import OwnedOperation


def original_task_tails(tasks):
    lines = []
    visited = set()

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
                    if isinstance(value, asyncio.Task):
                        nested.append(value)
                    elif isinstance(value, OwnedOperation) and value.task is not None:
                        nested.append(value.task)
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
