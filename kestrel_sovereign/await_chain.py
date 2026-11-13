"""Where a suspended asyncio task is parked, across task boundaries (#4001).

``Task.get_stack()`` returns a single frame for a suspended coroutine, and
``asyncio.format_call_graph`` (Python 3.14 and later) shows who awaits a task,
never what the task is waiting on. Neither names the frame a stuck turn is
blocked in: ``process_input`` runs its turn in an isolated child task
(``invocation-turn:<id>``), and the turn parks inside async context managers,
whose bodies are async generators driven by ``__aenter__``/``__aexit__``.

This walk follows a coroutine's ``cr_await`` (a generator's ``gi_yieldfrom``)
down to what it awaits. Where that is an async generator being driven through
``anext``/``athrow`` by an ``@asynccontextmanager``, it continues into the
generator's own frame and ``ag_await``. From a task it follows the task it
awaits, the tasks behind a ``gather``, and the tasks joined through
``asyncio.wait`` or ``await_owned_task``, which wait on a private future: the
journal retirement that held the privacy lock in #4001 joined its Tasks that
way.

asyncio keeps the awaited future as ``Task._fut_waiter`` and a ``gather``'s
tasks as ``_children`` (unchanged from 3.11 through 3.14). Neither is public
API, nor are ``asyncio.tasks._wait`` and
``contextlib._AsyncGeneratorContextManager``: if any of them moves, the walk
stops at the awaited future instead of following it, and still reports every
frame up to it. The walk reads attributes only of the interpreter's own
coroutine, generator and async-generator objects and of known asyncio frames,
compares other values by exact type, and so never calls code it finds there.
"""

from __future__ import annotations

import asyncio
import contextlib
import types
from dataclasses import dataclass
from typing import Any, Iterator, Optional

#: A chain longer than this is cut, saying so, so one report stays readable.
MAX_AWAIT_CHAIN_LINES = 120

#: The deepest chain of awaited tasks followed before the walk stops.
MAX_AWAITED_TASK_DEPTH = 16


@dataclass(frozen=True)
class AwaitFrame:
    """One suspended frame: its qualified name, file and current line."""

    name: str
    filename: str
    lineno: int

    def render(self) -> str:
        return f"{self.name} ({self.filename}:{self.lineno})"


@dataclass(frozen=True)
class AwaitNode:
    """One task in the chain: its frames, outermost first, and what it awaits."""

    depth: int
    task: asyncio.Task
    frames: tuple[AwaitFrame, ...]
    waiter: Any
    awaited_tasks: tuple[asyncio.Task, ...]
    #: Tasks it awaits that the walk did not follow (depth limit).
    unfollowed: int = 0


_ASYNC_GENERATOR_STEPS = frozenset({"async_generator_asend", "async_generator_athrow"})
# Private names, resolved once; a missing one only shortens the walk.
_ASYNC_GENERATOR_CONTEXT_MANAGER = getattr(contextlib, "_AsyncGeneratorContextManager", None)
_ASYNCIO_WAIT_CODE = getattr(getattr(asyncio.tasks, "_wait", None), "__code__", None)
_GATHERING_FUTURE = getattr(asyncio.tasks, "_GatheringFuture", None)
# The interpreter's own suspendable objects, and the attributes that hold
# their frame and what they await. Nothing else is read.
_SUSPENDED_ATTRIBUTES = {
    types.CoroutineType: ("cr_frame", "cr_await"),
    types.AsyncGeneratorType: ("ag_frame", "ag_await"),
    types.GeneratorType: ("gi_frame", "gi_yieldfrom"),
}


def _is_task(value: Any) -> bool:
    """Whether ``value`` is an asyncio Task, judged by its type alone.

    ``isinstance`` would read a mismatched value's ``__class__``, which a
    proxy can compute.
    """
    return issubclass(type(value), asyncio.Task)


def _frame_of(awaitable: Any):
    attributes = _SUSPENDED_ATTRIBUTES.get(type(awaitable))
    return None if attributes is None else getattr(awaitable, attributes[0])


def _awaited_by(awaitable: Any) -> Any:
    attributes = _SUSPENDED_ATTRIBUTES.get(type(awaitable))
    return None if attributes is None else getattr(awaitable, attributes[1])


def _driven_async_generator(frame) -> Optional[Any]:
    """The async generator ``frame`` is stepping through ``anext``/``athrow``.

    The ``async_generator_asend`` object a coroutine awaits exposes no link to
    its generator. ``_AsyncGeneratorContextManager`` keeps it as ``self.gen``;
    otherwise it is the running async generator among the frame's locals.
    """
    local_values = frame.f_locals
    owner = local_values.get("self")
    candidates = list(local_values.values())
    if _ASYNC_GENERATOR_CONTEXT_MANAGER is not None and issubclass(
        type(owner), _ASYNC_GENERATOR_CONTEXT_MANAGER
    ):
        candidates.insert(0, owner.gen)
    for candidate in candidates:
        if type(candidate) is types.AsyncGeneratorType and candidate.ag_running:
            return candidate
    return None


def _suspended_frames(task: asyncio.Task) -> list:
    """The frame objects of ``task``'s coroutine chain, outermost first."""
    frames = []
    awaitable: Any = task.get_coro()
    seen: set[int] = set()
    while awaitable is not None and id(awaitable) not in seen:
        seen.add(id(awaitable))
        frame = _frame_of(awaitable)
        if frame is None:
            break
        frames.append(frame)
        awaited = _awaited_by(awaitable)
        if type(awaited).__name__ in _ASYNC_GENERATOR_STEPS:
            generator = _driven_async_generator(frame)
            if generator is not None:
                awaited = generator
        awaitable = awaited
    return frames


def _render_frame(frame) -> AwaitFrame:
    return AwaitFrame(frame.f_code.co_qualname, frame.f_code.co_filename, frame.f_lineno)


def task_frames(task: asyncio.Task) -> tuple[AwaitFrame, ...]:
    """The suspended frames of ``task``'s coroutine chain, outermost first."""
    return tuple(_render_frame(frame) for frame in _suspended_frames(task))


def _joined_by_frame(frame) -> list:
    """Tasks a known join frame waits for through a private future."""
    code = frame.f_code
    if _ASYNCIO_WAIT_CODE is not None and code is _ASYNCIO_WAIT_CODE:
        joined = frame.f_locals.get("fs")
    elif code.co_name == "await_owned_task" and code.co_filename.endswith(
        "_async_ownership.py"
    ):
        joined = frame.f_locals.get("task")
    else:
        return []
    if _is_task(joined):
        return [joined]
    if type(joined) in (set, frozenset, list, tuple):
        return [item for item in joined if _is_task(item)]
    return []


def _awaited_tasks(waiter: Any, frames: list) -> list[asyncio.Task]:
    if _is_task(waiter):
        behind = [waiter]
    elif _GATHERING_FUTURE is not None and issubclass(type(waiter), _GATHERING_FUTURE):
        children = waiter._children
        behind = list(children) if type(children) in (list, tuple) else []
    else:
        behind = []
        for frame in reversed(frames):
            behind = _joined_by_frame(frame)
            if behind:
                break
    return [child for child in behind if _is_task(child) and not child.done()]


def walk_await_chain(task: asyncio.Task) -> Iterator[AwaitNode]:
    """Yield ``task`` and, depth first, every task it is waiting on."""
    seen: set[asyncio.Task] = set()

    def visit(node: asyncio.Task, depth: int) -> Iterator[AwaitNode]:
        seen.add(node)
        waiter = getattr(node, "_fut_waiter", None)
        frames = _suspended_frames(node)
        children = tuple(
            child for child in _awaited_tasks(waiter, frames) if child not in seen
        )
        unfollowed = 0
        if depth >= MAX_AWAITED_TASK_DEPTH:
            unfollowed, children = len(children), ()
        yield AwaitNode(
            depth,
            node,
            tuple(_render_frame(frame) for frame in frames),
            waiter,
            children,
            unfollowed,
        )
        for child in children:
            if child not in seen:
                yield from visit(child, depth + 1)

    yield from visit(task, 0)


def _describe_waiter(waiter: Any) -> str:
    """Name the awaited object without running its ``__repr__``.

    Only asyncio's own future and task types are asked whether they are done.
    """
    if type(waiter) in (asyncio.Future, asyncio.Task):
        state = "done" if waiter.done() else "pending"
        return f"<{type(waiter).__name__} {state}>"
    return f"<{type(waiter).__name__}>"


def format_await_chain(task: asyncio.Task) -> str:
    """Render where ``task`` is parked: each task's frames, outermost first.

    The chain ends where the work is actually suspended, past every task
    boundary it awaits, rather than at the first task. A finished task, or a
    task that has not started, says so instead.
    """
    if task.done():
        return f"task {task.get_name()!r} has finished"
    lines: list[str] = []
    for node in walk_await_chain(task):
        indent = "  " * (node.depth + 1)
        lines.append(f"{indent}task {node.task.get_name()!r}:")
        if not node.frames:
            lines.append(f"{indent}  (no suspended frame: not started, or running now)")
        lines.extend(f"{indent}  {frame.render()}" for frame in node.frames)
        if node.unfollowed:
            lines.append(
                f"{indent}  ... {node.unfollowed} awaited task(s) beyond "
                f"depth {MAX_AWAITED_TASK_DEPTH} not followed"
            )
        elif node.waiter is not None and not node.awaited_tasks:
            lines.append(f"{indent}  awaiting {_describe_waiter(node.waiter)}")
    if len(lines) > MAX_AWAIT_CHAIN_LINES:
        omitted = len(lines) - MAX_AWAIT_CHAIN_LINES
        lines = lines[:MAX_AWAIT_CHAIN_LINES]
        lines.append(f"  ... {omitted} more line(s) omitted")
    return "\n".join(lines)


def parked_frames(task: asyncio.Task) -> tuple[AwaitFrame, ...]:
    """The innermost frame of each branch of ``task``'s chain that is waiting."""
    return tuple(
        node.frames[-1]
        for node in walk_await_chain(task)
        if not node.awaited_tasks and node.frames
    )


__all__ = [
    "AwaitFrame",
    "AwaitNode",
    "MAX_AWAIT_CHAIN_LINES",
    "MAX_AWAITED_TASK_DEPTH",
    "format_await_chain",
    "parked_frames",
    "task_frames",
    "walk_await_chain",
]
