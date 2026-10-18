"""Original native stream Tasks remain owned at every constructor publication."""
import asyncio

import pytest

from kestrel_sovereign._async_ownership import (
    OwnedTaskAdmission, await_owned_task, create_owned_task, owned_task_error,
)
from kestrel_sovereign.execution_custody import (
    ExecutionCommitOutcomeError, ExecutionStreamLifecycle, _explicit_control_chain,
    execution_commit_outcome, execution_stream,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase


@pytest.mark.asyncio
@pytest.mark.parametrize("eager", [False, True] if hasattr(asyncio, "eager_task_factory") else [False])
@pytest.mark.parametrize("phase,factory_kind,first_kind", [
    (phase, factory, kind)
    for phase in ("owner-admission", "owner-completion", "lifecycle", "watcher-allocation", "watcher-admission", "watcher-completion")
    for factory, kind in [("default", None), *[("custom", item) for item in (None, "ordinary", "unknown", "committed")]]
    if phase != "owner-admission" or kind is None
])
async def test_native_constructor_failure_joins_all_original_tasks(tmp_path, factory_kind, first_kind, phase, eager):
    db = await AsyncDatabase.sqlite(str(tmp_path / "stream-effects.db"))
    await db.execute("CREATE TABLE effects(id INTEGER)")
    first = (None if first_kind is None else OSError("original gate A")
        if first_kind == "ordinary" else ExecutionCommitOutcomeError(first_kind))
    second = OSError("constructor publication B: " + phase)
    accepted, bodies = [], []
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()

    class PublicationTask(asyncio.Task):
        def add_done_callback(self, callback, *, context=None):
            name = getattr(callback, "__name__", "")
            if self.role == "run_owned":
                if name == "close_unstarted" and phase == "owner-admission":
                    raise second
                if name == "close_unstarted" and first is not None:
                    raise first
                if name == "_publish_owner_completion" and phase == "owner-completion":
                    raise second
            if self.role == "_watch_owner_cancellation":
                if (name == "close_unstarted" and phase == "watcher-admission") or (
                    name == "_interrupt_if_watch_never_started" and phase == "watcher-completion"
                ):
                    raise second
            return super().add_done_callback(callback, context=context)

    def native_factory(loop, coroutine, **options):
        operation = coroutine.cr_frame.f_locals.get("operation") if coroutine.cr_frame else None
        role = getattr(getattr(operation, "cr_code", None), "co_name", "")
        if role in {"run_owned", "_watch_owner_cancellation"}:
            if role == "_watch_owner_cancellation" and phase == "watcher-allocation":
                raise second
            task = PublicationTask(coroutine, loop=loop, **options, **({"eager_start": True} if eager else {}))
            task.role = role
            accepted.append(task)
            return task
        return asyncio.Task(coroutine, loop=loop, **options)

    def custom_factory(body, context):
        admission = OwnedTaskAdmission()
        try:
            return create_owned_task(body(), context=context, admission=admission, retain_originals=True)
        except BaseException:
            if admission.task is None:
                raise
            return admission.task

    def started(iterator, task):
        if phase == "lifecycle":
            raise second

    async def source():
        bodies.append(asyncio.current_task())
        await db.execute_commit("INSERT INTO effects VALUES (1)")
        yield b"must never dispatch"

    stream = execution_stream(None, source(), lifecycle=ExecutionStreamLifecycle(
        None if factory_kind == "default" else custom_factory, started, lambda: None))
    loop.set_task_factory(native_factory)
    reader = asyncio.create_task(anext(stream))
    try:
        done, _ = await asyncio.wait({reader}, timeout=1)
        assert reader in done, "constructor failure left its original reader blocked"
        with pytest.raises(BaseException) as caught:
            await reader
        assert accepted and all(task.done() for task in accepted), "constructor returned with an unjoined original Task"
        chain = tuple(_explicit_control_chain(caught.value))
        assert second in chain
        if first is not None:
            assert first in chain
            owner, = [task for task in accepted if task.role == "run_owned"]
            assert first in tuple(_explicit_control_chain(owned_task_error(owner)))
        assert execution_commit_outcome(caught.value) == (first_kind if first_kind in {"unknown", "committed"} else None)
        assert second.__cause__ is None and (first is None or first.__cause__ is None)
        assert not bodies and await db.fetchval("SELECT COUNT(*) FROM effects") == 0
    finally:
        loop.set_task_factory(previous)
        try:
            for task in accepted:
                if not task.done():
                    task.cancel()
                try:
                    await await_owned_task(task)
                except asyncio.CancelledError:
                    assert task.done()  # Even a broken error cache cannot skip DB cleanup.
            if not reader.done():
                reader.cancel()
            await await_owned_task(reader)
            closing = create_owned_task(stream.aclose(), retain_originals=True)
            await await_owned_task(closing)
        finally:
            await db.close()
