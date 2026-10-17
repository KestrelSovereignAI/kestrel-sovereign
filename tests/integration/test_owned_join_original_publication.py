"""A failed completion callback cannot abandon the actual native owner."""
import asyncio

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain, execution_terminal_error
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.unit.test_host_context_native_cleanup import _actual_operation
from tests.utils.original_failure import OriginalFailure


@pytest.mark.asyncio
@pytest.mark.parametrize("after", [False, True])
@pytest.mark.parametrize("worker_failure", [False, True])
@pytest.mark.parametrize("withdraw", [False, True])
@pytest.mark.parametrize("eager", [False, True] if hasattr(asyncio, "eager_task_factory") else [False])
@pytest.mark.parametrize("remove_failure", [False, True])
async def test_owned_join_publication_failure_still_joins_original_native_effect(
    native_pg, monkeypatch, after, worker_failure, withdraw, eager, remove_failure,
):
    backend, observer, schema = native_pg
    publication = await _original_control(backend, monkeypatch)
    source = OriginalFailure.capture(publication)
    physical = OSError("actual native operation observer failed after write")
    physical_source = OriginalFailure.capture(physical)
    removal = OSError("original completion callback removal failed")
    removal_source = OriginalFailure.capture(removal)
    entered, release = asyncio.Event(), asyncio.Event()
    accepted, observed = [], []
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    class PublicationTask(asyncio.Task):
        refused = False
        removal_refused = False
        def add_done_callback(self, callback, *args, **kwargs):
            if not self.refused and getattr(callback, "__name__", "") == "mark_complete":
                self.refused = True
                if after:
                    super().add_done_callback(callback, *args, **kwargs)
                raise publication
            return super().add_done_callback(callback, *args, **kwargs)
        def remove_done_callback(self, callback):
            result = super().remove_done_callback(callback)
            if remove_failure and not self.removal_refused and getattr(callback, "__name__", "") == "mark_complete":
                self.removal_refused = True
                raise removal
            return result
    async def native_effect():
        await backend.execute("INSERT INTO effects VALUES (242, 'held original native effect')")
        entered.set()
        await release.wait()
        if worker_failure:
            raise physical
    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        if getattr(getattr(operation, "cr_code", None), "co_qualname", "").endswith(".<locals>.native_effect"):
            task = PublicationTask(coroutine, loop=loop, **kwargs, **({"eager_start": True} if eager else {}))
            accepted.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    loop.set_task_factory(factory)
    native = create_owned_task(native_effect(), retain_originals=True)
    loop.set_task_factory(previous)
    async def joined():
        outcome = await await_owned_task(native)
        observed.append(outcome)
        terminal = execution_terminal_error(
            outcome.error, outcome.cancellation, retain_originals=True,
            isolate_control_originals=True,
        )
        if terminal is not None:
            raise terminal
    owner = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        owner = create_owned_task(joined(), retain_originals=True)
        done, _ = await asyncio.wait({owner}, timeout=0.05)
        assert not done, "completion publication failure abandoned an accepted original native Task"
        if withdraw:
            owner.cancel("actual completion observer withdrew")
            await asyncio.sleep(0)
            assert not owner.done()
        release.set()
        outcome = await await_owned_task(owner)
        assert native.done() and accepted == [native]
        source.assert_retained(outcome.error)
        if worker_failure:
            physical_source.assert_retained(outcome.error)
        if remove_failure:
            removal_source.assert_retained(outcome.error)
        if withdraw:
            assert len(observed) == 1 and observed[0].cancellation is not None, "canonical join discarded actual observer withdrawal"
            assert observed[0].cancellation in tuple(_explicit_control_chain(outcome.error)), "canonical join replaced actual observer withdrawal"
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=242') == 1
    finally:
        loop.set_task_factory(previous)
        release.set()
        if owner is not None:
            await await_owned_task(owner)
        await await_owned_task(native)


@pytest.mark.asyncio
async def test_borrowed_signal_withdrawal_keeps_removal_control_without_adopting_producer(native_pg, monkeypatch):
    from kestrel_sovereign.signals.dispatcher import SignalDispatchHandle

    backend, observer, schema = native_pg
    original = await _original_control(backend, monkeypatch)
    source = OriginalFailure.capture(original)
    entered, release, observed = asyncio.Event(), asyncio.Event(), []
    class RemovalTask(asyncio.Task):
        def remove_done_callback(self, callback):
            result = super().remove_done_callback(callback)
            if getattr(callback, "__name__", "") == "mark_complete":
                raise original
            return result
    async def produce():
        await backend.execute("INSERT INTO effects VALUES (244, 'original borrowed signal producer')")
        entered.set()
        await release.wait()
    native = RemovalTask(produce(), loop=asyncio.get_running_loop())
    receipt = asyncio.get_running_loop().create_future()
    # Actual SDK handle, not a replacement dispatcher/task implementation.
    handle = SignalDispatchHandle(signal_id="borrowed-original", task=native, durable_admission=receipt)
    async def borrow():
        try:
            await handle.wait()
        except BaseException as error:
            observed.append(error)
            raise
    borrower = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        borrower = create_owned_task(borrow(), retain_originals=True)
        await asyncio.sleep(0)
        borrower.cancel("actual borrowed signal observer withdrew")
        done, _ = await asyncio.wait({borrower}, timeout=0.1)
        assert done, "borrowed signal cancellation adopted the producer lifetime"
        outcome = await await_owned_task(borrower)
        source.assert_retained(outcome.error)
        assert observed and any(isinstance(error, asyncio.CancelledError) and error.args == ("actual borrowed signal observer withdrew",)
                                for error in _explicit_control_chain(observed[0])), "borrowed completion lost actual caller withdrawal"
        assert not native.done() and not native.cancelling()
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=244') == 1
    finally:
        release.set()
        if borrower is not None:
            await await_owned_task(borrower)
        await await_owned_task(native)


@pytest.mark.asyncio
@pytest.mark.parametrize("after", [False, True])
async def test_borrowed_completion_publication_failure_does_not_adopt_native_owner(native_pg, monkeypatch, after):
    backend, observer, schema = native_pg
    publication = await _original_control(backend, monkeypatch)
    source = OriginalFailure.capture(publication)
    entered, release = asyncio.Event(), asyncio.Event()
    class PublicationTask(asyncio.Task):
        refused = False
        def add_done_callback(self, callback, *args, **kwargs):
            if not self.refused and getattr(callback, "__name__", "") == "mark_complete":
                self.refused = True
                if after:
                    super().add_done_callback(callback, *args, **kwargs)
                raise publication
            return super().add_done_callback(callback, *args, **kwargs)
    async def native_effect():
        await backend.execute("INSERT INTO effects VALUES (243, 'borrowed original native effect')")
        entered.set()
        await release.wait()
    native = PublicationTask(native_effect(), loop=asyncio.get_running_loop())
    borrower = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        borrower = create_owned_task(await_owned_task(native, defer_cancellation=False), retain_originals=True)
        done, _ = await asyncio.wait({borrower}, timeout=0.05)
        assert done, "borrowed publication failure adopted an unrelated original native lifetime"
        outcome = await await_owned_task(borrower)
        source.assert_retained(outcome.error)
        assert not native.done() and not native.cancelling()
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects WHERE id=243') == 1
    finally:
        release.set()
        if borrower is not None:
            await await_owned_task(borrower)
        await await_owned_task(native)
