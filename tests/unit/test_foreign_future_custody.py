"""The asyncio Future-like protocol does not require writable private metadata."""

from __future__ import annotations

import asyncio
import gc
import inspect

import pytest

from kestrel_sovereign._async_ownership import (
    OwnedTaskAdmission,
    await_owned_operation,
    await_owned_task,
    create_owned_task,
    owned_task_error,
    owned_task_join_ancestors,
    owned_task_join_dependency,
)
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    ExecutionCommitOutcomeError,
    execution_commit_outcome,
)
from tests.utils.original_failure import OriginalFailure


class SlotFutureLike:
    """Read-only protocol facade over an ACTUAL asyncio Future/Task."""

    __slots__ = ("_future",)
    _asyncio_future_blocking = False

    def __init__(self, future):
        self._future = future

    def __await__(self):
        return self._future.__await__()

    def __getattr__(self, name):
        return getattr(self._future, name)


class ReadOnlyNativeFuture(asyncio.Future):
    def __setattr__(self, name, value):
        if name.startswith("_kestrel_"):
            raise AttributeError("native Future subclass does not grant private writes")
        return super().__setattr__(name, value)


def _protocol_source(loop, kind):
    source = loop.create_future() if kind == "slots_only" else ReadOnlyNativeFuture(loop=loop)
    return source, SlotFutureLike(source) if kind == "slots_only" else source


@pytest.mark.asyncio
@pytest.mark.parametrize("future_kind", ["slots_only", "native_readonly"])
@pytest.mark.parametrize("completed", [False, True])
@pytest.mark.parametrize("kind", ["receipt", "error", "denied", "unknown", "cancelled"])
async def test_slots_only_future_protocol_needs_no_mutation_or_task_allocation(kind, future_kind, completed):
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    source, foreign = _protocol_source(loop, future_kind)
    assert asyncio.isfuture(foreign)
    assert asyncio.ensure_future(foreign) is foreign
    original = {
        "error": ValueError("original foreign producer failure"),
        "denied": ExecutionAuthorityError("original producer native denial"),
        "unknown": ExecutionCommitOutcomeError("unknown"),
    }.get(kind)
    refused = []

    def complete():
        if kind == "receipt":
            source.set_result("exact foreign receipt")
        elif kind == "cancelled":
            source.cancel("exact foreign producer cancellation")
        else:
            source.set_exception(original)

    def refuse(_loop, _coroutine, **_kwargs):
        refused.append("surrogate Task allocation attempted")
        raise RuntimeError("original loop refuses new Tasks")

    if completed:
        complete()
    else:
        loop.call_soon(complete)
    outcome = None
    try:
        loop.set_task_factory(refuse)
        outcome = await await_owned_task(foreign)
    finally:
        loop.set_task_factory(previous_factory)
        if outcome is None:
            # A rejected guard mutant still leaves test-owned source custody.
            await asyncio.sleep(0)
            if source.done() and not source.cancelled():
                source.exception()
    assert refused == []
    assert outcome.cancellation is None
    if kind == "receipt":
        assert outcome.result == "exact foreign receipt" and outcome.error is None
    elif kind == "cancelled":
        assert isinstance(outcome.error, asyncio.CancelledError)
        assert outcome.error.args == ("exact foreign producer cancellation",)
    else:
        assert outcome.error is original
        if kind == "unknown":
            assert execution_commit_outcome(outcome.error) == "unknown"
    assert not hasattr(source, "_kestrel_owned_terminal_error"), "foreign protocol does not grant metadata mutation"


@pytest.mark.asyncio
@pytest.mark.parametrize("future_kind", ["slots_only", "native_readonly"])
@pytest.mark.parametrize("failure", [False, True])
async def test_slots_only_future_retains_its_join_through_repeated_caller_cancellation(failure, future_kind):
    loop = asyncio.get_running_loop()
    source, foreign = _protocol_source(loop, future_kind)
    original = LookupError("original foreign effect tail failed")
    owner = asyncio.create_task(await_owned_task(foreign))
    previous_factory = loop.get_task_factory()
    refused = []

    def refuse(_loop, _coroutine, **_kwargs):
        refused.append("surrogate Task allocation attempted")
        raise RuntimeError("loop refuses new Tasks")

    try:
        loop.set_task_factory(refuse)
        for _ in range(4):
            await asyncio.sleep(0)
            owner.cancel("first original foreign observer cancellation")
            assert not owner.done() and not source.done()
        if failure:
            source.set_exception(original)
        else:
            source.set_result("original completed tail receipt")
        outcome = await owner
    finally:
        loop.set_task_factory(previous_factory)
        if not source.done():
            source.set_result("test finalizer released original")
        await await_owned_task(owner)
    assert refused == []
    assert not source.cancelled()
    assert outcome.cancellation.args == ("first original foreign observer cancellation",)
    assert outcome.error is (original if failure else None)
    if not failure:
        assert outcome.result == "original completed tail receipt"


@pytest.mark.asyncio
async def test_readonly_foreign_facade_reuses_producer_retained_native_original():
    original = asyncio.CancelledError("original Core-owned producer cancellation")
    cause = OSError("original native close failure")

    async def producer():
        raise original from cause

    source = create_owned_task(producer(), retain_originals=True)
    foreign = SlotFutureLike(source)
    try:
        first = await await_owned_task(foreign)
        second = await await_owned_task(foreign)
    finally:
        await await_owned_task(source)
    assert first.error is second.error is original
    assert original.__cause__ is cause
    assert owned_task_error(source) is original


@pytest.mark.asyncio
@pytest.mark.parametrize("eager", [False, True] if hasattr(asyncio, "eager_task_factory") else [False])
async def test_core_owned_task_subclass_keeps_exact_admission_edges_and_original_error(eager):
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    release = asyncio.Event()
    original = asyncio.CancelledError("actual custom Core-owned source cancellation")
    cause = OSError("actual source cause")

    class ProducerTask(asyncio.Task):
        pass

    def factory(loop, coroutine, **kwargs):
        options = {"eager_start": True} if eager else {}
        return ProducerTask(coroutine, loop=loop, **kwargs, **options)

    async def producer():
        await release.wait()
        raise original from cause

    source = None
    try:
        loop.set_task_factory(factory)
        source = create_owned_task(producer(), retain_originals=True)
        loop.set_task_factory(previous_factory)
        assert type(source) is ProducerTask
        assert source._kestrel_owned_admission.task is source
        with owned_task_join_dependency(source):
            assert owned_task_join_ancestors(source) == {asyncio.current_task()}
        assert not source._kestrel_owned_join_waiters
        foreign = SlotFutureLike(source)
        assert foreign._kestrel_owned_admission is source._kestrel_owned_admission
        with owned_task_join_dependency(foreign):
            assert not source._kestrel_owned_join_waiters, "forwarded receipt is not task-bound to this facade"
        release.set()
        first = await await_owned_task(source)
        second = await await_owned_task(foreign)
        assert first.error is second.error is original
        assert original.__cause__ is cause
    finally:
        loop.set_task_factory(previous_factory)
        release.set()
        if source is not None:
            await await_owned_task(source)


@pytest.mark.asyncio
@pytest.mark.parametrize("eager", [False, True] if hasattr(asyncio, "eager_task_factory") else [False])
@pytest.mark.parametrize("mode", ["before", "after", "all_private"])
async def test_metadata_receipt_publication_failure_joins_actual_accepted_task_without_running_payload(mode, eager):
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    opening = ExecutionCommitOutcomeError("unknown")
    cache_failure = OSError("actual metadata terminal publication refused")
    opening_source, cache_source = OriginalFailure.capture(opening), OriginalFailure.capture(cache_failure)
    accepted, body = [], []

    class RefusingTask(asyncio.Task):
        def __setattr__(self, name, value):
            if name == "_kestrel_owned_admission":
                if mode == "after":
                    super().__setattr__(name, value)
                raise opening
            if mode == "all_private" and name.startswith("_kestrel_"):
                raise cache_failure
            return super().__setattr__(name, value)

    def factory(loop, coroutine, **kwargs):
        options = {"eager_start": True} if eager else {}
        task = RefusingTask(coroutine, loop=loop, **kwargs, **options)
        accepted.append(task)
        return task

    async def payload():
        body.append("paid effect must never start after failed publication")

    operation = payload()
    outcome = None
    try:
        loop.set_task_factory(factory)
        outcome = await await_owned_operation(operation, retain_originals=True)
    finally:
        loop.set_task_factory(previous_factory)
        for task in accepted:
            await await_owned_task(task)
    assert len(accepted) == 1 and accepted[0].done()
    assert body == [] and inspect.getcoroutinestate(operation) == inspect.CORO_CLOSED
    assert execution_commit_outcome(outcome.error) == "unknown"
    opening_source.assert_retained(outcome.error)
    if mode == "all_private":
        cache_source.assert_retained(outcome.error)


@pytest.mark.asyncio
@pytest.mark.parametrize("eager", [False, True] if hasattr(asyncio, "eager_task_factory") else [False])
@pytest.mark.parametrize("mode", ["marker_before", "marker_after", "all_private", "callback_before", "callback_after"])
@pytest.mark.parametrize("close_fails", [False, True])
@pytest.mark.parametrize("source_kind", ["unknown", "cancelled"])
@pytest.mark.parametrize("explicit_admission", [False, True])
async def test_creator_preserves_exact_failed_task_when_postaccept_publication_is_refused(mode, eager, close_fails, source_kind, explicit_admission):
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    opening = (ExecutionCommitOutcomeError("unknown") if source_kind == "unknown"
               else asyncio.CancelledError("original postaccept source cancellation"))
    cache_error = OSError("accepted Task refused terminal cache")
    close_error = OSError("original payload refused unstarted close")
    opening_source, cache_source = OriginalFailure.capture(opening), OriginalFailure.capture(cache_error)
    close_source = OriginalFailure.capture(close_error)
    accepted, body, closed, diagnostics = [], [], [], []
    previous_handler = loop.get_exception_handler()

    class RefusingTask(asyncio.Task):
        def __setattr__(self, name, value):
            if not mode.startswith("callback") and name == "_kestrel_owned_admission":
                if mode == "marker_after":
                    super().__setattr__(name, value)
                raise opening
            if mode == "all_private" and name.startswith("_kestrel_"):
                raise cache_error
            return super().__setattr__(name, value)

        def add_done_callback(self, callback, *, context=None):
            if mode.startswith("callback"):
                if mode == "callback_after":
                    super().add_done_callback(callback, context=context)
                raise opening
            return super().add_done_callback(callback, context=context)

    def factory(loop, coroutine, **kwargs):
        options = {"eager_start": True} if eager else {}
        task = RefusingTask(coroutine, loop=loop, **kwargs, **options)
        accepted.append(task)
        return task

    async def payload():
        body.append("refused paid effect must never start")

    class CloseFailingPayload:
        def __await__(self):
            return payload().__await__()

        def close(self):
            closed.append("exact original close")
            raise close_error

    operation = CloseFailingPayload() if close_fails else payload()
    outcome = None
    try:
        loop.set_exception_handler(lambda _loop, context: diagnostics.append(context))
        loop.set_task_factory(factory)
        if explicit_admission:
            admission = OwnedTaskAdmission()
            with pytest.raises(BaseException) as caught:
                create_owned_task(operation, retain_originals=True, admission=admission)
            task = admission.task
            opening_source.assert_retained(caught.value)
        else:
            task = create_owned_task(operation, retain_originals=True)
        assert task is accepted[0], "default caller must receive its exact accepted Task"
        outcome = await await_owned_task(task)
    finally:
        loop.set_task_factory(previous_factory)
        for task in accepted:
            await await_owned_task(task)
        loop.set_exception_handler(previous_handler)
    assert len(accepted) == 1 and accepted[0].done()
    assert body == []
    if close_fails:
        assert closed == ["exact original close"], "refused close must not be retried by a late callback"
        close_source.assert_retained(outcome.error)
    else:
        assert inspect.getcoroutinestate(operation) == inspect.CORO_CLOSED
    opening_source.assert_retained(outcome.error)
    if source_kind == "unknown":
        assert execution_commit_outcome(outcome.error) == "unknown"
    if mode == "all_private":
        cache_source.assert_retained(outcome.error)
    loop.set_exception_handler(lambda _loop, context: diagnostics.append(context))
    try:
        accepted.clear()
        task = None
        outcome = None
        gc.collect()
        await asyncio.sleep(0)
        assert diagnostics == []
    finally:
        loop.set_exception_handler(previous_handler)


@pytest.mark.asyncio
@pytest.mark.parametrize("eager", [False, True] if hasattr(asyncio, "eager_task_factory") else [False])
@pytest.mark.parametrize("source_kind", ["unknown", "cancelled"])
async def test_native_task_cache_contract_failure_does_not_mask_actual_payload_outcome(eager, source_kind):
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    original = (ExecutionCommitOutcomeError("unknown") if source_kind == "unknown"
                else asyncio.CancelledError("actual payload source cancellation"))
    publication = OSError("native producer refused promised terminal metadata")
    source, publication_source = OriginalFailure.capture(original), OriginalFailure.capture(publication)
    accepted, diagnostics = [], []
    previous_handler = loop.get_exception_handler()

    class RefusingCacheTask(asyncio.Task):
        def __setattr__(self, name, value):
            if name == "_kestrel_owned_terminal_error":
                raise publication
            return super().__setattr__(name, value)

    def factory(loop, coroutine, **kwargs):
        options = {"eager_start": True} if eager else {}
        task = RefusingCacheTask(coroutine, loop=loop, **kwargs, **options)
        accepted.append(task)
        return task

    async def payload():
        raise original

    try:
        loop.set_exception_handler(lambda _loop, context: diagnostics.append(context))
        loop.set_task_factory(factory)
        task = create_owned_task(payload(), retain_originals=True)
        first = await await_owned_task(task)
        source.assert_retained(first.error)
        publication_source.assert_retained(first.error)
        # The actual Task outcome, not only a finite harvest envelope, must
        # retain the source when its own cache publication was refused.
        physical = task.exception()
        source.assert_retained(physical)
        publication_source.assert_retained(physical)
        second = await await_owned_task(task)
        source.assert_retained(second.error)
        publication_source.assert_retained(second.error)
        if source_kind == "unknown":
            assert execution_commit_outcome(second.error) == "unknown"
        task = None
        accepted.clear()
        gc.collect()
        await asyncio.sleep(0)
        assert diagnostics == []
    finally:
        loop.set_task_factory(previous_factory)
        for task in accepted:
            await await_owned_task(task)
        loop.set_exception_handler(previous_handler)


@pytest.mark.asyncio
async def test_unstarted_close_error_cannot_strand_an_accepted_metadata_refusal_gate():
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    opening = ExecutionCommitOutcomeError("unknown")
    close_error = OSError("original unstarted effect close failed")
    opening_source = OriginalFailure.capture(opening)
    close_source = OriginalFailure.capture(close_error)
    accepted, closed, body = [], [], []

    class RefusingTask(asyncio.Task):
        def __setattr__(self, name, value):
            if name == "_kestrel_owned_admission":
                raise opening
            return super().__setattr__(name, value)

    def factory(loop, coroutine, **kwargs):
        task = RefusingTask(coroutine, loop=loop, **kwargs)
        accepted.append(task)
        return task

    class Effect:
        def __await__(self):
            body.append("refused payload must not start")
            return iter(())

        def close(self):
            closed.append("exact original close")
            raise close_error

    owner = asyncio.create_task(await_owned_operation(Effect(), retain_originals=True))
    outcome = None
    try:
        loop.set_task_factory(factory)
        # No I/O or timer is involved: allow every actual admitted Task/ACK
        # callback to settle. A failed mutant is still joined in the finalizer.
        for _ in range(20):
            await asyncio.sleep(0)
            if owner.done():
                break
        assert owner.done(), "failed original close stranded an accepted Task behind its gate"
        outcome = await owner
    finally:
        loop.set_task_factory(previous_factory)
        for task in accepted:
            if not task.done():
                task.cancel("test finalizer joins stranded guard mutant")
            await await_owned_task(task)
        await await_owned_task(owner)
    assert len(accepted) == 1 and accepted[0].done()
    assert closed == ["exact original close"] and body == []
    assert execution_commit_outcome(outcome.error) == "unknown"
    opening_source.assert_retained(outcome.error)
    close_source.assert_retained(outcome.error)


@pytest.mark.asyncio
async def test_native_task_still_retains_first_cancelled_original_across_owners():
    original = asyncio.CancelledError("native original before any retained metadata")

    async def producer():
        raise original

    source = asyncio.create_task(producer())
    first = await await_owned_task(source)
    second = await await_owned_task(source)
    assert first.error is second.error is original
    assert owned_task_error(source) is original
