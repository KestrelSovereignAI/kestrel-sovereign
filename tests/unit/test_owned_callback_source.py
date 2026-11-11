"""Original callback Task/context, one-item handoff and actual FD retirement."""

import asyncio
import contextvars
import errno
import os
from contextlib import AsyncExitStack
from pathlib import Path

import pytest
import anyio

from kestrel_sovereign._async_ownership import (
    OwnedCallbackScope,
    OwnedCallbackSource,
    await_owned_task,
    create_owned_iterator,
    create_owned_task,
    raise_owned_outcome,
)
from kestrel_sovereign.execution_custody import (
    bind_native_execution_admission,
    capture_native_execution_admission,
)
from kestrel_sovereign.storage.privacy_wrapper import ReentrantTransitionLock


@pytest.mark.asyncio
@pytest.mark.parametrize("fresh_context", [False, True])
@pytest.mark.parametrize("resource_admission", [False, True])
@pytest.mark.parametrize("fail_close", [False, True])
async def test_callback_context_is_separate_from_explicit_resource_admission(
    tmp_path: Path, fresh_context: bool, resource_admission: bool, fail_close: bool,
):
    marker = contextvars.ContextVar("sender-private-callback", default="fresh")
    sender_token = marker.set("sender-private")
    coordinator = ReentrantTransitionLock() if resource_admission else None
    closing, release = asyncio.Event(), asyncio.Event()
    descriptors, closed, owners, retired, errors = [], [], [], [], []
    expected_marker = "fresh" if fresh_context else "sender-private"

    async def sender_validator():
        raise AssertionError("A callback must not execute the sender's validator")

    async def original():
        assert marker.get() == expected_marker
        assert capture_native_execution_admission() == (() if fresh_context else (sender_validator,))
        original_token = marker.set("original-callback")
        try:
            async with AsyncExitStack() as resources:
                if coordinator is not None:
                    await resources.enter_async_context(coordinator)
                    assert coordinator._writer is asyncio.current_task()
                descriptor = os.open(
                    tmp_path / "original-context-admitted-resource",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600,
                )
                descriptors.append(descriptor)
                owners.append(asyncio.current_task())
                try:
                    os.write(descriptor, b"one original admitted resource effect")
                    closing.set()
                    await release.wait()
                finally:
                    os.close(descriptor)
                    closed.append(descriptor)
                    retired.append(asyncio.current_task())
                    if fail_close:
                        try:
                            os.fsync(descriptor)
                        except OSError as error:
                            errors.append(error)
                            raise
        finally:
            marker.reset(original_token)
            assert marker.get() == expected_marker

    child = None
    try:
        with bind_native_execution_admission(sender_validator):
            async with AsyncExitStack() as resources:
                if coordinator is not None:
                    await resources.enter_async_context(coordinator)
                scope = OwnedCallbackScope()
                child = scope.start(
                    original, context=contextvars.Context() if fresh_context else None,
                    name="original-context-admitted-callback",
                )
                observed_close = create_owned_task(closing.wait(), retain_originals=True)
                try:
                    async with scope.wait():
                        completed, _ = await asyncio.wait(
                            {child, observed_close}, return_when=asyncio.FIRST_COMPLETED,
                        )
                        if child in completed:
                            raise_owned_outcome(
                                await await_owned_task(child), operation="callback-before-resource-close",
                                retain_originals=True,
                            )
                        assert closing.is_set()
                        assert not child.done() and closed == []
                        os.fstat(descriptors[0])
                        assert marker.get() == "sender-private"
                        assert capture_native_execution_admission() == (sender_validator,)
                        release.set()
                        outcome = await await_owned_task(child)
                    if coordinator is not None:
                        assert coordinator._writer is asyncio.current_task()
                    if fail_close:
                        assert len(errors) == 1 and outcome.error is errors[0]
                        assert errors[0].errno == errno.EBADF
                    else:
                        raise_owned_outcome(outcome, operation="explicit-resource-callback", retain_originals=True)
                    assert closed == descriptors and len(closed) == 1
                    assert owners == retired == [child] and child.done()
                    assert (tmp_path / "original-context-admitted-resource").read_bytes() == b"one original admitted resource effect"
                    with pytest.raises(OSError) as captured:
                        os.fstat(descriptors[0])
                    assert captured.value.errno == errno.EBADF
                finally:
                    release.set()
                    if not observed_close.done():
                        observed_close.cancel()
                    await await_owned_task(observed_close)
                    if child is not None:
                        async with scope.wait():
                            await await_owned_task(child)
                    scope.close()
        assert capture_native_execution_admission() == ()
    finally:
        marker.reset(sender_token)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_close", [False, True])
async def test_fresh_callback_iterator_retains_physical_close_and_privacy_writer(
    tmp_path: Path, fail_close: bool,
):
    coordinator = ReentrantTransitionLock()
    marker = contextvars.ContextVar("fresh-stream-sender", default="fresh")
    sender_token = marker.set("sender")
    closing, release, outsider_entered = asyncio.Event(), asyncio.Event(), asyncio.Event()
    descriptors, closed, owners, retired, errors = [], [], [], [], []
    source = close_owner = outsider = None

    async def sender_validator():
        raise AssertionError("Fresh application source inherited sender validators")

    async def produce(emit):
        assert marker.get() == "fresh" and capture_native_execution_admission() == ()
        token = marker.set("original-application")
        try:
            async with coordinator:
                descriptor = os.open(tmp_path / "fresh-application-fd",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                descriptors.append(descriptor)
                owners.append(asyncio.current_task())
                try:
                    os.write(descriptor, b"first original frame")
                    await emit(b"first original frame")
                finally:
                    closing.set()
                    await release.wait()
                    os.close(descriptor)
                    closed.append(descriptor)
                    retired.append(asyncio.current_task())
                    if fail_close:
                        try:
                            os.fsync(descriptor)
                        except OSError as error:
                            errors.append(error)
                            raise
        finally:
            marker.reset(token)
            assert marker.get() == "fresh"

    async def unrelated_writer():
        async with coordinator:
            outsider_entered.set()
            (tmp_path / "later-unrelated-writer").write_bytes(b"after original policy span")

    try:
        with bind_native_execution_admission(sender_validator):
            async with coordinator:
                scope = OwnedCallbackScope()

                def start_original(operation, context):
                    return scope.start(operation, context=context, name="fresh-original-application")

                source = await create_owned_iterator(
                    lambda: OwnedCallbackSource(produce), operation="fresh-application-frames",
                    owner_context=contextvars.Context(), owner_factory=start_original,
                )
                try:
                    assert await anext(source) == b"first original frame"
                    assert coordinator._writer is asyncio.current_task()
                    assert marker.get() == "sender"
                    assert capture_native_execution_admission() == (sender_validator,)
                    outsider = asyncio.create_task(unrelated_writer(), context=contextvars.Context())
                    async with scope.wait():
                        close_owner = create_owned_task(source.aclose(), retain_originals=True)
                        observed_close = create_owned_task(closing.wait(), retain_originals=True)
                        try:
                            completed, _ = await asyncio.wait(
                                {close_owner, observed_close}, return_when=asyncio.FIRST_COMPLETED)
                            if close_owner in completed:
                                raise_owned_outcome(await await_owned_task(close_owner),
                                    operation="fresh-application-before-close", retain_originals=True)
                            assert closing.is_set() and not close_owner.done()
                            assert coordinator._writer is source.producer_task
                            assert closed == [] and not outsider_entered.is_set()
                            os.fstat(descriptors[0])
                            release.set()
                            outcome = await await_owned_task(close_owner)
                        finally:
                            release.set()
                            if not observed_close.done():
                                observed_close.cancel()
                            await await_owned_task(observed_close)
                    if fail_close:
                        assert len(errors) == 1 and outcome.error is errors[0]
                        assert errors[0].errno == errno.EBADF
                    else:
                        raise_owned_outcome(outcome, operation="fresh-application-close", retain_originals=True)
                    assert source.producer_task.done() and source.owner_task.done()
                    assert owners == retired == [source.producer_task]
                    assert descriptors == closed and len(closed) == 1
                    assert coordinator._writer is asyncio.current_task()
                    assert not outsider_entered.is_set()
                finally:
                    release.set()
                    async with scope.wait():
                        if close_owner is not None:
                            await await_owned_task(close_owner)
                        elif source is not None:
                            await source.aclose()
                    scope.close()
            await outsider
            assert outsider_entered.is_set()
        assert (tmp_path / "later-unrelated-writer").read_bytes() == b"after original policy span"
        with pytest.raises(OSError) as captured:
            os.fstat(descriptors[0])
        assert captured.value.errno == errno.EBADF
    finally:
        release.set()
        if outsider is not None:
            if not outsider.done():
                outsider.cancel()
            await asyncio.gather(outsider, return_exceptions=True)
        marker.reset(sender_token)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_close", [False, True])
async def test_callback_close_cannot_abort_an_already_running_original_finally(
    tmp_path: Path, fail_close: bool,
):
    closing, release = asyncio.Event(), asyncio.Event()
    descriptors, closed, owners, retired, errors = [], [], [], [], []
    marker = contextvars.ContextVar("already-running-callback-finally", default="caller")

    async def produce(emit):
        descriptor = os.open(tmp_path / "already-running-original-finally",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        descriptors.append(descriptor)
        owners.append(asyncio.current_task())
        token = marker.set("original-callback")
        try:
            try:
                os.write(descriptor, b"original physical effect")
                await emit("first")
            finally:
                closing.set()
                await release.wait()
                os.close(descriptor)
                closed.append(descriptor)
                retired.append(asyncio.current_task())
                if fail_close:
                    try:
                        os.fsync(descriptor)
                    except OSError as error:
                        errors.append(error)
                        raise
        finally:
            marker.reset(token)

    source = await create_owned_iterator(
        lambda: OwnedCallbackSource(produce), operation="already-running-callback-finally")
    reader = closer = None
    try:
        assert await anext(source) == "first"
        reader = create_owned_task(anext(source), retain_originals=True)
        await closing.wait()
        closer = create_owned_task(source.aclose(), retain_originals=True)
        completed, _ = await asyncio.wait({closer}, timeout=.01)
        assert not completed, "Close acknowledged after aborting the original active finally"
        assert not source.producer_task.done() and closed == []
        os.fstat(descriptors[0])
        release.set()
        close_outcome = await await_owned_task(closer)
        reader_outcome = await await_owned_task(reader)
        if fail_close:
            assert len(errors) == 1 and errors[0].errno == errno.EBADF
            assert close_outcome.error is errors[0]
            assert reader_outcome.error is errors[0]
        else:
            raise_owned_outcome(close_outcome, operation="active-finally-close", retain_originals=True)
            assert isinstance(reader_outcome.error, StopAsyncIteration)
        assert owners == retired == [source.producer_task]
        assert descriptors == closed and len(closed) == 1
        assert source.producer_task.done() and source.owner_task.done()
        assert marker.get() == "caller"
        with pytest.raises(OSError) as captured:
            os.fstat(descriptors[0])
        assert captured.value.errno == errno.EBADF
    finally:
        release.set()
        if closer is not None:
            await await_owned_task(closer)
        else:
            await source.aclose()
        if reader is not None:
            await await_owned_task(reader)
        # Failure-only fixture recovery is not an original retirement receipt.
        for descriptor in descriptors:
            if descriptor not in closed:
                os.close(descriptor)


@pytest.mark.asyncio
async def test_concurrent_callback_emitters_do_not_hold_writer_while_waiting_for_handoff(
    tmp_path: Path,
):
    coordinator = ReentrantTransitionLock()
    first_entered, second_entered = asyncio.Event(), asyncio.Event()
    children, descriptors, closed, frames = [], {}, set(), []

    async def produce(emit):
        async def child(name):
            if name == "second":
                await first_entered.wait()
            async with coordinator:
                descriptor = os.open(tmp_path / name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                descriptors[name] = descriptor
                try:
                    os.write(descriptor, name.encode())
                    (first_entered if name == "first" else second_entered).set()
                    await emit(name)
                finally:
                    os.close(descriptor)
                    closed.add(name)

        children.extend(create_owned_task(child(name), retain_originals=True)
                        for name in ("first", "second"))
        for original in children:
            raise_owned_outcome(await await_owned_task(original),
                operation="original concurrent callback", retain_originals=True)

    async def parent():
        async with coordinator:
            scope = OwnedCallbackScope()

            def start_original(operation, context):
                return scope.start(operation, context=context)

            source = await create_owned_iterator(
                lambda: OwnedCallbackSource(produce), operation="concurrent privacy callbacks",
                owner_context=contextvars.Context(), owner_factory=start_original)
            try:
                async with scope.wait():
                    # Both original children enter their resource writer before
                    # the consumer acknowledges the first delivered frame.
                    await second_entered.wait()
                    frames.append(await anext(source))
                    frames.append(await anext(source))
                    with pytest.raises(StopAsyncIteration):
                        await anext(source)
                assert coordinator._writer is asyncio.current_task()
                assert source.producer_task.done() and source.owner_task.done()
            finally:
                async with scope.wait():
                    await source.aclose()
                scope.close()

    original_parent = create_owned_task(parent(), retain_originals=True)
    try:
        completed, _ = await asyncio.wait({original_parent}, timeout=.5)
        assert completed, "An emitter held its privacy writer while waiting for another handoff"
        raise_owned_outcome(await await_owned_task(original_parent),
            operation="concurrent callback parent", retain_originals=True)
        assert frames == ["first", "second"]
        assert closed == {"first", "second"} and all(task.done() for task in children)
        for descriptor in descriptors.values():
            with pytest.raises(OSError) as captured:
                os.fstat(descriptor)
            assert captured.value.errno == errno.EBADF
    finally:
        # Only failure recovery interrupts the exact test-owned children to
        # break a reproduced writer/mutex cycle. It is never the passing ACK.
        if not original_parent.done():
            for original in children:
                if not original.done():
                    original.cancel()
        await await_owned_task(original_parent)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancellation", ["task", "scope"])
@pytest.mark.parametrize("ending", ["advance", "stop"])
@pytest.mark.parametrize("repeated", [False, True])
async def test_cancelled_emitter_keeps_its_original_handoff_until_its_ack_or_stop(
    tmp_path: Path, cancellation: str, ending: str, repeated: bool,
):
    second_ready, second_returned = asyncio.Event(), asyncio.Event()
    children, scopes, descriptors, retired, owners, closed, controls, results = [], {}, {}, {}, {}, set(), [], {}
    withdrawal = object()
    later_withdrawal = object()

    async def produce(emit):
        async def child(name):
            descriptor = os.open(tmp_path / name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            descriptors[name] = descriptor
            owners[name] = asyncio.current_task()
            with anyio.CancelScope() as scope:
                scopes[name] = scope
                try:
                    os.write(descriptor, name.encode())
                    if name == "second":
                        second_ready.set()
                    results[name] = await emit(name)
                    if name == "second":
                        second_returned.set()
                except asyncio.CancelledError as error:
                    controls.append(error)
                    raise
                finally:
                    os.close(descriptor)
                    closed.add(name)
                    retired[name] = asyncio.current_task()

        async with asyncio.TaskGroup() as tasks:
            children.append(tasks.create_task(child("first")))
            children.append(tasks.create_task(child("second")))

    source = await create_owned_iterator(
        lambda: OwnedCallbackSource(produce), operation="cancelled original emitter")
    try:
        assert await anext(source) == "first"
        await second_ready.wait()
        if cancellation == "task":
            children[0].cancel(withdrawal)
        else:
            scopes["first"].cancel()
        completed, _ = await asyncio.wait({children[0]}, timeout=.01)
        assert not completed, "Emitter abandoned its outstanding frame before ACK or Stop"
        if repeated:
            if cancellation == "task":
                children[0].cancel(later_withdrawal)
            else:
                scopes["first"].cancel()
            completed, _ = await asyncio.wait({children[0]}, timeout=.01)
            assert not completed, "Repeated cancellation abandoned the outstanding frame"
        assert closed == set() and controls == []
        os.fstat(descriptors["first"])
        os.fstat(descriptors["second"])
        if ending == "advance":
            assert await anext(source) == "second"
            completed, _ = await asyncio.wait({children[1]}, timeout=.01)
            assert not completed and not second_returned.is_set(), "Second emitter consumed first frame's ACK"
            os.fstat(descriptors["second"])
            with pytest.raises(StopAsyncIteration):
                await anext(source)
            assert results == {"second": True}
        else:
            await source.aclose()
            assert results == {"second": False}
        assert closed == {"first", "second"} and owners == retired
        assert len(controls) == 1 and isinstance(controls[0], asyncio.CancelledError)
        if cancellation == "task":
            assert len(controls[0].args) == 1 and controls[0].args[0] is withdrawal
            if repeated:
                from kestrel_sovereign.execution_custody import _explicit_control_chain
                originals = tuple(_explicit_control_chain(controls[0]))
                for marker in (withdrawal, later_withdrawal):
                    assert any(error is not controls[0] and isinstance(error, asyncio.CancelledError)
                               and len(error.args) == 1 and error.args[0] is marker
                               for error in originals)
        assert all(task.done() for task in children)
        assert source.producer_task.done() and source.owner_task.done()
        for descriptor in descriptors.values():
            with pytest.raises(OSError) as captured:
                os.fstat(descriptor)
            assert captured.value.errno == errno.EBADF
    finally:
        await source.aclose()


@pytest.mark.asyncio
async def test_callback_runs_and_resets_in_one_original_context():
    context = contextvars.ContextVar("original-callback-context", default="empty")
    consumer_token = context.set("consumer")
    tasks, continued = [], []
    retired = asyncio.Event()

    async def produce(emit):
        assert context.get() == "empty"
        token = context.set("producer")
        try:
            for item in ("first", "second"):
                tasks.append(asyncio.current_task())
                if not await emit((item, context.get())):
                    return
                continued.append(item)
        finally:
            context.reset(token)
            retired.set()

    source = await create_owned_iterator(
        lambda: OwnedCallbackSource(produce), operation="original-callback",
        owner_context=contextvars.Context(),
    )
    original = source.producer_task
    try:
        assert await anext(source) == ("first", "producer")
        assert continued == [] and not retired.is_set()
        assert context.get() == "consumer"
        assert await anext(source) == ("second", "producer")
        assert continued == ["first"] and not retired.is_set()
        assert tasks == [original, original]
        with pytest.raises(StopAsyncIteration):
            await anext(source)
        assert retired.is_set() and original.done() and source.owner_task.done()
        assert continued == ["first", "second"]
        assert context.get() == "consumer"
    finally:
        await source.aclose()
        context.reset(consumer_token)


@pytest.mark.asyncio
async def test_callback_preserves_actual_original_filesystem_failure(tmp_path: Path):
    errors = []

    async def produce(emit):
        if not await emit("first"):
            return
        try:
            (tmp_path / "missing-original-source").read_bytes()
        except FileNotFoundError as error:
            errors.append(error)
            raise

    source = await create_owned_iterator(
        lambda: OwnedCallbackSource(produce), operation="original-callback-error",
    )
    try:
        assert await anext(source) == "first"
        with pytest.raises(FileNotFoundError) as captured:
            await anext(source)
        assert errors == [captured.value]
        assert captured.value.filename == str(tmp_path / "missing-original-source")
        assert source.producer_task.done() and source.owner_task.done()
        frames = []
        traceback = captured.value.__traceback__
        while traceback is not None:
            frames.append(traceback.tb_frame.f_code.co_name)
            traceback = traceback.tb_next
        assert "produce" in frames
    finally:
        await source.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_close", [False, True])
async def test_callback_close_joins_the_original_fd_and_late_error(
    tmp_path: Path, fail_close: bool,
):
    opened, closing, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    descriptors, owners, errors = [], [], []
    original_path = tmp_path / "original-callback-fd"

    async def produce(emit):
        original_fd = os.open(original_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        descriptors.append(original_fd)
        owners.append(asyncio.current_task())
        opened.set()
        try:
            os.write(original_fd, b"original source effect")
            await emit("first")
        finally:
            closing.set()
            await release.wait()
            os.close(original_fd)
            if fail_close:
                try:
                    os.fsync(original_fd)
                except OSError as error:
                    errors.append(error)
                    raise

    source = await create_owned_iterator(
        lambda: OwnedCallbackSource(produce), operation="original-callback-fd",
    )
    close_owner = None
    try:
        assert await anext(source) == "first"
        assert opened.is_set() and owners == [source.producer_task]
        assert original_path.read_bytes() == b"original source effect"
        close_owner = create_owned_task(source.aclose(), retain_originals=True)
        await closing.wait()
        assert not close_owner.done() and not source.producer_task.done()
        os.fstat(descriptors[0])
        release.set()
        outcome = await await_owned_task(close_owner)
        if fail_close:
            assert errors and outcome.error is errors[0]
            assert errors[0].errno == errno.EBADF
        else:
            raise_owned_outcome(outcome, operation="original callback close", retain_originals=True)
        assert source.producer_task.done() and source.owner_task.done()
        with pytest.raises(OSError) as closed:
            os.fstat(descriptors[0])
        assert closed.value.errno == errno.EBADF
    finally:
        release.set()
        if close_owner is not None:
            await await_owned_task(close_owner)
        else:
            await source.aclose()


def test_callback_rejects_a_noncallable_producer():
    with pytest.raises(TypeError, match="callable producer"):
        OwnedCallbackSource(None)


@pytest.mark.asyncio
async def test_callback_rejects_non_none_completion():
    async def produce(emit):
        return "not an original completion"

    source = await create_owned_iterator(
        lambda: OwnedCallbackSource(produce), operation="invalid-callback-result",
    )
    try:
        with pytest.raises(TypeError, match="must return None"):
            await anext(source)
        assert source.producer_task.done() and source.owner_task.done()
    finally:
        await source.aclose()


@pytest.mark.asyncio
async def test_structured_callback_children_keep_each_original_fd_until_its_own_ack(tmp_path):
    opened = {name: asyncio.Event() for name in ("first", "second")}
    retired = {name: asyncio.Event() for name in opened}
    descriptors, children, producers = {}, [], []

    async def produce(emit):
        producers.append(asyncio.current_task())

        async def child(name):
            original = os.open(tmp_path / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            descriptors[name] = original
            children.append(asyncio.current_task())
            try:
                os.write(original, name.encode())
                opened[name].set()
                await emit(name)
            finally:
                os.close(original)
                retired[name].set()

        # Like a trusted ASGI application's structured streaming children,
        # the original producer physically joins BOTH before returning.
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(child("first"))
            await opened["first"].wait()
            tasks.create_task(child("second"))

    source = await create_owned_iterator(
        lambda: OwnedCallbackSource(produce), operation="structured-callback-frames",
    )
    try:
        assert await anext(source) == "first"
        await opened["second"].wait()
        assert not any(event.is_set() for event in retired.values())
        assert await anext(source) == "second"
        await retired["first"].wait()
        # The first frame's ACK must not resume the second original child.
        # Its actual descriptor remains open until its OWN frame is advanced.
        os.fstat(descriptors["second"])
        assert not retired["second"].is_set()
        assert not source.producer_task.done()
        with pytest.raises(StopAsyncIteration):
            await anext(source)
        assert producers == [source.producer_task]
        assert all(task is not source.producer_task and task.done() for task in children)
        assert all(event.is_set() for event in retired.values())
        for original in descriptors.values():
            with pytest.raises(OSError) as closed:
                os.fstat(original)
            assert closed.value.errno == errno.EBADF
    finally:
        await source.aclose()
