"""Core delegates source mechanics without losing either original iterator."""

from __future__ import annotations

import asyncio
import contextvars
import errno
import os
from collections.abc import AsyncIterable
from pathlib import Path

import pytest

from kestrel_sdk.native_a2a.utils.source_ownership import CallbackSource
from kestrel_sovereign._async_ownership import OwnedAsyncIterator, OwnedCallbackSource
from kestrel_sovereign.execution_custody import execution_evidence_carriers


def test_callback_source_is_the_shared_sdk_contract() -> None:
    assert OwnedCallbackSource is CallbackSource


@pytest.mark.asyncio
@pytest.mark.parametrize("close_early", [False, True])
async def test_distinct_outer_and_inner_retire_in_the_original_source(
    tmp_path: Path, close_early: bool,
) -> None:
    path = tmp_path / "original-source"
    path.write_bytes(b"x")
    current = contextvars.ContextVar("shared-source-context", default="outside")
    descriptors: list[int] = []
    original: list[asyncio.Task] = []
    retired: list[tuple[str, asyncio.Task | None, str]] = []

    def acquire() -> AsyncIterable[bytes]:
        owner = asyncio.current_task()
        assert owner is not None
        original.append(owner)
        token = current.set("original")
        outer_fd = os.open(path, os.O_RDONLY)
        try:
            inner_fd = os.open(path, os.O_RDONLY)
        except OSError:
            os.close(outer_fd)
            current.reset(token)
            raise
        descriptors.extend((outer_fd, inner_fd))

        class Inner:
            def __aiter__(self) -> Inner:
                return self

            async def __anext__(self) -> bytes:
                value = os.read(inner_fd, 1)
                if not value:
                    raise StopAsyncIteration
                return value

            async def aclose(self) -> None:
                retired.append(("inner", asyncio.current_task(), current.get()))
                os.close(inner_fd)
                current.reset(token)

        inner = Inner()

        class Outer:
            def __aiter__(self) -> Inner:
                return inner

            async def aclose(self) -> None:
                retired.append(("outer", asyncio.current_task(), current.get()))
                os.close(outer_fd)

        return Outer()

    source = OwnedAsyncIterator(acquire, operation="shared-original-source")
    try:
        assert await anext(source) == b"x"
        if not close_early:
            with pytest.raises(StopAsyncIteration):
                await anext(source)
        await source.aclose()
        assert original == [source.producer_task]
        assert source.producer_task.done()
        for descriptor in descriptors:
            with pytest.raises(OSError) as captured:
                os.fstat(descriptor)
            assert captured.value.errno == errno.EBADF
        assert retired == [
            ("inner", source.producer_task, "original"),
            ("outer", source.producer_task, "outside"),
        ]
        assert current.get() == "outside"
    finally:
        await source.aclose()
        # Test-only cleanup keeps the negative pre-fix run from leaking an FD.
        # It is not an original-owner retirement receipt or a second aclose.
        for descriptor in descriptors:
            try:
                os.fstat(descriptor)
            except OSError as error:
                if error.errno != errno.EBADF:
                    raise
            else:
                os.close(descriptor)


@pytest.mark.asyncio
async def test_source_and_failure_classifier_errors_remain_original(
    tmp_path: Path,
) -> None:
    path = tmp_path / "original-resource"
    path.write_bytes(b"x")
    body_error = ValueError("original source failure")
    body_graph = execution_evidence_carriers(body_error)
    predicate_errors: list[OSError] = []
    predicate_graphs: list[tuple[BaseException, ...]] = []
    descriptors: list[int] = []
    retired: list[asyncio.Task | None] = []

    def cleanup_requested() -> bool:
        try:
            with (tmp_path / "missing-predicate-state").open("rb"):
                raise AssertionError("The predicate state must be absent")
        except FileNotFoundError as error:
            predicate_errors.append(error)
            predicate_graphs.append(execution_evidence_carriers(error))
            raise

    class Source:
        def __aiter__(self) -> Source:
            return self

        async def __anext__(self) -> bytes:
            raise body_error

        async def aclose(self) -> None:
            os.close(descriptors[0])
            retired.append(asyncio.current_task())

    def acquire() -> Source:
        descriptors.append(os.open(path, os.O_RDONLY))
        return Source()

    source = OwnedAsyncIterator(
        acquire, operation="original-source-and-classifier-errors",
        cleanup_requested=cleanup_requested,
    )
    try:
        with pytest.raises(BaseException) as captured:
            await anext(source)
        evidence = execution_evidence_carriers(captured.value)
        assert any(error is body_error for error in evidence)
        assert len(predicate_errors) == 1
        assert any(error is predicate_errors[0] for error in evidence)
        assert execution_evidence_carriers(body_error) == body_graph
        assert execution_evidence_carriers(predicate_errors[0]) == predicate_graphs[0]
        assert source.producer_task.done()
        assert retired == [source.producer_task]
        with pytest.raises(OSError) as closed:
            os.fstat(descriptors[0])
        assert closed.value.errno == errno.EBADF
    finally:
        await source.aclose()


@pytest.mark.asyncio
async def test_original_closer_error_graphs_are_not_rewritten() -> None:
    descriptors: list[int] = []
    failures: list[tuple[OSError, tuple[BaseException, ...]]] = []
    retired: list[asyncio.Task | None] = []

    def close_original(index: int) -> None:
        os.close(descriptors[index])
        retired.append(asyncio.current_task())
        try:
            os.fstat(descriptors[index])
        except OSError as error:
            assert error.errno == errno.EBADF
            failures.append((error, execution_evidence_carriers(error)))
            raise
        raise AssertionError("The original descriptor remained open")

    class Inner:
        def __aiter__(self) -> Inner:
            return self

        async def __anext__(self) -> bytes:
            return b"original frame"

        async def aclose(self) -> None:
            close_original(1)

    class Outer:
        def __aiter__(self) -> Inner:
            return Inner()

        async def aclose(self) -> None:
            close_original(0)

    def acquire() -> Outer:
        first = os.open(os.devnull, os.O_RDONLY)
        try:
            second = os.open(os.devnull, os.O_RDONLY)
        except OSError:
            os.close(first)
            raise
        descriptors.extend((first, second))
        return Outer()

    source = OwnedAsyncIterator(acquire, operation="immutable-original-close-errors")
    try:
        assert await anext(source) == b"original frame"
        with pytest.raises(BaseException) as captured:
            await source.aclose()
        evidence = execution_evidence_carriers(captured.value)
        assert len(failures) == 2
        for error, original_graph in failures:
            assert any(carrier is error for carrier in evidence)
            assert execution_evidence_carriers(error) == original_graph
        assert retired == [source.producer_task, source.producer_task]
        assert source.producer_task.done() and source.owner_task.done()
    finally:
        await source.aclose()
        for descriptor in descriptors:
            try:
                os.fstat(descriptor)
            except OSError as error:
                if error.errno != errno.EBADF:
                    raise
            else:
                os.close(descriptor)


@pytest.mark.asyncio
async def test_concurrent_close_cannot_acknowledge_a_pending_original_finally(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pending-original-close"
    path.write_bytes(b"x")
    closing, release = asyncio.Event(), asyncio.Event()
    descriptors: list[int] = []
    retired: list[asyncio.Task | None] = []
    closers: list[asyncio.Task] = []

    class Source:
        def __aiter__(self) -> Source:
            return self

        async def __anext__(self) -> bytes:
            value = os.read(descriptors[0], 1)
            if not value:
                raise StopAsyncIteration
            return value

        async def aclose(self) -> None:
            closing.set()
            await release.wait()
            os.close(descriptors[0])
            retired.append(asyncio.current_task())

    def acquire() -> Source:
        descriptors.append(os.open(path, os.O_RDONLY))
        return Source()

    source = OwnedAsyncIterator(acquire, operation="concurrent-original-close")
    try:
        assert await anext(source) == b"x"
        closers.append(asyncio.create_task(source.aclose()))
        await closing.wait()
        closers.append(asyncio.create_task(source.aclose()))
        completed, _ = await asyncio.wait(closers, timeout=.01)
        assert not completed, "A closer ACKed while the original finally still owns its FD"
        assert not source.producer_task.done()
        os.fstat(descriptors[0])
        release.set()
        assert await asyncio.gather(*closers) == [None, None]
        assert retired == [source.producer_task]
        assert source.producer_task.done() and source.owner_task.done()
        with pytest.raises(OSError) as closed:
            os.fstat(descriptors[0])
        assert closed.value.errno == errno.EBADF
    finally:
        release.set()
        if closers:
            await asyncio.gather(*closers)
        await source.aclose()
