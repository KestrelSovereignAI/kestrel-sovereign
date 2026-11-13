"""Original context retirement uses real Tasks, tokens and file descriptors."""

from __future__ import annotations

import asyncio
import contextvars
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from kestrel_sovereign._async_ownership import OriginalContextExit
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    execution_cleanup_debts,
    execution_evidence_carriers,
    execution_native_source_error,
    is_execution_control_error,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("body_fails", [False, True])
async def test_original_context_keeps_actual_task_token_and_descriptor(
    tmp_path: Path,
    body_fails: bool,
) -> None:
    owner = asyncio.current_task()
    marker = contextvars.ContextVar("original-context", default="outside")
    observations = []
    descriptors = []
    original = ValueError("ordinary body refusal")

    @asynccontextmanager
    async def resource() -> AsyncIterator[int]:
        token = marker.set("inside")
        descriptor = os.open(
            tmp_path / "original", os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600
        )
        descriptors.append(descriptor)
        observations.append(("enter", asyncio.current_task(), marker.get()))
        try:
            yield descriptor
        finally:
            observations.append(("exit", asyncio.current_task(), marker.get()))
            os.close(descriptor)
            marker.reset(token)

    context = resource()
    wrapped = OriginalContextExit(context)
    try:
        async with wrapped as descriptor:
            assert descriptor == descriptors[0]
            os.fstat(descriptor)
            assert marker.get() == "inside"
            if body_fails:
                raise original
    except ValueError as error:
        assert body_fails and error is original
        assert not is_execution_control_error(error)
    assert observations == [("enter", owner, "inside"), ("exit", owner, "inside")]
    assert marker.get() == "outside"
    with pytest.raises(OSError):
        os.fstat(descriptors[0])
    with pytest.raises(RuntimeError, match="cannot be entered twice"):
        await wrapped.__aenter__()
    with pytest.raises(RuntimeError, match="already attempted"):
        await wrapped.__aexit__(None, None, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("body_fails", [False, True])
async def test_failed_real_close_retains_exact_resource_actor_and_all_originals(
    tmp_path: Path,
    body_fails: bool,
) -> None:
    path = tmp_path / "readonly-original"
    path.write_bytes(b"original")
    marker = contextvars.ContextVar("failed-original-context", default="outside")
    owner = asyncio.current_task()
    descriptors, tokens, attempts, physical = [], [], [], []
    body = ValueError("ordinary original body error")

    @asynccontextmanager
    async def resource() -> AsyncIterator[int]:
        tokens.append(marker.set("inside"))
        descriptor = os.open(path, os.O_RDONLY)
        descriptors.append(descriptor)
        try:
            yield descriptor
        finally:
            attempts.append((asyncio.current_task(), marker.get()))
            try:
                os.write(descriptor, b"must fail before close")
            except OSError as error:
                physical.append(error)
                raise
            os.close(descriptor)
            marker.reset(tokens[0])

    context = resource()
    wrapped = OriginalContextExit(context)
    try:
        with pytest.raises(BaseException) as failed:
            async with wrapped:
                if body_fails:
                    raise body
        assert attempts == [(owner, "inside")] and len(physical) == 1
        os.fstat(descriptors[0])
        assert marker.get() == "inside", (
            "A failed original finally was moved or acknowledged"
        )
        assert is_execution_control_error(failed.value)
        assert execution_cleanup_debts(failed.value) == ((context, owner),)
        graph = execution_evidence_carriers(failed.value)
        assert any(error is physical[0] for error in graph)
        if body_fails:
            assert any(error is body for error in graph)
        projected = execution_evidence_carriers(
            execution_native_source_error(failed.value)
        )
        assert any(error is physical[0] for error in projected)
        assert body.__cause__ is None and physical[0].__cause__ is None
        with pytest.raises(BaseException) as repeated:
            await wrapped.__aexit__(None, None, None)
        assert repeated.value is failed.value and len(attempts) == 1
    finally:
        # Test-only release of the actual leaked witness, in its original Task.
        # This does NOT change the production retirement failure into success.
        if descriptors:
            os.close(descriptors[0])
        if tokens:
            marker.reset(tokens[0])


@pytest.mark.asyncio
async def test_foreign_task_cannot_retire_the_original_context(tmp_path: Path) -> None:
    owner = asyncio.current_task()
    closed = []

    @asynccontextmanager
    async def resource() -> AsyncIterator[int]:
        descriptor = os.open(
            tmp_path / "owner", os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600
        )
        try:
            yield descriptor
        finally:
            closed.append(asyncio.current_task())
            os.close(descriptor)

    wrapped = OriginalContextExit(resource())
    descriptor = await wrapped.__aenter__()
    try:
        foreign = asyncio.create_task(wrapped.__aexit__(None, None, None))
        with pytest.raises(
            ExecutionAuthorityError, match="Only the original context Task"
        ):
            await foreign
        assert not closed
        os.fstat(descriptor)
    finally:
        await wrapped.__aexit__(None, None, None)
    assert closed == [owner]


@pytest.mark.asyncio
async def test_context_rejects_unentered_exit_and_retains_original_opening_failure(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "does-not-exist"
    attempts = []

    @asynccontextmanager
    async def resource() -> AsyncIterator[int]:
        attempts.append(asyncio.current_task())
        descriptor = os.open(missing, os.O_RDONLY)
        try:
            yield descriptor
        finally:
            os.close(descriptor)

    wrapped = OriginalContextExit(resource())
    with pytest.raises(RuntimeError, match="has not entered"):
        await wrapped.__aexit__(None, None, None)
    assert not attempts
    with pytest.raises(FileNotFoundError) as opening:
        await wrapped.__aenter__()
    assert not is_execution_control_error(opening.value)
    with pytest.raises(RuntimeError, match="cannot be entered twice"):
        await wrapped.__aenter__()
    with pytest.raises(RuntimeError, match="has not entered"):
        await wrapped.__aexit__(None, None, None)
    assert attempts == [asyncio.current_task()] and not missing.exists()


@pytest.mark.asyncio
async def test_context_without_actual_task_refuses_before_resource_dispatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "must-not-open"

    @asynccontextmanager
    async def resource() -> AsyncIterator[int]:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        try:
            yield descriptor
        finally:
            os.close(descriptor)

    wrapped = OriginalContextExit(resource())
    coroutine = wrapped.__aenter__()
    observed = asyncio.get_running_loop().create_future()

    def actual_loop_callback() -> None:
        # A real loop callback has no current Task. No Task/owner receipt is
        # mocked and the actual cold coroutine must fail before resource work.
        try:
            coroutine.send(None)
        except RuntimeError as error:
            observed.set_result(error)

    asyncio.get_running_loop().call_soon(actual_loop_callback)
    error = await observed
    assert str(error) == "An original context requires an actual asyncio Task"
    assert not path.exists() and coroutine.cr_frame is None


@pytest.mark.asyncio
async def test_cancellation_during_original_exit_remains_physical_retirement_failure(
    tmp_path: Path,
) -> None:
    marker = contextvars.ContextVar("cancelled-context", default="outside")
    closing = asyncio.Event()
    release = asyncio.Event()
    descriptors, originals, actors = [], [], []

    @asynccontextmanager
    async def resource() -> AsyncIterator[int]:
        token = marker.set("inside")
        descriptor = os.open(
            tmp_path / "cancelled", os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600
        )
        descriptors.append(descriptor)
        try:
            yield descriptor
        finally:
            actors.append((asyncio.current_task(), marker.get()))
            closing.set()
            try:
                await release.wait()
            except asyncio.CancelledError as error:
                originals.append(error)
                raise
            finally:
                os.close(descriptor)
                marker.reset(token)

    context = resource()

    async def actual_owner() -> None:
        async with OriginalContextExit(context):
            pass

    actor = asyncio.create_task(actual_owner())
    await closing.wait()
    actor.cancel("Actual original context exit was interrupted")
    with pytest.raises(RuntimeError) as failed:
        await actor
    assert actors == [(actor, "inside")] and len(originals) == 1
    assert is_execution_control_error(failed.value)
    assert execution_cleanup_debts(failed.value) == ((context, actor),)
    assert any(
        error is originals[0] for error in execution_evidence_carriers(failed.value)
    )
    assert marker.get() == "outside"
    with pytest.raises(OSError):
        os.fstat(descriptors[0])
