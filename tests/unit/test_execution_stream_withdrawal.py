"""Explicit cooperative forwarding keeps the actual original Task/context alive."""

import asyncio
import errno
import os
from contextvars import ContextVar

import pytest
from kestrel_sdk.native_a2a.utils.source_ownership import SourceWithdrawal

from kestrel_sovereign.execution_custody import execution_evidence_carriers, owned_execution_stream


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", ["idle", "retiring"])
@pytest.mark.parametrize("close_kind", ["observer", "scope"])
@pytest.mark.parametrize("fail", [False, True])
async def test_explicit_withdrawal_preserves_actual_finalizer_task_context_and_fds(trigger, close_kind, fail):
    signal = SourceWithdrawal()
    waiting, closing, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    value = ContextVar("original-source-token", default=None)
    descriptors, owners, retired, cancelled, physical = [], [], [], [], []
    pull, closer = None, None
    assertions_finished = False

    async def original():
        owner = asyncio.current_task()
        owners.append(owner)
        token = value.set(owner)
        descriptors.extend(os.pipe())
        try:
            yield "first"
            waiting.set()
            await signal.wait()
        finally:
            closing.set()
            try:
                await release.wait()
            except asyncio.CancelledError as error:
                cancelled.append(error)
                raise
            assert asyncio.current_task() is owner and owner.cancelling() == 0
            assert value.get() is owner
            value.reset(token)
            for fd in descriptors:
                os.close(fd)
            retired.append(owner)
            if fail:
                try:
                    os.fsync(descriptors[0])
                except OSError as error:
                    physical.append(error)
                    raise

    try:
        async with owned_execution_stream(object(), original(), withdrawal=signal) as events:
            try:
                async with asyncio.timeout(5):
                    assert await anext(events) == "first" and value.get() is None
                pull = asyncio.create_task(anext(events))
                async with asyncio.timeout(5):
                    await waiting.wait()
                if trigger == "retiring":
                    signal.request_stop()
                    async with asyncio.timeout(5):
                        await closing.wait()
                if close_kind == "observer":
                    pull.cancel()
                else:
                    closer = asyncio.create_task(events.aclose())
                async with asyncio.timeout(5):
                    await closing.wait()
                await asyncio.wait((pull,) if closer is None else (pull, closer), timeout=0.02)
                assert not retired and not cancelled
                assert len(owners) == 1 and not owners[0].done() and owners[0].cancelling() == 0
                for fd in descriptors:
                    os.fstat(fd)
                release.set()
                async with asyncio.timeout(5):
                    results = await asyncio.gather(*[task for task in (pull, closer) if task is not None], return_exceptions=True)
                assert retired == owners and not cancelled and value.get() is None
                if fail:
                    assert len(physical) == 1 and physical[0].errno == errno.EBADF
                    assert any(
                        error is physical[0]
                        for result in results if isinstance(result, BaseException)
                        for error in execution_evidence_carriers(result)
                    )
                assertions_finished = True
            finally:
                # Recovery MUST release before the lexical owner joins close.
                signal.request_stop()
                release.set()
    except BaseException as error:
        graph = execution_evidence_carriers(error)
        if not assertions_finished or not fail or not physical or not any(node is physical[0] for node in graph):
            raise
    finally:
        # Explicit failed-baseline cleanup only; it cannot satisfy assertions.
        signal.request_stop()
        release.set()
        pending = [task for task in (pull, closer) if task is not None]
        if pending:
            async with asyncio.timeout(5):
                await asyncio.gather(*pending, return_exceptions=True)
        for fd in descriptors:
            try:
                os.fstat(fd)
            except OSError as error:
                assert error.errno == errno.EBADF
            else:
                os.close(fd)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["plain", "derived"])
async def test_bad_withdrawal_refuses_before_iterator_acquisition(kind):
    acquired = []

    class DerivedWithdrawal(SourceWithdrawal):
        pass

    class Source:
        def __aiter__(self):
            acquired.append(True)
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    with pytest.raises(TypeError, match="SourceWithdrawal"):
        async with owned_execution_stream(object(), Source(), withdrawal=object() if kind == "plain" else DerivedWithdrawal()):
            pytest.fail("Invalid withdrawal entered an execution source")
    assert not acquired
