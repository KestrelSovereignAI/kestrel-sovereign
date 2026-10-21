"""Cancelled local SDK work keeps actual foreign native accounting denial.

Decision coverage is the actual shared recording finisher with a real caller
withdrawal, not decision-model or full Agent READY acceptance.
"""

import asyncio

import pytest

from kestrel_sovereign.execution_custody import (
    _explicit_control_chain,
    execution_commit_outcome,
    is_execution_control_error,
)
from kestrel_sovereign.llm.adapter import LLMResponse
from tests.integration.native_memory_original_control import (
    native_memory_owners as native_memory_owners,
)
from tests.integration.test_execution_authority_postgres import native_pg as native_pg


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["embedding", "decision", "stream"])
@pytest.mark.parametrize(
    "accounting", ["native-denial", "native-cancel", "ordinary-error", "success"]
)
async def test_aborted_accounting_preserves_original_unscoped_native_denial(
    native_memory_owners, monkeypatch, door, accounting
):
    n = native_memory_owners
    service = n.service
    actual_record = (
        service._record_streamed_usage
        if door == "stream"
        else service.record_modality_call
    )
    recorded, denied, body_errors, accounting_withdrawals = [], [], [], []

    async def record_then_original_denial(*args, **kwargs):
        await actual_record(*args, **kwargs)
        # Observe real SQL after the actual shared recorder settles. No fake
        # accounting result, SQL, native exception, scope or commit marker.
        assert (
            await n.backend.fetch_val("SELECT SUM(total_tokens) FROM model_usage") >= 3
        )
        recorded.append(True)
        if accounting == "success":
            return
        if accounting == "ordinary-error":
            # Explicit callback failure input AFTER real shared accounting;
            # not a native ACK-loss or fabricated authority receipt.
            error = OSError("ordinary recording callback failed")
            denied.append(error)
            raise error
        try:
            n.lease.require_live()
        except BaseException as error:
            denied.append(error)
            if accounting == "native-cancel":
                # Actual callback Task withdrawal, explicitly carrying the
                # real native denial; not a driver-generated COMMIT failure.
                asyncio.current_task().cancel("actual accounting callback withdrew")
                try:
                    await asyncio.sleep(0)
                except asyncio.CancelledError as withdrawal:
                    accounting_withdrawals.append(withdrawal)
                    raise withdrawal from error
            raise

    monkeypatch.setattr(
        service,
        "_record_streamed_usage" if door == "stream" else "record_modality_call",
        record_then_original_denial,
    )
    waiting = asyncio.Event()

    async def cancel_original_body():
        waiting.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as error:
            body_errors.append(error)
            raise

    provider = service.providers[0]
    if door == "embedding":
        embed = service.get_embedding_service()
        method = provider["adapter"].aembed

        async def actual_sdk_then_wait(client, payload, *, usage_sink=None, **kwargs):
            result = await method(client, payload, usage_sink=usage_sink, **kwargs)
            assert result == [1.0, 0.0, 0.0]
            await cancel_original_body()

        operation = asyncio.create_task(
            embed._dispatch("embed", actual_sdk_then_wait, "local fixture", 1)
        )
    elif door == "decision":

        async def finish_original_withdrawal():
            try:
                await cancel_original_body()
            except asyncio.CancelledError as error:
                await service._finish_decision_record(
                    provider=provider,
                    model="local-decision-accounting",
                    caller="local original withdrawal",
                    question_count=1,
                    duration_ms=1,
                    body={"usage": {"input_tokens": 3}},
                    error=error,
                    calibrated=None,
                    context=service.snapshot_invocation_context(),
                )
                raise

        operation = asyncio.create_task(finish_original_withdrawal())
    else:
        stream = service._stream_adapter_with_usage(
            adapter=provider["adapter"],
            client=provider["client"],
            model="local-chat",
            messages=[{"role": "user", "content": "local"}],
            provider_name=provider["name"],
            path="local-original-accounting",
            invocation_context=service.snapshot_invocation_context(),
            expose_protocol_events=True,
        )
        assert await anext(stream) == "local"
        terminal = await anext(stream)
        assert isinstance(terminal, LLMResponse) and terminal.total_tokens == 4
        operation = asyncio.create_task(stream.aclose())

    failure = None
    try:
        if door != "stream":
            await asyncio.wait_for(waiting.wait(), 5)
            operation.cancel("actual caller withdrew")
        try:
            await operation
        except BaseException as error:
            failure = error
        assert recorded == [True], repr(failure)
        if accounting != "success":
            assert len(denied) == 1
            original = denied[0]
            assert is_execution_control_error(original) is (
                accounting.startswith("native-")
            )
            assert execution_commit_outcome(original) is None
            assert failure is not None and original in tuple(
                _explicit_control_chain(failure)
            ), f"{door} erased original accounting outcome: {failure!r}"
            assert is_execution_control_error(failure) is (
                accounting.startswith("native-")
            )
        else:
            assert denied == []
            if door == "stream":
                assert failure is None
        if door != "stream":
            assert len(body_errors) == 1
            assert body_errors[0] in tuple(_explicit_control_chain(failure))
            if not accounting.startswith("native-"):
                assert failure is body_errors[0]
        if accounting == "native-cancel":
            assert len(accounting_withdrawals) == 1
            assert accounting_withdrawals[0] in tuple(_explicit_control_chain(failure))
            if door == "stream":
                assert any(
                    isinstance(error, GeneratorExit)
                    for error in _explicit_control_chain(failure)
                )
        if door != "decision":
            assert n.requests, "actual localhost SDK operation was not dispatched"
    finally:
        if not operation.done():
            operation.cancel()
            await asyncio.gather(operation, return_exceptions=True)
        if door == "stream":
            try:
                await stream.aclose()
            except BaseException as error:
                if not is_execution_control_error(error):
                    raise
