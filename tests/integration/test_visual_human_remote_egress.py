"""Provisional combined owner conformance using genuine local TLS/native custody."""
import asyncio
import json
import os
import threading
import sys
from pathlib import Path
from contextlib import contextmanager

import pytest
from kestrel_sovereign._async_ownership import OwnedTaskOutcome, await_owned_task
from tests.integration.test_visual_custody_backends import SCOPE
from tests.integration.test_visual_custody_backends import custody as custody
from tests.integration.test_visual_runpod_acquisition import vendor as vendor
from tests.integration.test_visual_runpod_quote import quoted_operation
from tests.integration.test_visual_runpod_quote import quote_peer as quote_peer
from tests.integration.test_visual_runpod_lifecycle import lifecycle


@contextmanager
def original_native_joins():
    """Read the actual original helper's delivered outcome; never replace it.

    The asyncio executor thread is reusable infrastructure. Its per-operation
    native unit is joined by the canonical owned outcome, not Thread.is_alive.
    Complete process/thread retirement is separately checked at native exit.
    """
    from kestrel_sovereign.execution_sync_http import await_native_sync_http
    original = sys.getprofile()
    assert original is None, "proof refuses to replace an existing profiler"
    code = await_native_sync_http.__code__
    observed = []
    def observe(frame, event, value):
        if event != "return" or frame.f_code is not code:
            return
        values = frame.f_locals
        outcome, thread = values.get("outcome"), values.get("thread")
        if type(outcome) is OwnedTaskOutcome and thread is not None and not thread.active:
            observed.append((thread, outcome))
    sys.setprofile(observe)
    try:
        yield observed
    finally:
        sys.setprofile(original)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["quote", "inspect", "release"])
async def test_original_remote_owner_stop_in_final_human_egress(
    custody, vendor, quote_peer, monkeypatch, tmp_path, kind,
):
    loop = asyncio.get_running_loop()
    ready, release = asyncio.Event(), threading.Event()
    stop_error = call_error = None
    facts = {"kind": kind}
    manager = (quoted_operation(custody, vendor, quote_peer, monkeypatch)
        if kind == "quote" else lifecycle(custody, vendor, quote_peer, monkeypatch,
            release=kind == "release"))
    try:
        async with manager as h:
            call = h.call if kind == "quote" else h.lifecycle
            session_id = h.record.session_id if kind == "quote" else h.row.session_id
            def hold_reply():
                loop.call_soon_threadsafe(ready.set)
                if not release.wait(5):
                    raise TimeoutError("original owned reply was not retired")
            quote_peer.before_reply = hold_reply
            async def stop_original():
                nonlocal stop_error
                await asyncio.wait_for(ready.wait(), 5)
                facts["active_original_types"] = [type(x).__name__ for x in h.runtime.egress._calls.values()]
                facts["original_call_type"] = type(call).__name__
                try:
                    await h.runtime.egress.revoke_session(SCOPE, session_id)
                except BaseException as error:
                    stop_error = error
                    facts["stop_error"] = type(error).__name__
                    facts["stop_error_message"] = str(error)
                    if isinstance(error, AttributeError):
                        facts["missing_attribute"] = error.name
                        facts["missing_attribute_original_type"] = type(error.obj).__name__
                finally:
                    facts["original_active_before_scope_close"] = call._active
                    facts["unrelated_active_attribute"] = getattr(call, "active", None)
                    release.set()
            stopper = asyncio.create_task(stop_original())
            try:
                with original_native_joins() as original_joins:
                    try:
                        await call.run_original()
                    except BaseException as error:
                        call_error = error
                        facts["call_error"] = type(error).__name__
            finally:
                release.set()
                outcome = await await_owned_task(stopper)
                if outcome.error is not None:
                    raise outcome.error
                if outcome.cancellation is not None:
                    raise outcome.cancellation
            worker = call if kind == "quote" else call._stage
            joined = [outcome for thread, outcome in original_joins if thread is worker._thread]
            facts.update(actual_canonical_join_outcomes=len(joined),
                join_outcome_types=[type(outcome).__name__ for outcome in joined],
                join_error_types=[type(outcome.error).__name__ if outcome.error is not None else None for outcome in joined],
                join_owner_cancellation_types=[type(outcome.cancellation).__name__ if outcome.cancellation is not None else None for outcome in joined])
            facts.update(native_requests=len(quote_peer.requests), worker_entered=worker._worker_entered,
                worker_exited=worker._worker_exited, client_closed=worker._client_closed,
                executor_worker_alive_after_unit_join=worker._worker.is_alive(), sockets=len(worker._thread.sockets),
                requests=len(worker._thread.requests))
        facts.update(calls_after_scope=len(h.runtime.egress._calls),
            original_results_after_scope=len(h.store._original_results),
            original_holds_after_scope=len(h.store.db.backend._original_result_holds))
        assert worker._worker_entered and worker._worker_exited and worker._client_closed
        assert joined and all(type(outcome) is OwnedTaskOutcome for outcome in joined)
        assert not worker._thread.active and not worker._thread.sockets and not worker._thread.requests
        assert not h.runtime.egress._calls and not h.store._original_results and not h.store.db.backend._original_result_holds
        assert stop_error is None, f"original {kind} Stop failed before native drain: {stop_error!r}"
        assert call_error is not None, "Stop must withdraw original native continuation"
    finally:
        release.set()
        output = Path(os.environ.get("CURRENT_ROW_PROOF_DIR", str(tmp_path)))
        (output / ("same-map-stop-" + kind + ".json")).write_text(json.dumps(facts, indent=2)+"\n")


@pytest.mark.asyncio
async def test_original_remote_cleanup_refuses_copy_foreign_map_and_closed_owner(
    custody, vendor, quote_peer, monkeypatch,
):
    import copy
    from kestrel_sdk.operator import OperatorAuthorizationError
    from kestrel_sovereign.visual_computer.egress import ActiveCallEgress

    async with quoted_operation(custody, vendor, quote_peer, monkeypatch) as h:
        call = h.call
        for egress, candidate in ((h.runtime.egress, copy.copy(call)), (ActiveCallEgress(), call)):
            with pytest.raises(OperatorAuthorizationError):
                await egress._close_environment_call(candidate)
        stranger = SCOPE.model_copy(update={'principal_id': 'foreign-owner'})
        await h.runtime.egress.revoke_session(stranger, h.record.session_id)
        assert call._active and h.runtime.egress._calls[call._key] is call
        assert not call._used and not quote_peer.requests
        await call.run_original()
    with pytest.raises(OperatorAuthorizationError):
        await h.runtime.egress._close_environment_call(call)
    assert not h.runtime.egress._calls


@pytest.mark.asyncio
async def test_original_remote_running_owner_cannot_acknowledge_its_own_stop(
    custody, vendor, quote_peer, monkeypatch,
):
    from contextlib import asynccontextmanager
    from kestrel_sovereign.execution_custody import execution_cleanup_debts

    async with quoted_operation(custody, vendor, quote_peer, monkeypatch) as h:
        original = h.store._current_runpod_quote
        observed = []
        @asynccontextmanager
        async def observe_current_unit(call):
            async with original(call) as mutation:
                assert call is h.call and call._owner_task is asyncio.current_task()
                assert call._used and not call._finished.done()
                with pytest.raises(TimeoutError) as refused:
                    await h.runtime.egress.revoke_session(SCOPE, h.record.session_id)
                assert any(resource is call and task is asyncio.current_task()
                    for resource, task in execution_cleanup_debts(refused.value))
                observed.append(refused.value)
                assert not call._active and h.runtime.egress._calls[call._key] is call
                yield mutation
        monkeypatch.setattr(h.store, '_current_runpod_quote', observe_current_unit)
        with pytest.raises(Exception):
            await h.call.run_original()
        assert len(observed) == 1 and not quote_peer.requests and not quote_peer.snis
        assert h.call._finished.done() and h.call._thread is None
    assert not h.runtime.egress._calls and not h.store._original_results


@pytest.mark.asyncio
async def test_repeated_stop_caller_withdrawal_retains_original_remote_until_join(
    custody, vendor, quote_peer, monkeypatch,
):
    from contextlib import asynccontextmanager

    loop = asyncio.get_running_loop()
    ready, cleanup = asyncio.Event(), asyncio.Event()
    reply, cleanup_release = threading.Event(), asyncio.Event()
    async with quoted_operation(custody, vendor, quote_peer, monkeypatch) as h:
        original = h.store._current_runpod_quote
        @asynccontextmanager
        async def hold_original_cleanup(call):
            async with original(call) as mutation:
                try:
                    yield mutation
                finally:
                    cleanup.set()
                    await cleanup_release.wait()
        monkeypatch.setattr(h.store, '_current_runpod_quote', hold_original_cleanup)
        def hold_reply():
            loop.call_soon_threadsafe(ready.set)
            if not reply.wait(5):
                raise TimeoutError('original peer reply was not retired')
        quote_peer.before_reply = hold_reply
        async def stop():
            await asyncio.wait_for(ready.wait(), 5)
            await h.runtime.egress.revoke_session(SCOPE, h.record.session_id)
        stopper = asyncio.create_task(stop())
        observed = []
        async def withdraw():
            try:
                await asyncio.wait_for(cleanup.wait(), 5)
                assert not h.call._active and h.runtime.egress._calls[h.call._key] is h.call
                assert not h.call._finished.done() and not stopper.done()
                stopper.cancel('first-original-withdrawal')
                await asyncio.sleep(0)
                stopper.cancel('second-original-withdrawal')
                await asyncio.sleep(0)
                assert h.runtime.egress._calls[h.call._key] is h.call
                assert not h.call._finished.done() and not stopper.done()
                observed.append(True)
            finally:
                reply.set()
                cleanup_release.set()
        controller = asyncio.create_task(withdraw())
        try:
            with original_native_joins() as joins:
                with pytest.raises(Exception):
                    await h.call.run_original()
        finally:
            reply.set()
            cleanup_release.set()
            controlled = await await_owned_task(controller)
            stopped = await await_owned_task(stopper)
        assert controlled.error is None and controlled.cancellation is None and observed == [True]
        assert type(stopped.error) is asyncio.CancelledError
        assert stopped.error.args == ('first-original-withdrawal',)
        assert stopped.cancellation is None
        assert joins and all(outcome.cancellation is None for _, outcome in joins)
        assert h.call._finished.done() and h.call._client_closed
    assert not h.runtime.egress._calls and not h.store._original_results
