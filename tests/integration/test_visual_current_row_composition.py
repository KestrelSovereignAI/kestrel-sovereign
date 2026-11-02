"""Both genuine current owners share one native engine, never late admission.

Only local fixture TLS and bounded adapters are used. The original canonical
SQLite/PostgreSQL programs, original permission frames and writer joins are real.
"""

import asyncio
import copy
from dataclasses import replace

import pytest
from kestrel_sdk.resources import ResourceOutcomeUnknown

from kestrel_sovereign.execution_custody import ExecutionAuthorityError
from kestrel_sovereign.storage.db.native_row_custody import _native_row_profile
from kestrel_sovereign.visual_computer.dispatch import original_dispatch
from tests.integration.test_visual_custody_backends import SCOPE
from tests.integration.test_visual_custody_backends import custody as custody
from tests.integration.test_visual_original_result_backends import prepared
from tests.integration.test_visual_runpod_acquisition import original_acquisition
from tests.integration.test_visual_runpod_acquisition import vendor as vendor
from tests.integration.test_visual_runpod_acquisition import peer as peer
from tests.integration.test_visual_runpod_quote import quoted_operation
from tests.integration.test_visual_runpod_quote import quote_peer as quote_peer
from tests.integration.test_visual_step_backends import invoke, original_step_host


def _assert_native_reset(backend, kind):
    assert not backend._original_result_holds
    assert backend.owns_open_transaction is False
    if kind == "sqlite":
        native = backend._connection._conn
        assert native._current_row_owner is None
        assert native._journal_program_seal is None
        assert not hasattr(native, "_current_quote_owner")


@pytest.mark.asyncio
async def test_genuine_families_inverse_copy_roles_and_same_connection_reset(
    custody, vendor, quote_peer, monkeypatch
):
    store, _ = custody
    backend = store.db.backend
    methods = {
        "step": backend._commit_current_visual_step,
        "quote": backend._commit_current_runpod_quote,
    }
    engine = backend._write_native_row
    probing = False
    entered = []
    refused_copies = []
    accepted = []

    async def observed_engine(owner):
        assert not probing, "wrong current/late/copy door reached native engine"
        phase = owner._phase
        holds = tuple(backend._original_result_holds.items())
        try:
            result = await engine(owner)
        except ExecutionAuthorityError:
            # Exact-class copied metadata reaches the shared engine, whose
            # first original-registration/backing check must refuse it BEFORE
            # begin, native SQL, hold changes or native connection sealing.
            assert owner._phase == phase
            assert tuple(backend._original_result_holds.items()) == holds
            assert backend.owns_open_transaction is False
            if store.db.backend_type == "sqlite":
                assert backend._connection._conn._current_row_owner is None
                assert backend._connection._conn._journal_program_seal is None
            refused_copies.append(owner)
            raise
        entered.append(owner)
        return result

    monkeypatch.setattr(backend, "_write_native_row", observed_engine)

    def guarded(family):
        async def commit(owner):
            nonlocal probing
            probing = True
            try:
                assert _native_row_profile(owner)[0] == (
                    "visual_computer_actions" if family == "step"
                    else "visual_computer_requests"
                )
                for role in ("owners", "requests"):
                    if family == "step":
                        assert _native_row_profile(owner, role=role)[0].endswith(role)
                    else:
                        with pytest.raises(ExecutionAuthorityError):
                            _native_row_profile(owner, role=role)
                with pytest.raises(ExecutionAuthorityError):
                    await methods["quote" if family == "step" else "step"](owner)
                for late in (
                    backend._write_original_result,
                    backend._append_original_signal_outcome,
                    backend._settle_original_action_result,
                    backend._settle_original_acquisition_result,
                ):
                    with pytest.raises(ExecutionAuthorityError):
                        await late(owner)
                # Metadata has the same exact class/profile; its copied backing
                # or issuer cannot authenticate the original native writer.
                probing = False
                with pytest.raises(ExecutionAuthorityError):
                    await methods[family](copy.copy(owner))
                descriptor = owner._native_backing
                owner._native_backing = replace(descriptor)
                try:
                    with pytest.raises(ExecutionAuthorityError):
                        await methods[family](owner)
                finally:
                    owner._native_backing = descriptor
            finally:
                probing = False
            result = await methods[family](owner)
            assert owner.has_acknowledged_commit()
            if store.db.backend_type == "sqlite":
                assert backend._connection._conn._current_row_owner is None
                assert backend._connection._conn._journal_program_seal is None
            accepted.append((family, owner))
            return result
        return commit

    monkeypatch.setattr(backend, "_commit_current_visual_step", guarded("step"))
    monkeypatch.setattr(backend, "_commit_current_runpod_quote", guarded("quote"))
    original_connection = backend._connection if store.db.backend_type == "sqlite" else None
    async with original_step_host(custody, request_id="step-current-family") as h:
        async with quoted_operation(custody, vendor, quote_peer, monkeypatch) as q:
            quote = await q.call.run_original()
            assert quote.request_id == q.request.request_id
        async def plan(owner):
            return {"action_request": None, "answer": "Finished"}
        with h.bind():
            assert (await invoke(h, plan))["state"] == "finished"
        assert h.adapter.effects == 0
    assert [family for family, _ in accepted] == ["quote", "step", "step"]
    assert [owner for _, owner in accepted] == entered
    assert len(refused_copies) == 2 * len(accepted)
    assert len(quote_peer.requests) == 1
    if original_connection is not None:
        assert backend._connection is original_connection
    _assert_native_reset(backend, store.db.backend_type)


def _error_graph(error):
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        pending.extend(e for e in (
            current.__cause__, current.__context__, *getattr(current, "exceptions", ())
        ) if isinstance(e, BaseException))


@pytest.mark.asyncio
@pytest.mark.parametrize("family", ["step", "quote"])
async def test_current_cutoff_rolls_back_then_other_family_uses_same_native_engine(
    custody, vendor, quote_peer, monkeypatch, family
):
    store, _ = custody
    backend = store.db.backend
    observed = []
    target = "_CurrentVisualStepMutation" if family == "step" else "_CurrentRunpodQuoteMutation"

    def hook(retire):
        if store.db.backend_type == "sqlite":
            connection = backend._connection
            execute = connection._execute
            async def queued(fn, *args, **kwargs):
                owner = connection._conn._current_row_owner
                if (getattr(fn, "__name__", "") == "commit"
                        and type(owner).__name__ == target and not observed):
                    def withdraw():
                        observed.append("actual-queued-commit")
                        retire()
                        return fn(*args, **kwargs)
                    return await execute(withdraw)
                return await execute(fn, *args, **kwargs)
            monkeypatch.setattr(connection, "_execute", queued)
        else:
            from kestrel_sovereign.storage.db import postgres
            prove = postgres._prove_committable
            async def after_probe(connection):
                await prove(connection)
                if (asyncio.current_task().get_name() == "current-visual-step"
                        if family == "step" else
                        asyncio.current_task().get_name() == "current-runpod-quote") and not observed:
                    observed.append("actual-post-probe")
                    retire()
            monkeypatch.setattr(postgres, "_prove_committable", after_probe)

    if family == "quote":
        # The native refusal is caught deliberately to inspect its original
        # cause. Normal context exit still cannot deliver under that withdrawn
        # grant: preserve the independent exact final-delivery UNKNOWN as well.
        with pytest.raises(
            ResourceOutcomeUnknown,
            match="visual operation completed without current delivery authority",
        ):
            async with quoted_operation(custody, vendor, quote_peer, monkeypatch) as q:
                hook(lambda: q.current.__setitem__("live", False))
                with pytest.raises(BaseException) as denied:
                    await q.call.run_original()
        original_cause = "original host grant withdrawn"
        row = await store.get(SCOPE, request_id="original")
        assert row.phase == "admitted" and row.quote is None
    else:
        async with original_step_host(custody, request_id="step-withdrawn") as h:
            hook(lambda: h.revoked.add("retired"))
            calls = []
            async def must_not_plan(owner):
                calls.append(owner)
                return {"action_request": None, "answer": ""}
            with h.bind(), pytest.raises(BaseException) as denied:
                await invoke(h, must_not_plan)
            assert not calls and h.adapter.effects == 0
        original_cause = "original native host permission withdrawn"
        assert (await store.db.fetchone("SELECT COUNT(*) FROM visual_computer_actions"))[0] == 0
    assert observed
    assert any(isinstance(e, PermissionError) and str(e) == original_cause
               for e in _error_graph(denied.value))
    _assert_native_reset(backend, store.db.backend_type)
    # The original denied frame stays denied. Only fresh independent authority
    # can enter a different original native operation after the rollback.
    if family == "step":
        async with quoted_operation(custody, vendor, quote_peer, monkeypatch) as fresh:
            assert (await fresh.call.run_original()).request_id == fresh.request.request_id
    async with original_step_host(custody, request_id="step-after-rollback") as fresh:
        async def done(owner):
            return {"action_request": None, "answer": "Finished"}
        with fresh.bind():
            assert (await invoke(fresh, done))["state"] == "finished"
    _assert_native_reset(backend, store.db.backend_type)


@pytest.mark.asyncio
async def test_genuine_original_action_cannot_enter_current_writer(custody):
    store, _ = custody
    current, session, request = await prepared(store)
    backend = store.db.backend
    with original_dispatch(session, request) as origin:
        async with store.original_action_result(SCOPE, session, request, guard=lambda: None) as result:
            for role in ("owners", "requests"):
                with pytest.raises(ExecutionAuthorityError):
                    _native_row_profile(result, role=role)
            for operation in (backend._commit_current_visual_step, backend._commit_current_runpod_quote):
                with pytest.raises(ExecutionAuthorityError):
                    await operation(result)
            await store.action(SCOPE, session.session_id, request.action_id,
                               dispatch_revision=current.revision, _result_owner=result)
            await result.not_dispatched(origin, origin.refuse_unentered(RuntimeError("no native driver entered")))
    _assert_native_reset(backend, store.db.backend_type)


@pytest.mark.asyncio
async def test_genuine_original_acquisition_cannot_enter_current_writer(
    custody, vendor, peer, monkeypatch
):
    store, _ = custody
    backend = store.db.backend
    async with original_acquisition(custody, vendor, peer, monkeypatch) as original:
        # The public SDK attempt facade is not a native result. The inverse
        # must also reject the genuine issued Acquisition root behind it.
        for owner in (original.attempt, original.attempt._result):
            for role in ("owners", "requests"):
                with pytest.raises(ExecutionAuthorityError):
                    _native_row_profile(owner, role=role)
            for operation in (backend._commit_current_visual_step, backend._commit_current_runpod_quote,
                              backend._settle_original_action_result, backend._append_original_signal_outcome):
                with pytest.raises(ExecutionAuthorityError):
                    await operation(owner)
        assert await original.attempt.begin_dispatch() is True
        assert original.attempt.checkpoint.state == 'dispatching'
        assert not peer.requests
    _assert_native_reset(backend, store.db.backend_type)


@pytest.mark.asyncio
async def test_genuine_original_signal_refuses_current_doors_then_native_insert_ack(
    custody, monkeypatch
):
    from kestrel_sovereign.signals import (
        OrderedLockManager, SignalDispatcher, SignalLogStore, SourceRegistry,
    )
    from tests.unit.test_signals_dispatcher import _FakeAgent, _action_reg, _signal

    store, _ = custody
    backend = store.db.backend
    log = SignalLogStore(backend)
    await log.initialize()
    registry = SourceRegistry()
    agent = _FakeAgent()
    dispatcher = SignalDispatcher(agent=agent, registry=registry,
        lock_manager=OrderedLockManager(), store=log)
    effects, acknowledgments = [], []
    original = backend._append_original_signal_outcome
    engine = backend._write_native_row
    probing = False

    async def observed_engine(owner):
        assert not probing, "original Signal entered engine through a CURRENT door"
        return await engine(owner)

    async def original_writer(owner):
        nonlocal probing
        assert _native_row_profile(owner)[0] == "signal_log"
        probing = True
        try:
            for role in ("owners", "requests"):
                with pytest.raises(ExecutionAuthorityError):
                    _native_row_profile(owner, role=role)
            for current in (
                backend._commit_current_visual_step, backend._commit_current_runpod_quote,
            ):
                with pytest.raises(ExecutionAuthorityError):
                    await current(owner)
        finally:
            probing = False
        result = await original(owner)
        assert owner.has_acknowledged_commit()
        acknowledgments.append(owner)
        return result

    monkeypatch.setattr(backend, "_write_native_row", observed_engine)
    monkeypatch.setattr(backend, "_append_original_signal_outcome", original_writer)
    async def handler(payload):
        effects.append("one original bounded handler")
        return {"completed": True}
    registry.register(_action_reg("current.row.inverse.signal", handler=handler))
    signal = _signal("current.row.inverse.signal")
    try:
        assert (await dispatcher.dispatch_signal(signal)).status.value == "ok"
        assert effects == ["one original bounded handler"]
        assert len(acknowledgments) == 1
        assert dispatcher._log_write_failures == 0
        assert await backend.fetch_val(
            "SELECT COUNT(*) FROM signal_log WHERE id=?", (signal.id,)
        ) == 1
    finally:
        await dispatcher.shutdown_durable_delivery()
        if agent.background_tasks:
            await asyncio.gather(*agent.background_tasks)
    _assert_native_reset(backend, store.db.backend_type)
