"""Genuine original Acquisition ownership closes only after its native writer joins."""

import copy
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest
from kestrel_sdk.computer_use.acquisition import EnvironmentAcquisitionPreparation
from pydantic import SecretBytes

from kestrel_sovereign.execution_custody import ExecutionAuthorityError
from kestrel_sovereign.visual_computer.repository import CustodyRecord
from tests.integration.test_visual_custody_backends import SCOPE
from tests.integration.test_visual_custody_backends import custody as custody
from tests.integration.test_visual_runpod_acquisition import original_acquisition, original_host, original_quote
from tests.integration.test_visual_runpod_acquisition import peer as peer
from tests.integration.test_visual_runpod_acquisition import vendor as vendor


@asynccontextmanager
async def issued(custody, vendor, peer, monkeypatch):
    # Exact native database and original authenticated host/turn/privacy Work.
    # This is the real pre-producer interval, not a constructed result owner.
    async with original_host(custody, vendor, peer, monkeypatch) as h:
        row = await h.store.admit(
            SCOPE, CustodyRecord(request=h.request, session_id="original-close-session"), h.check
        )
        row.quote, row.phase = original_quote(h.plan), "quoted"
        await h.store.save(SCOPE, row, h.check)
        async with h.store.original_acquisition_result(
            SCOPE, row, h.request, row.quote, guard=h.check
        ) as attempt:
            yield h, attempt


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", [False, True], ids=["copied", "replaced-backing"])
async def test_copied_acquisition_close_cannot_retire_original(custody, vendor, peer, monkeypatch, replacement):
    async with issued(custody, vendor, peer, monkeypatch) as (h, attempt):
        original = attempt._result
        copied = copy.copy(original)
        if replacement:
            copied._native_backing = replace(original._native_backing, owner=copied)
            copied._join = replace(original._join, owner=copied)
            copied._registration = replace(
                original._registration, native_backing=copied._native_backing, join=copied._join
            )
        assert h.store._original_results[original._acceptance] is original
        assert h.store.db.backend._original_result_holds[original] is original._closed
        with pytest.raises(ExecutionAuthorityError):
            copied.close()
        assert h.store._original_results[original._acceptance] is original
        assert h.store.db.backend._original_result_holds[original] is original._closed
        assert not original._closed.done()
        assert attempt.checkpoint.state == "unprepared"
    assert original._closed.done()
    assert not h.store._original_results and not h.store.db.backend._original_result_holds
    assert peer.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", [False, True], ids=["copied", "replaced-backing"])
async def test_copied_acquisition_cannot_close_original_prepared_native_client(
    custody, vendor, peer, monkeypatch, replacement
):
    async with original_acquisition(custody, vendor, peer, monkeypatch) as h:
        original = h.attempt._result
        copied = copy.copy(original)
        if replacement:
            copied._native_backing = replace(original._native_backing, owner=copied)
            copied._join = replace(original._join, owner=copied)
            copied._registration = replace(
                original._registration, native_backing=copied._native_backing, join=copied._join
            )
        assert not h.native.is_closed
        with pytest.raises(ExecutionAuthorityError):
            await copied.aclose()
        assert not h.native.is_closed
        assert h.store._original_results[original._acceptance] is original
        assert not original._closed.done()
    assert h.native.is_closed and original._closed.done()
    assert peer.requests == []


@pytest.mark.asyncio
async def test_acquisition_close_requires_actual_native_writer_join(custody, vendor, peer, monkeypatch):
    from kestrel_sovereign.storage import original_acquisition_result as module

    async with issued(custody, vendor, peer, monkeypatch) as (h, attempt):
        root = attempt._result
        original_await = module.await_owned_task
        seen = []

        async def before_join(task):
            # SAME original issuing task, genuinely admitted stage writer.
            # Always physically join the original task, even under the mutant.
            refused = None
            try:
                root.close()
            except ExecutionAuthorityError as error:
                refused = error
            outcome = await original_await(task)
            seen.append(outcome)
            assert refused is not None
            assert h.store._original_results[root._acceptance] is root
            assert h.store.db.backend._original_result_holds[root] is root._closed
            assert not root._closed.done()
            return outcome

        dto = h.plan.pod_request(h.request, h.plan.profile, h.plan.noncompute, at=h.plan.planned_at)
        prepared = h.control.prepare_cpu_pod(dto, plan=h.plan)
        value = EnvironmentAcquisitionPreparation(
            create_payload=SecretBytes(prepared.exact_body()),
            worker_identity=SecretBytes(b"PRIVATE-ORIGINAL-CLOSE-WORKER-IDENTITY"),
        )
        monkeypatch.setattr(module, "await_owned_task", before_join)
        await attempt.prepare_once(value)
        assert len(seen) == 1 and root.has_acknowledged_commit()
        assert attempt.preparation == value
    assert root._closed.done()
    assert not h.store._original_results and not h.store.db.backend._original_result_holds
    assert peer.requests == []
