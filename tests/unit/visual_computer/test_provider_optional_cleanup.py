"""Missing optional cleanup configuration must not undo durable custody.

Real encrypted SQLite/authority/runtime; the canonical generic Adapter returns
SCRIPTED provider/driver facts. No native resource or physical ACK claim.
"""
from uuid import uuid4

import pytest
from kestrel_sdk.operator import OperatorAuthorizationError
from kestrel_sdk.resources import ResourceConflict, ResourceError, ResourceOutcomeUnknown
from kestrel_sovereign.visual_computer.repository import CustodyRecord
from kestrel_sovereign.visual_computer.runtime import VisualAction, VisualComputerRuntime


async def _flags(h, request):
    row = await h.db.fetchone(
        "SELECT released, retired FROM visual_computer_requests WHERE owner = ? AND request_id = ?",
        (h.repository._owner(request.scope), request.request_id),
    )
    return list(row)


async def _seed(h, context, state, monkeypatch):
    request = h.request(context, request_id="configuration-removed-" + state)
    if state in {"admitted", "quoted", "retired"}:
        async with h.runtime.operation(context, VisualAction.ADMIT) as current:
            record = await h.repository.admit(request.scope, CustodyRecord(
                request=request, session_id=uuid4().hex, last_activity_at=h.runtime._now()), current)
            if state == "quoted":
                record.quote = await h.adapter.quote(request)
                record.phase = "quoted"
                await h.repository.save(request.scope, record, current)
            elif state == "retired":
                await h.repository.retire_unacquired(request.scope, record, at=h.runtime._now(), guard=current)
    elif state == "acquiring":
        h.adapter.unknown_acquire = True
        with pytest.raises(ResourceOutcomeUnknown, match="acquisition acknowledgement lost"):
            await h.runtime.admit(context, request)
    elif state == "leased":
        refusal = OSError("SCRIPTED original connect refused before driver.open")
        async def refuse_connection(scope, lease):
            raise refusal
        with monkeypatch.context() as patch:
            patch.setattr(h.adapter, "connect", refuse_connection)
            with pytest.raises(OSError) as caught:
                await h.runtime.admit(context, request)
        assert caught.value is refusal
    else:
        session = await h.runtime.admit(context, request)
        if state == "released":
            assert (await h.runtime.release(context, session.session_id))["released"] is True
        elif state == "releasing":
            h.adapter.unknown_release = True
            with pytest.raises(ResourceOutcomeUnknown, match="physical release acknowledgement lost"):
                await h.runtime.release(context, session.session_id)
            h.adapter.unknown_release = False
    record = await h.repository.get(request.scope, request_id=request.request_id)
    assert record.phase == state
    return request, record


def _restarted(h, *, configured):
    return VisualComputerRuntime(h.repository, h.runtime.authority,
        providers=h.runtime.providers if configured else {}, drivers=h.runtime.drivers)


async def _capacity(h, runtime, context, *, charged):
    request = h.request(context, request_id="capacity-" + uuid4().hex)
    async with runtime.operation(context, VisualAction.ADMIT) as current:
        record = CustodyRecord(request=request, session_id=uuid4().hex, last_activity_at=runtime._now())
        if charged:
            with pytest.raises(ResourceConflict, match="request capacity exhausted"):
                await h.repository.admit(request.scope, record, current)
        else:
            await h.repository.admit(request.scope, record, current)
            await h.repository.retire_unacquired(request.scope, record, at=runtime._now(), guard=current)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["release", "reconcile"])
@pytest.mark.parametrize("state", ["admitted", "quoted", "retired", "released"])
async def test_removed_provider_does_not_block_unacquired_or_terminal_custody(harness, monkeypatch, state, operation):
    h = harness
    h.repository.max_requests = 1
    context = h.context()
    with h.bind(context):
        request, before = await _seed(h, context, state, monkeypatch)
        acquisitions = h.adapter.acquires
        restarted = _restarted(h, configured=False)
        result = await (restarted.release(context, before.session_id) if operation == "release"
                        else restarted.reconcile_request(context, request.request_id))
        if state == "released":
            assert result == {"session_id": before.session_id, "released": True}
        else:
            assert result == {"session_id": before.session_id, "request_id": request.request_id,
                              "retired": True, "acquired": False, "released": False}
        after = await h.repository.get(request.scope, request_id=request.request_id)
        assert after.phase == ("released" if state == "released" else "retired")
        assert await _flags(h, request) == ([1, 0] if state == "released" else [0, 1])
        assert h.adapter.acquires == acquisitions  # No replacement or provider entry.
        await _capacity(h, restarted, context, charged=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["release", "reconcile"])
@pytest.mark.parametrize("state", ["acquiring", "leased", "active", "releasing"])
async def test_removed_provider_keeps_actual_acquisition_charged_until_configured(harness, monkeypatch, state, operation):
    h = harness
    h.repository.max_requests = 1
    context = h.context()
    calls = []
    original_release = h.adapter.release
    async def release_original(scope, lease):
        calls.append(lease)
        return await original_release(scope, lease)
    with h.bind(context):
        request, before = await _seed(h, context, state, monkeypatch)
        monkeypatch.setattr(h.adapter, "release", release_original)
        restarted = _restarted(h, configured=False)
        if operation == "reconcile":
            with pytest.raises(OperatorAuthorizationError, match="explicit host grant"):
                await restarted.reconcile_request(context, request.request_id)
        elif before.lease is None:
            with pytest.raises(ResourceOutcomeUnknown, match="unknown acquisition retains"):
                await restarted.release(context, before.session_id)
        else:
            with pytest.raises(ResourceError, match="provider is unavailable"):
                await restarted.release(context, before.session_id)
        after = await h.repository.get(request.scope, request_id=request.request_id)
        assert after.phase in {state, "releasing"}
        assert after.lease == before.lease and after.connection == before.connection
        assert await _flags(h, request) == [0, 0] and calls == []
        await _capacity(h, restarted, context, charged=True)
        restarted.providers[request.provider_id] = h.adapter
        if after.lease is None:
            await restarted.reconcile_request(context, request.request_id)
        assert (await restarted.release(context, before.session_id))["released"] is True
        assert len(calls) == 1 and (await _flags(h, request)) == [1, 0]
        await _capacity(h, restarted, context, charged=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["release", "reconcile"])
async def test_configured_provider_still_reconciles_accepted_lease_and_releases(harness, monkeypatch, operation):
    h = harness
    context = h.context()
    with h.bind(context):
        request, record = await _seed(h, context, "leased" if operation == "reconcile" else "active", monkeypatch)
        runtime = _restarted(h, configured=True)
        if operation == "reconcile":
            session = await runtime.reconcile_request(context, request.request_id)
            assert session.session_id == record.session_id and session.state == "ready"
        assert (await runtime.release(context, record.session_id))["released"] is True
        assert (await _flags(h, request)) == [1, 0] and h.adapter.acquires == 1
