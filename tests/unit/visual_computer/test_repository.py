from datetime import UTC, datetime, timedelta

import pytest
from kestrel_sdk.resources import ResourceConflict, ResourceOutcomeUnknown
from kestrel_sovereign.visual_computer.repository import CustodyIntegrityError, VisualCustodyRepository


@pytest.mark.asyncio
async def test_request_ids_bind_exact_constraints_and_cas_preserves_newer_control(harness):
    h = harness
    ctx = h.context()
    req = h.request(ctx)
    with h.bind(ctx):
        session = await h.runtime.admit(ctx, req)
        changed = req.model_copy(update={"profile_id":"other"})
        with pytest.raises(ResourceConflict):
            await h.runtime.admit(ctx, changed)
        assert h.adapter.acquires == 1
        original = await h.repository.get(h.runtime.scope(ctx), session_id=session.session_id)
        stale = original.model_copy(deep=True)
        original.phase = "fenced"
        await h.repository.save(h.runtime.scope(ctx), original)
        stale.phase = "active"
        with pytest.raises(ResourceConflict):
            await h.repository.save(h.runtime.scope(ctx), stale)
        current = await h.repository.get(h.runtime.scope(ctx), session_id=session.session_id)
        assert current.phase == "fenced"


@pytest.mark.asyncio
async def test_missing_original_envelope_is_not_replaced(harness):
    h = harness
    ctx = h.context()
    with h.bind(ctx):
        await h.runtime.admit(ctx, h.request(ctx))
    await h.db.execute("DELETE FROM visual_computer_keys")
    restarted = VisualCustodyRepository(h.db)
    with pytest.raises(CustodyIntegrityError):
        await restarted.ensure_initialized()
    assert (await h.db.fetchone("SELECT COUNT(*) FROM visual_computer_keys"))[0] == 0


@pytest.mark.asyncio
async def test_idle_cleanup_and_released_request_tombstone_never_reacquire(harness):
    h = harness
    ctx = h.context()
    request = h.request(ctx)
    with h.bind(ctx):
        session = await h.runtime.admit(ctx, request)
        record = await h.repository.get(h.runtime.scope(ctx), session_id=session.session_id)
        record.last_activity_at = datetime.now(UTC)-timedelta(seconds=61)
        await h.repository.save(h.runtime.scope(ctx), record)
        assert await h.runtime.maintain_owned_sessions(h.runtime.scope(ctx)) == []
        record = await h.repository.get(h.runtime.scope(ctx), session_id=session.session_id)
        assert record.phase == "released" and record.lease.released_at is not None
        assert record.connection is None
        with pytest.raises(ResourceOutcomeUnknown) as error:
            await h.runtime.admit(ctx, request)
        assert "reconciliation" in str(error.value)
        assert h.adapter.acquires == 1
