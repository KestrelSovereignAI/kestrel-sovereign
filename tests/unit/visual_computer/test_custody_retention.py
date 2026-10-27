"""Bound live custody without forgetting released request/session identities."""

from datetime import UTC, datetime
import json

import pytest
from kestrel_sdk.computer_use import ActionReceipt
from kestrel_sdk.resources import ResourceConflict, ResourceNotFound, ResourceOutcomeUnknown
from kestrel_sdk.security.aead import AEADCipher

from kestrel_sovereign.visual_computer.repository import VisualCustodyRepository
from tests.unit.visual_computer.test_runtime import opened


@pytest.mark.asyncio
async def test_verified_release_reclaims_live_capacity_but_never_resurrects_ids(harness):
    h = harness
    ctx = h.context()
    h.repository.max_requests = 1
    h.repository.max_actions = 1
    with h.bind(ctx):
        session, action = await opened(h, ctx)
        original = (
            await h.repository.get(h.runtime.scope(ctx), session_id=session.session_id)
        ).request
        await h.runtime.act(ctx, action)
        with pytest.raises(ResourceConflict, match="capacity"):
            await h.runtime.admit(ctx, h.request(ctx, request_id="new"))
        assert (await h.runtime.release(ctx, session.session_id))["released"]
        tombstone = await h.repository.get(h.runtime.scope(ctx), session_id=session.session_id)
        assert tombstone.request == original and tombstone.session_id == session.session_id
        assert tombstone.lease.state == "released"
        assert (
            tombstone.connection
            is tombstone.frame
            is tombstone.last_control
            is tombstone.pending_control
            is None
        )
        assert (await h.db.fetchone("SELECT COUNT(*) FROM visual_computer_actions"))[0] == 0
        # Original history is available by exact ID, not scanned as live work.
        assert await h.repository.records(h.runtime.scope(ctx)) == []
        with pytest.raises(ResourceOutcomeUnknown):
            await h.runtime.admit(ctx, original)
        with pytest.raises(ResourceConflict):
            await h.runtime.admit(ctx, original.model_copy(update={"profile_id": "changed"}))
        with pytest.raises(ResourceConflict):
            await h.runtime.act(ctx, action)
        with pytest.raises(ResourceConflict, match="released"):
            await h.runtime.reconcile_session(ctx, session.session_id)
        tombstone.phase = "active"
        with pytest.raises(ResourceConflict, match="immutable"):
            await h.repository.save(h.runtime.scope(ctx), tombstone)
        newer = await h.runtime.admit(ctx, h.request(ctx, request_id="new"))
        assert newer.session_id != session.session_id and h.adapter.acquires == 2
        restarted = VisualCustodyRepository(h.db, max_requests=1, max_actions=1)
        recovered = await restarted.get(h.runtime.scope(ctx), request_id=original.request_id)
        assert recovered.phase == "released" and recovered.request == original
        assert [r.session_id for r in await restarted.records(h.runtime.scope(ctx))] == [
            newer.session_id
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("unknown_release", [False, True])
async def test_unknown_actions_and_unknown_leases_are_never_pruned_or_freed(
    harness, unknown_release
):
    h = harness
    ctx = h.context()
    h.repository.max_requests = 1
    h.repository.max_actions = 1
    with h.bind(ctx):
        session, action = await opened(h, ctx)
        h.adapter.unknown_action = True
        with pytest.raises(ResourceOutcomeUnknown):
            await h.runtime.act(ctx, action)
        h.adapter.unknown_release = unknown_release
        if unknown_release:
            with pytest.raises(ResourceOutcomeUnknown):
                await h.runtime.release(ctx, session.session_id)
        else:
            assert (await h.runtime.release(ctx, session.session_id))["released"]
        restarted = VisualCustodyRepository(h.db, max_requests=1, max_actions=1)
        stored = await restarted.action(h.runtime.scope(ctx), session.session_id, action.action_id)
        assert stored.request == action and stored.receipt is None and stored.dispatch_started
        assert await restarted.has_pending_actions(h.runtime.scope(ctx), session.session_id)
        assert len(await restarted.records(h.runtime.scope(ctx))) == 1
        with pytest.raises(ResourceConflict, match="capacity"):
            await h.runtime.admit(ctx, h.request(ctx, request_id="new"))
        with pytest.raises(ResourceOutcomeUnknown):
            await h.runtime.act(ctx, action)


@pytest.mark.asyncio
async def test_first_upgrade_backfills_authenticated_legacy_rows_without_replay(harness):
    h = harness
    ctx = h.context()
    scope = h.runtime.scope(ctx)
    with h.bind(ctx):
        session, action = await opened(h, ctx)
        await h.runtime.act(ctx, action)
        assert (await h.runtime.release(ctx, session.session_id))["released"]
        owner = h.repository._owner(scope)
        session_key = h.repository._address("session", session.session_id)
        for name in ["legacy-terminal", "legacy-unknown"]:
            request = action.model_copy(update={"action_id": name})
            receipt = (
                None
                if name.endswith("unknown")
                else ActionReceipt(
                    action_id=name,
                    session_id=session.session_id,
                    observation_id=request.observation_id,
                    control_epoch=0,
                    outcome="completed",
                    recorded_at=datetime.now(UTC),
                )
            )
            value = {
                "request": request.model_dump(mode="json"),
                "receipt": receipt.model_dump(mode="json") if receipt else None,
            }
            identity = f"{session_key}:{name}"
            blob = AEADCipher(h.repository._key).encrypt(
                json.dumps(value).encode(), aad=f"{owner}:{identity}".encode()
            )
            await h.db.execute(
                "INSERT INTO visual_computer_actions (owner,session_key,action_id,payload) VALUES (?,?,?,?)",
                (owner, session_key, name, blob),
            )
        await h.db.execute("DROP INDEX visual_computer_retired_requests")
        await h.db.execute("DROP INDEX visual_computer_live_requests")
        await h.db.execute("DROP INDEX visual_computer_pending_actions")
        await h.db.execute("ALTER TABLE visual_computer_requests DROP COLUMN released")
        await h.db.execute("ALTER TABLE visual_computer_actions DROP COLUMN pending")
        restarted = VisualCustodyRepository(h.db, max_requests=1)
        await restarted.ensure_initialized()
        with pytest.raises(ResourceNotFound):
            await restarted.action(scope, session.session_id, "legacy-terminal")
        unknown = await restarted.action(scope, session.session_id, "legacy-unknown")
        assert unknown.dispatch_started and unknown.receipt is None
        assert unknown.request.action == action.action
        assert (await restarted.get(scope, request_id="request")).phase == "released"
        assert (await h.db.fetchone("SELECT pending FROM visual_computer_actions"))[0] == 1
        with pytest.raises(ResourceConflict, match="capacity"):
            await restarted.admit(
                scope,
                (await h.repository.get(scope, request_id="request")).model_copy(
                    update={"request": h.request(ctx, request_id="next"), "session_id": "different"}
                ),
                lambda: None,
            )


@pytest.mark.asyncio
async def test_uncertain_physical_release_preserves_terminal_action_payload(harness):
    h = harness
    ctx = h.context()
    with h.bind(ctx):
        session, action = await opened(h, ctx)
        receipt = await h.runtime.act(ctx, action)
        h.adapter.unknown_release = True
        with pytest.raises(ResourceOutcomeUnknown):
            await h.runtime.release(ctx, session.session_id)
        retained = await h.repository.action(
            h.runtime.scope(ctx), session.session_id, action.action_id
        )
        assert retained.request == action and retained.receipt == receipt
        record = await h.repository.get(h.runtime.scope(ctx), session_id=session.session_id)
        assert record.phase == "fenced" and record.connection is not None
        assert (await h.db.fetchone("SELECT pending FROM visual_computer_actions"))[0] == 0
