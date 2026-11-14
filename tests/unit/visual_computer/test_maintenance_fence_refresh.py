"""Real runtime/SQLite interleavings; provider and driver IO are scripted.

These tests do not create or acknowledge a physical worker. Each timing gate
belongs to the exact original Host task, which is always joined in finally.
"""
import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from kestrel_sdk.computer_use import ActionRequest, ControlRequest
from kestrel_sdk.resources import ResourceConflict
from kestrel_sovereign._async_ownership import OwnedTaskAdmission, create_owned_task
from kestrel_sovereign.visual_computer.runtime import VisualAction
from tests.unit.visual_computer.test_two_domain_retirement import (
    _join_release_tasks, _raise_release_errors,
)


async def _wait(event):
    async with asyncio.timeout(5):
        await event.wait()


async def _two_idle_sessions(h, context, monkeypatch):
    now = datetime.now(UTC)  # after the original context's issued_at
    clock = [now]
    monkeypatch.setattr(h.runtime.authority, "clock", lambda: clock[0])
    sessions = []
    for request_id in ("maintenance-first", "refreshed-second"):
        sessions.append(await h.runtime.admit(context, h.request(context,
            request_id=request_id, requested_at=now,
            expires_at=now+timedelta(seconds=90), idle_ttl_seconds=10)))
    original_records = h.repository.records

    async def ordered_snapshot(scope):
        # Only ordering is scripted: every row is read from the real database.
        rows = await original_records(scope)
        return sorted(rows, key=lambda row: row.request.request_id)

    monkeypatch.setattr(h.repository, "records", ordered_snapshot)
    clock[0] = now+timedelta(seconds=11)
    return sessions, clock


def _action(session, observation, clock):
    return ActionRequest(action_id="after-refreshed-sweep", session_id=session.session_id,
        observation_id=observation.frame.observation_id,
        control_epoch=session.control_epoch, not_after=clock[0]+timedelta(seconds=5),
        action={"kind": "click", "x": 10, "y": 10})


@pytest.mark.asyncio
async def test_refresh_during_earlier_release_keeps_later_session_usable(harness, monkeypatch):
    h = harness
    context = h.context()
    first_entered, finish_first = asyncio.Event(), asyncio.Event()
    admission = OwnedTaskAdmission()
    primary = None
    release_ids = []
    original_release = h.adapter.release

    async def blocked_first_release(scope, lease):
        release_ids.append(lease.request_id)
        if lease.request_id == "maintenance-first":
            first_entered.set()
            await _wait(finish_first)
        return await original_release(scope, lease)

    monkeypatch.setattr(h.adapter, "release", blocked_first_release)
    with h.bind(context):
        (first, refreshed), clock = await _two_idle_sessions(h, context, monkeypatch)
        scope = h.runtime.scope(context)
        try:
            create_owned_task(h.runtime.maintain_owned_sessions(scope),
                admission=admission, name="original-stale-snapshot-maintenance")
            await _wait(first_entered)
            await h.runtime.observe(context, refreshed.session_id)
        except BaseException as error:
            primary = error
        finally:
            finish_first.set()
            outcomes, joins = await _join_release_tasks([admission])
        _raise_release_errors(primary, outcomes, joins)
        assert outcomes[0].result == []
        assert release_ids == ["maintenance-first"]
        assert (await h.runtime.status(context, first.session_id))["phase"] == "released"
        status = await h.runtime.status(context, refreshed.session_id)
        assert status["phase"] == "active" and status["requires_reconciliation"] is False
        observation = await h.runtime.observe(context, refreshed.session_id)
        assert (await h.runtime.act(context, _action(refreshed, observation, clock))).outcome == "completed"
        assert h.adapter.effects == 1


@pytest.mark.asyncio
async def test_overlapping_stale_sweeps_retire_only_their_created_fence(harness, monkeypatch):
    h = harness
    context = h.context()
    first_entered, finish_first = asyncio.Event(), asyncio.Event()
    both_snapshots, both_drains = asyncio.Event(), asyncio.Event()
    admissions = [OwnedTaskAdmission(), OwnedTaskAdmission()]
    original_release = h.adapter.release
    original_revoke = h.runtime.egress.revoke_session
    snapshots, drains, releases = [], [], []
    primary = None

    with h.bind(context):
        (_, refreshed), clock = await _two_idle_sessions(h, context, monkeypatch)
        ordered_records = h.repository.records
        scope = h.runtime.scope(context)

        async def snapshot(actual_scope):
            rows = await ordered_records(actual_scope)
            snapshots.append(asyncio.current_task())
            if len(snapshots) == 2:
                both_snapshots.set()
            return rows

        async def blocked_release(actual_scope, lease):
            releases.append(lease.request_id)
            if lease.request_id == "maintenance-first":
                first_entered.set()
                await _wait(finish_first)
            return await original_release(actual_scope, lease)

        async def joined_drains(actual_scope, session_id):
            if session_id == refreshed.session_id:
                task = asyncio.current_task()
                assert task in [owner.task for owner in admissions]
                drains.append(task)
                if len(drains) == 2:
                    both_drains.set()
                await _wait(both_drains)
            return await original_revoke(actual_scope, session_id)

        monkeypatch.setattr(h.repository, "records", snapshot)
        monkeypatch.setattr(h.adapter, "release", blocked_release)
        monkeypatch.setattr(h.runtime.egress, "revoke_session", joined_drains)
        try:
            create_owned_task(h.runtime.maintain_owned_sessions(scope),
                admission=admissions[0], name="original-first-stale-sweep")
            await _wait(first_entered)
            create_owned_task(h.runtime.maintain_owned_sessions(scope),
                admission=admissions[1], name="original-overlapping-stale-sweep")
            await _wait(both_snapshots)
            await h.runtime.observe(context, refreshed.session_id)
        except BaseException as error:
            primary = error
        finally:
            finish_first.set()
            if primary is not None:
                both_drains.set()
            outcomes, joins = await _join_release_tasks(admissions)
        _raise_release_errors(primary, outcomes, joins)
        assert len(snapshots) == len(drains) == 2
        assert len({id(task) for task in drains}) == 2
        assert [outcome.result for outcome in outcomes] == [[], []]
        assert releases == ["maintenance-first"]
        assert not (await h.runtime.status(context, refreshed.session_id))["requires_reconciliation"]
        observation = await h.runtime.observe(context, refreshed.session_id)
        assert (await h.runtime.act(context, _action(refreshed, observation, clock))).outcome == "completed"
        assert h.adapter.effects == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("safety", ["Stop", "Release"])
@pytest.mark.parametrize("ordering", ["pre-existing", "newer"])
async def test_refresh_sweep_preserves_original_safety_fence(harness, monkeypatch, safety, ordering):
    h = harness
    context = h.context()
    first_entered, finish_first = asyncio.Event(), asyncio.Event()
    sweep_drain, finish_sweep_drain = asyncio.Event(), asyncio.Event()
    safety_drain, finish_safety_drain = asyncio.Event(), asyncio.Event()
    sweep, control = OwnedTaskAdmission(), OwnedTaskAdmission()
    original_release, original_revoke = h.adapter.release, h.runtime.egress.revoke_session
    primary, safety_serial = None, None

    with h.bind(context):
        (_, refreshed), clock = await _two_idle_sessions(h, context, monkeypatch)
        scope = h.runtime.scope(context)
        key = h.runtime._key(scope, refreshed.session_id)

        async def blocked_first_release(actual_scope, lease):
            if lease.request_id == "maintenance-first":
                first_entered.set()
                await _wait(finish_first)
            return await original_release(actual_scope, lease)

        async def owned_drain(actual_scope, session_id):
            if session_id == refreshed.session_id:
                if asyncio.current_task() is control.task:
                    safety_drain.set()
                    await _wait(finish_safety_drain)
                elif asyncio.current_task() is sweep.task:
                    sweep_drain.set()
                    await _wait(finish_sweep_drain)
                else:
                    raise AssertionError("unexpected original revoke owner")
            return await original_revoke(actual_scope, session_id)

        def start_safety():
            if safety == "Stop":
                operation = h.runtime.control(context, refreshed.session_id,
                    ControlRequest(request_id="original-priority-stop", session_id=refreshed.session_id,
                        expected_control_epoch=refreshed.control_epoch, operation="pause"),
                    action=VisualAction.STOP)
            else:
                operation = h.runtime.release(context, refreshed.session_id)
            create_owned_task(operation, admission=control, name="original-priority-"+safety)

        monkeypatch.setattr(h.adapter, "release", blocked_first_release)
        monkeypatch.setattr(h.runtime.egress, "revoke_session", owned_drain)
        try:
            create_owned_task(h.runtime.maintain_owned_sessions(scope),
                admission=sweep, name="original-sweep-before-priority-control")
            await _wait(first_entered)
            observation = await h.runtime.observe(context, refreshed.session_id)
            if ordering == "pre-existing":
                start_safety()
                await _wait(safety_drain)
                safety_serial = h.runtime._control_serials[key]
            finish_first.set()
            await _wait(sweep_drain)
            if ordering == "newer":
                start_safety()
                await _wait(safety_drain)
                safety_serial = h.runtime._control_serials[key]
            finish_sweep_drain.set()
            # This is an actual original Task join, not a done() ACK.
            sweep_outcomes, sweep_joins = await _join_release_tasks([sweep])
            _raise_release_errors(None, sweep_outcomes, sweep_joins)
            assert key in h.runtime._fences
            assert h.runtime._control_serials[key] == safety_serial
            assert (await h.runtime.status(context, refreshed.session_id))["phase"] == "active"
            with pytest.raises(ResourceConflict, match="fenced or requires reconciliation"):
                await h.runtime.observe(context, refreshed.session_id)
            with pytest.raises(ResourceConflict, match="fenced or requires reconciliation"):
                await h.runtime.act(context, _action(refreshed, observation, clock))
            assert h.adapter.effects == 0
        except BaseException as error:
            primary = error
        finally:
            finish_first.set()
            finish_sweep_drain.set()
            finish_safety_drain.set()
            outcomes, joins = await _join_release_tasks([sweep, control])
        _raise_release_errors(primary, outcomes, joins)
        assert outcomes[0].result == []
        status = await h.runtime.status(context, refreshed.session_id)
        if safety == "Stop":
            assert outcomes[1].result.state == "paused"
            assert status["phase"] == "fenced" and status["session"]["controller"] == "none"
        else:
            assert outcomes[1].result["released"] is True
            assert status["phase"] == "released"
