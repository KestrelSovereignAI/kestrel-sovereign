"""A retained public borrow stays revoked after a sweep drops only its fence.

The real runtime and SQLite are used. Provider/driver IO is the canonical
scripted adapter; no physical worker or new external IO is claimed.
"""
import asyncio

import pytest
from kestrel_sdk.resources import ResourceConflict
from kestrel_sovereign._async_ownership import OwnedTaskAdmission, create_owned_task
from kestrel_sovereign.visual_computer.runtime import VisualAction
from tests.unit.visual_computer.test_maintenance_fence_refresh import (
    _action, _two_idle_sessions, _wait,
)
from tests.unit.visual_computer.test_two_domain_retirement import (
    _join_release_tasks, _raise_release_errors,
)


@pytest.mark.asyncio
async def test_refreshed_session_keeps_original_borrow_revoked_after_sweep(harness, monkeypatch):
    h = harness
    context = h.context()
    first_entered, finish_first = asyncio.Event(), asyncio.Event()
    admission = OwnedTaskAdmission()
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
        (_, refreshed), clock = await _two_idle_sessions(h, context, monkeypatch)
        scope = h.runtime.scope(context)
        # Retain the exact public checker from before either sweep fence exists.
        # session_operation also rechecks that same borrow when it exits.
        with pytest.raises(ResourceConflict, match="retained visual session control epoch changed"):
            async with h.runtime.session_operation(
                context, refreshed.session_id, VisualAction.OBSERVE,
            ) as (_, _, retained_check):
                retained_check()
                primary = None
                try:
                    create_owned_task(h.runtime.maintain_owned_sessions(scope),
                        admission=admission, name="original-sweep-with-retained-borrow")
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
                status = await h.runtime.status(context, refreshed.session_id)
                assert status["phase"] == "active" and status["requires_reconciliation"] is False
                # A fresh public borrow can observe and act while the retained
                # pre-fence borrow must still reject its captured old serial.
                observation = await h.runtime.observe(context, refreshed.session_id)
                assert (await h.runtime.act(context, _action(refreshed, observation, clock))).outcome == "completed"
                assert h.adapter.effects == 1
                retained_check()
