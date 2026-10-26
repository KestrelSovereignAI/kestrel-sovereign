"""Actual SQLite preparation/CAS/control races and original worker receipts."""

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest
from kestrel_sdk.computer_use import (
    ActionReceipt,
    ActionRequest,
    ComputerSession,
    ControlRequest,
    ScreenFrame,
    ScreenObservation,
)
from kestrel_sdk.resources import ResourceConflict, ResourceNotFound, ResourceOutcomeUnknown

from kestrel_sovereign.visual_computer.runtime import VisualAction, VisualComputerRuntime
from kestrel_sovereign.visual_computer.repository import VisualCustodyRepository
from tests.unit.visual_computer.test_runtime import opened


async def handback(h, ctx, session):
    for op, action in [("takeover", VisualAction.TAKEOVER), ("resume", VisualAction.HANDBACK)]:
        session = await h.runtime.control(
            ctx,
            session.session_id,
            ControlRequest(
                request_id=op,
                session_id=session.session_id,
                expected_control_epoch=session.control_epoch,
                operation=op,
            ),
            action=action,
        )
    return session


async def next_input(h, ctx, session):
    frame = (await h.runtime.observe(ctx, session.session_id)).frame
    return ActionRequest(
        action_id="fresh",
        session_id=session.session_id,
        observation_id=frame.observation_id,
        control_epoch=session.control_epoch,
        not_after=datetime.now(UTC) + timedelta(seconds=15),
        action={"kind": "click", "x": 1, "y": 1},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("at", ["accept", "dispatch"])
async def test_stop_between_acceptance_and_dispatch_retires_only_known_no_effect(
    harness, monkeypatch, at
):
    h = harness
    ctx = h.context()
    operator = h.context()
    entered, resume = asyncio.Event(), asyncio.Event()
    original = h.repository.action

    async def interleave(*args, **kw):
        result = await original(*args, **kw)
        if (at == "accept" and kw.get("accept") is not None) or (
            at == "dispatch" and kw.get("dispatch_revision") is not None
        ):
            entered.set()
            await resume.wait()
        return result

    with h.bind(ctx), h.bind(operator):
        session, request = await opened(h, ctx)
        monkeypatch.setattr(h.repository, "action", interleave)
        task = asyncio.create_task(h.runtime.act(ctx, request))
        await asyncio.wait_for(entered.wait(), 1)
        stopped = await h.runtime.control(
            operator,
            session.session_id,
            ControlRequest(
                request_id="stop",
                session_id=session.session_id,
                expected_control_epoch=0,
                operation="pause",
            ),
            action=VisualAction.STOP,
        )
        # A later handback cannot revive the old origin's control serial.
        session = await handback(h, operator, stopped)
        resume.set()
        with pytest.raises(ResourceConflict):
            await task
        monkeypatch.setattr(h.repository, "action", original)
        assert h.adapter.effects == 0 and h.runtime._inflight == 0
        denied = await h.runtime.action_status(ctx, session.session_id, request.action_id)
        assert denied.outcome == "denied" and denied.reason_code == "host_not_dispatched"
        assert await h.runtime.act(ctx, request) == denied
        assert (await h.runtime.act(ctx, await next_input(h, ctx, session))).outcome == "completed"
        assert h.adapter.effects == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("at", ["accept", "dispatch"])
async def test_viewer_frame_cas_race_does_not_leave_a_permanent_input_wedge(
    harness, monkeypatch, at
):
    h = harness
    ctx = h.context()
    viewing, show, accepted, resume = (asyncio.Event() for _ in range(4))
    get, action = h.repository.get, h.repository.action
    reads = 0

    async def viewer_get(*args, **kw):
        nonlocal reads
        if asyncio.current_task().get_name() == "retained-viewer":
            reads += 1
            if reads == 2:
                # The viewer already acquired its original session snapshot,
                # but has not yet read the row used for the observation CAS.
                viewing.set()
                await show.wait()
        return await get(*args, **kw)

    async def interleave(*args, **kw):
        result = await action(*args, **kw)
        if (at == "accept" and kw.get("accept") is not None) or (
            at == "dispatch" and kw.get("dispatch_revision") is not None
        ):
            accepted.set()
            await resume.wait()
        return result

    with h.bind(ctx):
        session, request = await opened(h, ctx)
        monkeypatch.setattr(h.repository, "get", viewer_get)
        screen = asyncio.create_task(
            h.runtime.observe(ctx, session.session_id), name="retained-viewer"
        )
        active = None
        try:
            await asyncio.wait_for(viewing.wait(), 1)
            monkeypatch.setattr(h.repository, "action", interleave)
            active = asyncio.create_task(h.runtime.act(ctx, request))
            await asyncio.wait_for(accepted.wait(), 1)
            show.set()
            await screen
            resume.set()
            with pytest.raises(ResourceConflict):
                await active
        finally:
            show.set()
            resume.set()
            await asyncio.gather(
                screen, *(tuple([active]) if active is not None else ()), return_exceptions=True
            )
        monkeypatch.setattr(h.repository, "action", action)
        monkeypatch.setattr(h.repository, "get", get)
        assert h.adapter.effects == 0
        assert (
            await h.runtime.action_status(ctx, session.session_id, request.action_id)
        ).outcome == "denied"
        assert (await h.runtime.act(ctx, await next_input(h, ctx, session))).outcome == "completed"


@pytest.mark.asyncio
async def test_prepared_restart_is_denied_but_dispatch_cas_is_one_shot(harness):
    h = harness
    ctx = h.context()
    with h.bind(ctx):
        session, request = await opened(h, ctx)
        scope = h.runtime.scope(ctx)
        await h.repository.action(
            scope, session.session_id, request.action_id, accept=request, prepare=True
        )
        restarted = VisualComputerRuntime(
            VisualCustodyRepository(h.db),
            h.runtime.authority,
            providers=h.runtime.providers,
            drivers=h.runtime.drivers,
        )
        denied = await restarted.action_status(ctx, session.session_id, request.action_id)
        assert denied.outcome == "denied"
        assert await restarted.act(ctx, request) == denied and h.adapter.effects == 0
        with pytest.raises(ResourceConflict):
            await h.repository.action(
                scope, session.session_id, request.action_id, dispatch_revision=0
            )
        other = request.model_copy(update={"action_id": "other"})
        await h.repository.action(
            scope, session.session_id, other.action_id, accept=other, prepare=True
        )
        current = await h.repository.get(scope, session_id=session.session_id)
        await h.repository.action(
            scope, session.session_id, other.action_id, dispatch_revision=current.revision
        )
        with pytest.raises(ResourceConflict):
            await h.repository.action(
                scope, session.session_id, other.action_id, dispatch_revision=current.revision
            )
        h.adapter.action_status = lambda *a: unavailable()
        with pytest.raises(ResourceOutcomeUnknown):
            await restarted.action_status(ctx, session.session_id, other.action_id)


async def unavailable():
    raise ResourceNotFound("original action unknown")


@pytest.mark.asyncio
async def test_lost_dispatch_ack_never_authorizes_a_second_dispatch(harness, monkeypatch):
    h = harness
    ctx = h.context()
    original = h.repository.action

    async def lost_ack(*args, **kw):
        result = await original(*args, **kw)
        if kw.get("dispatch_revision") is not None:
            raise ResourceOutcomeUnknown("dispatch checkpoint ack lost")
        return result

    with h.bind(ctx):
        session, request = await opened(h, ctx)
        monkeypatch.setattr(h.repository, "action", lost_ack)
        with pytest.raises(ResourceOutcomeUnknown):
            await h.runtime.act(ctx, request)
        monkeypatch.setattr(h.repository, "action", original)
        stored = await original(h.runtime.scope(ctx), session.session_id, request.action_id)
        assert stored.dispatch_started and stored.receipt is None and h.adapter.effects == 0
        with pytest.raises(ResourceOutcomeUnknown):
            await h.runtime.act(ctx, request)
        h.adapter.action_status = lambda *a: unavailable()
        with pytest.raises(ResourceOutcomeUnknown):
            await h.runtime.action_status(ctx, session.session_id, request.action_id)


@pytest.mark.asyncio
async def test_generic_driver_conflict_after_effect_is_never_a_negative_receipt(harness):
    h = harness
    ctx = h.context()

    async def ambiguous(*args):
        h.adapter.effects += 1
        raise ResourceConflict("downstream conflict after input")

    with h.bind(ctx):
        session, request = await opened(h, ctx)
        h.adapter.act = ambiguous
        with pytest.raises(ResourceConflict):
            await h.runtime.act(ctx, request)
        h.adapter.action_status = lambda *a: unavailable()
        with pytest.raises(ResourceOutcomeUnknown):
            await h.runtime.action_status(ctx, session.session_id, request.action_id)
        assert await h.repository.has_pending_actions(h.runtime.scope(ctx), session.session_id)
        with pytest.raises(ResourceOutcomeUnknown):
            await h.runtime.act(ctx, request)
        assert h.adapter.effects == 1


@pytest.mark.asyncio
async def test_actual_worker_stale_refusal_reconciles_exact_negative_receipt(harness):
    from tests.unit.test_visual_browser_controller import BrowserController, Page, Refused

    h = harness
    ctx = h.context()
    page = Page()
    worker = None

    async def open_worker(request, connection):
        nonlocal worker
        worker = BrowserController(
            page, request.model_dump(mode="json"), driver_id=h.adapter.driver_id
        )
        return ComputerSession.model_validate_json(
            json.dumps(worker.status(request.scope.model_dump(mode="json"), request.session_id))
        )

    async def observe(session):
        frame, image = await worker.observe(session.model_dump(mode="json"))
        return ScreenObservation(
            frame=ScreenFrame.model_validate_json(json.dumps(frame)), image=image
        )

    async def act(session, request):
        try:
            return ActionReceipt.model_validate_json(
                json.dumps(
                    await worker.act(
                        session.model_dump(mode="json"), request.model_dump(mode="json")
                    )
                )
            )
        except Refused as error:
            raise ResourceConflict(error.code) from None

    async def status(session, action_id):
        return ActionReceipt.model_validate_json(
            json.dumps(worker.action_status(session.model_dump(mode="json"), action_id))
        )

    h.adapter.open = open_worker
    h.adapter.observe = observe
    h.adapter.act = act
    h.adapter.action_status = status
    with h.bind(ctx):
        session = await h.runtime.admit(
            ctx, h.request(ctx, required_capabilities=frozenset({"screenshot", "click"}))
        )
        request = await next_input(h, ctx, session)
        await worker.observe(
            session.model_dump(mode="json")
        )  # actual worker frame advanced behind host
        with pytest.raises(ResourceConflict, match="stale_observation"):
            await h.runtime.act(ctx, request)
        denied = await h.runtime.action_status(ctx, session.session_id, request.action_id)
        assert (
            denied.outcome == "denied"
            and denied.reason_code == "stale_observation"
            and page.effects == []
        )
        assert await h.runtime.act(ctx, request) == denied
        fresh = (await next_input(h, ctx, session)).model_copy(update={"action_id": "after-denial"})
        assert (await h.runtime.act(ctx, fresh)).outcome == "completed" and len(page.effects) == 1


@pytest.mark.asyncio
async def test_preparation_reserves_normal_capacity_before_its_first_await(harness, monkeypatch):
    from dataclasses import replace

    h = harness
    ctx = h.context()
    entered, resume = asyncio.Event(), asyncio.Event()
    with h.bind(ctx):
        first, one = await opened(h, ctx)
        second = await h.runtime.admit(ctx, h.request(ctx, request_id="second-session"))
        two = await next_input(h, ctx, second)
        h.runtime.policy = replace(h.runtime.policy, max_inflight_actions=1)
        action = h.repository.action

        async def blocked(*args, **kw):
            if kw.get("accept") is not None:
                entered.set()
                await resume.wait()
            return await action(*args, **kw)

        monkeypatch.setattr(h.repository, "action", blocked)
        active = asyncio.create_task(h.runtime.act(ctx, one))
        try:
            await entered.wait()
            with pytest.raises(ResourceConflict, match="capacity"):
                await asyncio.wait_for(h.runtime.act(ctx, two), 0.5)
            assert h.runtime._inflight == 1 and h.adapter.effects == 0
        finally:
            resume.set()
            await active
        assert h.runtime._inflight == 0 and h.adapter.effects == 1
