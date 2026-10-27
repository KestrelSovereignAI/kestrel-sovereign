"""Real SQLite/Postgres claim/CAS/ACK and zero post-retirement inference.

The bounded adapter is synthetic; every database relation, transaction, native
COMMIT/reset, current host gate and execution lifetime is production code.
"""

import asyncio
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
import pytest_asyncio
from kestrel_sdk.computer_use import ActionRequest, ControlRequest
from kestrel_sdk.operator import OperatorContext
from kestrel_sdk.resources import (
    ResourceConflict,
    ResourceNotFound,
    ResourceOutcomeUnknown,
)

from kestrel_sovereign._async_rwlock import AsyncReaderWriterLock
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
)
from kestrel_sovereign.privacy import PrivacyConfig, PrivacyMode
from kestrel_sovereign.security.capability_authority import HostCapabilityAuthority
from kestrel_sovereign.storage.privacy_wrapper import DurablePersistenceGate
from kestrel_sovereign.visual_computer.runtime import (
    VisualAccessDenied,
    VisualAction,
    VisualComputerRuntime,
)
from kestrel_sovereign.visual_computer.step import PlanningRecord, run_step
from tests.integration.test_visual_custody_backends import (
    custody as custody,  # noqa: PLC0414 - pytest fixture reexport
)
from tests.integration.test_visual_custody_backends import record
from tests.unit.visual_computer.conftest import Adapter


@pytest_asyncio.fixture
async def step_host(custody):
    store, reopen = custody
    authority = HostCapabilityAuthority(
        capability="visual-computer",
        label="visual computer",
        non_disclosing_actions=frozenset(
            {VisualAction.STOP.value, VisualAction.RELEASE.value}
        ),
        priority_actions=frozenset(
            {VisualAction.STOP.value, VisualAction.RELEASE.value}
        ),
        access_denied=VisualAccessDenied,
    )
    adapter = Adapter()
    runtime = VisualComputerRuntime(
        store,
        authority,
        providers={"test-provider": adapter},
        drivers={"test-driver": adapter},
    )
    privacy = DurablePersistenceGate()
    gate = AsyncReaderWriterLock()
    holder = SimpleNamespace(
        privacy_mode=PrivacyMode.NORMAL,
        privacy_config=PrivacyConfig(),
        _get_durable_persistence_gate=lambda: privacy,
    )
    now = datetime.now(UTC)
    ctx = OperatorContext(
        tenant_id="tenant",
        principal_id="owner",
        acting_agent_id="agent",
        correlation_id="step-native",
        issued_at=now,
        expires_at=now + timedelta(minutes=2),
        granted_capabilities={"visual-computer"},
        permitted_boundary_ids={"visual-computer"},
        granted_actions={a.value for a in VisualAction},
    )
    revoked = set()

    def current(context, action, target):
        if revoked:
            raise PermissionError("original native host permission withdrawn")

    @contextmanager
    def bind():
        with authority.bind_authenticated(
            ctx, privacy_holder=holder, check_current=current, authority_gate=gate
        ):
            yield

    with bind():
        request = record().request.model_copy(
            update={
                "required_capabilities": frozenset(
                    {"screenshot", "click", "type", "takeover"}
                )
            }
        )
        session = await runtime.admit(ctx, request)
    yield SimpleNamespace(
        store=store,
        reopen=reopen,
        runtime=runtime,
        adapter=adapter,
        ctx=ctx,
        session=session,
        bind=bind,
        revoked=revoked,
        privacy=privacy,
    )


async def invoke(
    h, planner, *, identifier="stable-native", prompt="private original task"
):
    return await run_step(
        h.runtime,
        h.ctx,
        h.session.session_id,
        identifier,
        prompt,
        planner=planner,
        configuration={"model": "native-test-fixture"},
        configuration_current=lambda: None,
    )


@pytest.mark.asyncio
async def test_native_claim_ack_precedes_planner_and_reopen(step_host):
    h = step_host
    calls = []

    async def plan(owner):
        calls.append(owner)
        db = await h.reopen()
        try:
            raw = await db.fetchone(
                "SELECT payload,pending FROM visual_computer_actions"
            )
            assert raw[1] == 1 and b"private original task" not in bytes(raw[0])
            from kestrel_sovereign.visual_computer.repository import (
                VisualCustodyRepository,
            )

            opened = VisualCustodyRepository(db)
            stored = await opened.action(
                h.runtime.scope(h.ctx), h.session.session_id, "stable-native"
            )
            assert type(stored) is PlanningRecord and stored.state == "planning"
        finally:
            await db.close()
        return {"action_request": None, "answer": "Done"}

    with h.bind():
        result = await invoke(h, plan)
        assert result["receipt"] is None and len(calls) == 1
        assert await invoke(h, plan) == result and len(calls) == 1
        assert not await h.store.has_pending_actions(
            h.runtime.scope(h.ctx), h.session.session_id
        )
    assert (
        not h.store._original_results and not h.store.db.backend._original_result_holds
    )


@pytest.mark.asyncio
async def test_unknown_native_claim_ack_never_starts_planning(step_host, monkeypatch):
    from kestrel_sovereign.storage.current_visual_step import _CurrentVisualStepMutation

    h = step_host
    calls = []
    original = _CurrentVisualStepMutation.acknowledged

    def lost(self, backend):
        original(self, backend)
        raise RuntimeError("original native ACK observation lost")

    monkeypatch.setattr(_CurrentVisualStepMutation, "acknowledged", lost)

    async def plan(owner):
        calls.append(owner)
        return {"action_request": None, "answer": ""}

    with h.bind():
        with pytest.raises((RuntimeError, BaseExceptionGroup, ExecutionAuthorityError)):
            await invoke(h, plan)
        assert not calls and not h.adapter.effects
        monkeypatch.undo()
        with pytest.raises(ResourceOutcomeUnknown):
            await invoke(h, plan)
        assert not calls and await h.store.has_pending_actions(
            h.runtime.scope(h.ctx), h.session.session_id
        )


@pytest.mark.asyncio
async def test_current_host_retirement_at_actual_commit_keeps_zero_model(
    step_host, monkeypatch
):
    h = step_host
    calls = []
    observed = []
    backend = h.store.db.backend
    if h.store.db.backend_type == "sqlite":
        connection = backend._connection
        execute = connection._execute

        async def queued(fn, *args, **kwargs):
            if (
                getattr(fn, "__name__", "") == "commit"
                and connection._conn._current_row_owner is not None
            ):

                def withdraw():
                    observed.append("native-queued-commit")
                    h.revoked.add("retired")
                    return fn(*args, **kwargs)

                return await execute(withdraw)
            return await execute(fn, *args, **kwargs)

        monkeypatch.setattr(connection, "_execute", queued)
    else:
        from kestrel_sovereign.storage.db import postgres

        original = postgres._prove_committable

        async def after_probe(connection):
            await original(connection)
            if asyncio.current_task().get_name() == "current-visual-step":
                observed.append("native-post-probe")
                h.revoked.add("retired")

        monkeypatch.setattr(postgres, "_prove_committable", after_probe)

    async def plan(owner):
        calls.append(owner)
        return {"action_request": None, "answer": ""}

    with (
        h.bind(),
        pytest.raises(
            (
                RuntimeError,
                BaseExceptionGroup,
                ExecutionAuthorityError,
                VisualAccessDenied,
            )
        ),
    ):
        await invoke(h, plan)
    assert observed and not calls and h.adapter.effects == 0
    monkeypatch.undo()
    h.revoked.clear()
    assert (await h.store.db.fetchone("SELECT COUNT(*) FROM visual_computer_actions"))[
        0
    ] == 0
    assert not backend._original_result_holds


@pytest.mark.asyncio
async def test_same_id_concurrency_never_second_planning(step_host):
    h = step_host
    entered = asyncio.Event()
    resume = asyncio.Event()
    calls = []

    async def plan(owner):
        calls.append(owner)
        entered.set()
        await resume.wait()
        return {"action_request": None, "answer": "Done"}

    with h.bind():
        original = asyncio.create_task(invoke(h, plan))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            with pytest.raises(ResourceOutcomeUnknown):
                await invoke(h, plan)
            assert len(calls) == 1
        finally:
            resume.set()
            await original


@pytest.mark.asyncio
async def test_stop_during_planning_keeps_unknown_host_intent_without_input(step_host):
    h = step_host
    entered = asyncio.Event()
    resume = asyncio.Event()

    async def plan(owner):
        entered.set()
        await resume.wait()
        return {"action_request": None, "answer": "must not deliver"}

    with h.bind():
        task = asyncio.create_task(invoke(h, plan))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            request = ControlRequest(
                request_id="original-stop",
                session_id=h.session.session_id,
                expected_control_epoch=0,
                operation="pause",
            )
            await asyncio.wait_for(
                h.runtime.control(
                    h.ctx, h.session.session_id, request, action=VisualAction.STOP
                ),
                3,
            )
            resume.set()
            with pytest.raises(ResourceConflict):
                await task
        finally:
            resume.set()
            await asyncio.gather(task, return_exceptions=True)
        stored = await h.store.action(
            h.runtime.scope(h.ctx), h.session.session_id, "stable-native"
        )
        assert (
            type(stored) is PlanningRecord
            and stored.pending()
            and h.adapter.effects == 0
        )


@pytest.mark.asyncio
async def test_current_step_cannot_enter_any_original_late_writer(
    step_host, monkeypatch
):
    h = step_host
    backend = h.store.db.backend
    original = backend._commit_current_visual_step
    seen = []

    async def denied_late(owner):
        from kestrel_sovereign.storage.db.original_result_custody import (
            _original_result_profile,
        )

        for call in (lambda: _original_result_profile(owner),):
            with pytest.raises(ExecutionAuthorityError):
                call()
        for operation in (
            backend._write_original_result,
            backend._settle_original_action_result,
            backend._append_original_signal_outcome,
        ):
            with pytest.raises(ExecutionAuthorityError):
                await operation(owner)
        seen.append(owner)
        return await original(owner)

    monkeypatch.setattr(backend, "_commit_current_visual_step", denied_late)

    async def plan(owner):
        return {"action_request": None, "answer": "Done"}

    with h.bind():
        assert (await invoke(h, plan))["state"] == "finished"
    assert len(seen) == 2


@pytest.mark.asyncio
async def test_native_step_promotion_uses_same_dispatch_and_receipt(step_host):
    h = step_host
    calls = []

    async def plan(owner):
        calls.append(owner)
        observation = await h.runtime.observe(h.ctx, h.session.session_id)
        request = ActionRequest(
            action_id=owner.proposal_action_id(),
            session_id=h.session.session_id,
            observation_id=observation.frame.observation_id,
            control_epoch=0,
            not_after=datetime.now(UTC) + timedelta(seconds=10),
            action={"kind": "click", "x": 1, "y": 1},
        )
        return {"action_request": request.model_dump(mode="json"), "answer": ""}

    with h.bind():
        result = await invoke(h, plan)
        assert (
            result["receipt"]["action_id"] == "stable-native" and h.adapter.effects == 1
        )
        assert (await invoke(h, plan))["receipt"] == result["receipt"]
        assert h.adapter.effects == 1 and len(calls) == 1


@pytest.mark.asyncio
async def test_auxiliary_owner_trigger_refuses_before_planning(step_host):
    h = step_host
    if h.store.db.backend_type == "sqlite":
        await h.store.db.execute(
            "CREATE TRIGGER changed_step_owner AFTER UPDATE ON visual_computer_owners BEGIN SELECT 1; END"
        )
    else:
        await h.store.db.execute(
            "ALTER TABLE visual_computer_owners ENABLE ROW LEVEL SECURITY"
        )
    calls = []

    async def plan(owner):
        calls.append(owner)
        return {"action_request": None, "answer": ""}

    with h.bind(), pytest.raises(ExecutionAuthorityError):
        await invoke(h, plan)
    assert (
        not calls
        and (await h.store.db.fetchone("SELECT COUNT(*) FROM visual_computer_actions"))[
            0
        ]
        == 0
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("at", ["before-insert", "before-commit"])
async def test_replacement_runtime_same_id_native_claim_serializes_old_writer(
    step_host, monkeypatch, at
):
    """Absence never suffices: only one of TWO genuine native INSERTs ACKs."""
    from contextlib import asynccontextmanager

    from kestrel_sovereign.visual_computer.repository import VisualCustodyRepository

    h = step_host
    db = await h.reopen()
    other_store = VisualCustodyRepository(db)
    await other_store.ensure_initialized()
    other = VisualComputerRuntime(
        other_store,
        h.runtime.authority,
        providers=h.runtime.providers,
        drivers=h.runtime.drivers,
    )
    entered, resume = asyncio.Event(), asyncio.Event()
    calls = []
    new_absent = asyncio.Event()
    backend = h.store.db.backend
    if at == "before-insert":
        original = backend._commit_current_visual_step

        async def hold(owner):
            if owner._mode == "claim":
                entered.set()
                await resume.wait()
            return await original(owner)

        monkeypatch.setattr(backend, "_commit_current_visual_step", hold)
    elif h.store.db.backend_type == "sqlite":
        original = backend._connection._execute

        async def hold(fn, *args, **kwargs):
            owner = backend._connection._conn._current_row_owner
            if (
                getattr(fn, "__name__", "") == "commit"
                and owner is not None
                and owner._mode == "claim"
            ):
                entered.set()
                await resume.wait()
            return await original(fn, *args, **kwargs)

        monkeypatch.setattr(backend._connection, "_execute", hold)
    else:
        original = backend._tracked_native_transaction

        @asynccontextmanager
        async def hold(*args, **kwargs):
            async with original(*args, **kwargs):
                yield
                owner = kwargs.get("current_mutation")
                if owner is not None and owner._mode == "claim":
                    entered.set()
                    await resume.wait()

        monkeypatch.setattr(backend, "_tracked_native_transaction", hold)

    async def old_plan(owner):
        calls.append("old")
        return {"action_request": None, "answer": "old"}

    async def new_plan(owner):
        calls.append("new")
        return {"action_request": None, "answer": "new"}

    lookup = other_store.action

    async def observe_absence(*args, **kwargs):
        try:
            return await lookup(*args, **kwargs)
        except ResourceNotFound:
            new_absent.set()
            raise

    monkeypatch.setattr(other_store, "action", observe_absence)
    tasks = []
    try:
        with h.bind():
            # A genuine fresh STATUS operation verifies the existing session;
            # its DTO is not a replacement admission or manually set flag.
            await other.reconcile_session(h.ctx, h.session.session_id)
            old = asyncio.create_task(invoke(h, old_plan))
            tasks.append(old)
            await asyncio.wait_for(entered.wait(), 3)
            new = asyncio.create_task(
                run_step(
                    other,
                    h.ctx,
                    h.session.session_id,
                    "stable-native",
                    "private original task",
                    planner=new_plan,
                    configuration={"model": "native-test-fixture"},
                    configuration_current=lambda: None,
                )
            )
            tasks.append(new)
            await asyncio.wait_for(new_absent.wait(), 3)
            if at == "before-insert":
                assert (await asyncio.wait_for(asyncio.shield(new), 3))[
                    "answer"
                ] == "new"
                assert calls == ["new"]
            else:
                await asyncio.sleep(0.03)
                assert not new.done() and not calls
            resume.set()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            assert sum(type(result) is dict for result in results) == 1
            assert calls == (["new"] if at == "before-insert" else ["old"])
            row = await other_store.action(
                other.scope(h.ctx), h.session.session_id, "stable-native"
            )
            assert type(row) is PlanningRecord and row.state == "finished"
            assert (await db.fetchone("SELECT COUNT(*) FROM visual_computer_actions"))[
                0
            ] == 1
    finally:
        resume.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await db.close()


@pytest.mark.asyncio
async def test_native_owner_lock_one_last_action_slot_across_sessions(step_host):
    h = step_host
    h.store.max_actions = 1
    entered = asyncio.Event()
    resume = asyncio.Event()
    calls = []

    async def plan(owner):
        calls.append(owner)
        entered.set()
        await resume.wait()
        return {"action_request": None, "answer": ""}

    with h.bind():
        other = await h.runtime.admit(
            h.ctx,
            record().request.model_copy(
                update={
                    "request_id": "second-session",
                    "required_capabilities": frozenset(
                        {"screenshot", "click", "type", "takeover"}
                    ),
                }
            ),
        )
        first = asyncio.create_task(invoke(h, plan))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            with pytest.raises(BaseExceptionGroup):
                await run_step(
                    h.runtime,
                    h.ctx,
                    other.session_id,
                    "last-slot",
                    "private original task",
                    planner=plan,
                    configuration={},
                    configuration_current=lambda: None,
                )
            assert len(calls) == 1
        finally:
            resume.set()
            await first
        assert (
            await h.store.db.fetchone("SELECT COUNT(*) FROM visual_computer_actions")
        )[0] == 1
