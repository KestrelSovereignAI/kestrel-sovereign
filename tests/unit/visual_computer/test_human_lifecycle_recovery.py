"""Native encrypted SQLite and original authority, with controlled SDK endpoints.

These deterministic service schedules establish admission/recovery behavior;
they do not establish physical Docker or browser behavior.
"""
import asyncio
import json
import os
from pathlib import Path

import pytest
from kestrel_sdk.computer_use import ControlRequest
from kestrel_sdk.operator import OperatorAuthorizationError
from kestrel_sdk.resources import ResourceConflict, ResourceOutcomeUnknown
from kestrel_sovereign.visual_computer.runtime import VisualAction
from kestrel_sovereign._async_ownership import await_owned_operation, raise_owned_outcome
from kestrel_sovereign.visual_computer.docker_human import human_operation_id
from tests.unit.visual_computer.test_human import setup, input_request
from tests.unit.visual_computer.test_runtime import opened


def persist(tmp_path, name, facts):
    out = Path(os.environ.get("CURRENT_ROW_PROOF_DIR", tmp_path))
    (out / (name + ".json")).write_text(json.dumps(facts, indent=2) + "\n")


@pytest.mark.asyncio
async def test_same_handback_id_reconciles_failed_fresh_agent_observation(harness, tmp_path, monkeypatch):
    h = harness
    context = h.context()
    with h.bind(context):
        session, _, extension = await setup(h, context)
        original_observe = h.adapter.observe
        calls = []
        failure = ResourceOutcomeUnknown("owned fresh observation acknowledgement unavailable")

        async def fail_once(selected):
            calls.append(selected)
            if len(calls) == 1:
                raise failure
            return await original_observe(selected)

        monkeypatch.setattr(h.adapter, "observe", fail_once)
        request = ControlRequest(request_id="same-original-handback", session_id=session.session_id,
            expected_control_epoch=session.control_epoch, operation="resume")
        with pytest.raises(ResourceOutcomeUnknown) as caught:
            await h.runtime.control(context, session.session_id, request,
                action=VisualAction.HANDBACK, share_screen=True)
        assert caught.value is failure
        accepted = await h.repository.get(h.runtime.scope(context), session_id=session.session_id)
        assert accepted.last_control == request and accepted.session.control_epoch == 2
        assert accepted.frame is None and accepted.session.controller == "agent"
        result = await h.runtime.control(context, session.session_id, request,
            action=VisualAction.HANDBACK, share_screen=True)
        current = await h.repository.get(h.runtime.scope(context), session_id=session.session_id)
        facts = {"observe_calls": len(calls), "same_original_control": current.last_control == request,
            "control_epoch": current.session.control_epoch, "fresh_agent_frame": current.frame is not None,
            "original_control_receipt_retained": current.control_receipt == accepted.control_receipt,
            "human_lease_revoked": extension.lease.state == "revoked"}
        persist(tmp_path, "handback-recovery", facts)
        assert current.frame is not None, facts
        assert len(calls) == 2 and current.frame.control_epoch == result.control_epoch == 2
        assert result == accepted.control_receipt and current.last_control == request
        assert extension.lease.state == "revoked" and extension.effects == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["observe", "act"])
async def test_release_refuses_late_human_admission_and_joins_before_ack(
    harness, tmp_path, monkeypatch, operation,
):
    h = harness
    context = h.context()
    with h.bind(context):
        session, lease, extension = await setup(h, context)
        request = await input_request(h, context, session, lease) if operation == "act" else None
        original_release = h.adapter.release
        release_entered, finish_release = asyncio.Event(), asyncio.Event()
        human_entered, finish_human = asyncio.Event(), asyncio.Event()

        async def held_release(scope, selected):
            release_entered.set()
            await finish_release.wait()
            return await original_release(scope, selected)

        async def held_human():
            human_entered.set()
            await finish_human.wait()

        monkeypatch.setattr(h.adapter, "release", held_release)
        extension.before = held_human
        release = asyncio.create_task(h.runtime.release(context, session.session_id))
        late = entered_wait = None
        result = None
        facts = {"operation": operation}
        try:
            await asyncio.wait_for(release_entered.wait(), 5)
            call = (h.runtime.human.observe(context, session.session_id, "human") if operation == "observe"
                else h.runtime.human.act(context, session.session_id, "human", request))
            late = asyncio.create_task(call)
            entered_wait = asyncio.create_task(human_entered.wait())
            await asyncio.wait_for(asyncio.wait({late, entered_wait}, return_when=asyncio.FIRST_COMPLETED), 5)
            facts["sdk_admitted_while_releasing"] = human_entered.is_set()
            finish_release.set()
            result = await release
            facts["release_ack"] = result
            facts["late_done_at_release_ack"] = late.done()
            facts["effects_at_release_ack"] = extension.effects
        finally:
            finish_release.set()
            finish_human.set()
            if entered_wait is not None:
                entered_wait.cancel()
            pending = [t for t in (release, late, entered_wait) if t is not None]
            outcomes = await asyncio.gather(*pending, return_exceptions=True)
            late_result = None if late is None else outcomes[pending.index(late)]
            facts["late_result_type"] = type(late_result).__name__
            facts["final_effects"] = extension.effects
            facts["egress_calls_after_original_children_joined"] = len(h.runtime.egress._calls)
            facts["native_transaction_open"] = h.db.backend._in_transaction
            facts["original_result_holds"] = len(h.db.backend._original_result_holds)
            persist(tmp_path, "release-" + operation, facts)
        assert result == {"session_id": session.session_id, "released": True}
        assert not facts["sdk_admitted_while_releasing"], facts
        assert facts["late_done_at_release_ack"], facts
        assert isinstance(late_result, ResourceConflict), facts
        assert not h.runtime.egress._calls
        assert extension.effects == 0


@pytest.mark.asyncio
async def test_missing_handback_frame_requires_fresh_observe_grant_and_confirmation(harness, monkeypatch):
    h = harness
    context = h.context()
    with h.bind(context):
        session, _, _ = await setup(h, context)
        original = h.adapter.observe
        calls = []
        async def unavailable(selected):
            calls.append(selected)
            raise ResourceOutcomeUnknown("original observation unavailable")
        monkeypatch.setattr(h.adapter, "observe", unavailable)
        request = ControlRequest(request_id="original-resume", session_id=session.session_id,
            expected_control_epoch=session.control_epoch, operation="resume")
        with pytest.raises(ResourceOutcomeUnknown):
            await h.runtime.control(context, session.session_id, request, action=VisualAction.HANDBACK, share_screen=True)
        with pytest.raises(ResourceConflict, match="confirm"):
            await h.runtime.control(context, session.session_id, request, action=VisualAction.HANDBACK)
        assert len(calls) == 1
    limited = h.context(actions={VisualAction.HANDBACK.value})
    with h.bind(limited), pytest.raises(OperatorAuthorizationError):
        await h.runtime.control(limited, session.session_id, request, action=VisualAction.HANDBACK, share_screen=True)
    assert len(calls) == 1
    with h.bind(context):
        monkeypatch.setattr(h.adapter, "observe", original)
        result = await h.runtime.control(context, session.session_id, request, action=VisualAction.HANDBACK, share_screen=True)
        # A completed cached result requires no replacement frame or SDK call.
        monkeypatch.setattr(h.adapter, "observe", unavailable)
        assert await h.runtime.control(context, session.session_id, request, action=VisualAction.HANDBACK) == result
        assert len(calls) == 1


@pytest.mark.asyncio
async def test_release_intent_is_monotonic_and_maintenance_retries_original_lease(harness):
    h = harness
    context = h.context()
    with h.bind(context):
        session, _ = await opened(h, context)
        scope = h.runtime.scope(context)
        original = await h.repository.get(scope, session_id=session.session_id)
        h.adapter.unknown_release = True
        with pytest.raises(ResourceOutcomeUnknown):
            await h.runtime.release(context, session.session_id)
        pending = await h.repository.get(scope, session_id=session.session_id)
        assert pending.phase == "releasing" and pending.lease == original.lease
        assert (await h.runtime.status(context, session.session_id))["requires_reconciliation"]
        # A fresh revision does not permit a late control result to revive it.
        with pytest.raises(ResourceConflict, match="release intent"):
            await h.repository.save(scope, pending.model_copy(update={"phase": "active"}))
        with pytest.raises(ResourceConflict):
            await h.runtime.reconcile_session(context, session.session_id)
        request = ControlRequest(request_id="late-control", session_id=session.session_id,
            expected_control_epoch=session.control_epoch, operation="takeover")
        with pytest.raises(ResourceConflict):
            await h.runtime.control(context, session.session_id, request, action=VisualAction.TAKEOVER)
        assert await h.runtime.fence_owned_sessions(scope) == [session.session_id]
        assert (await h.repository.get(scope, session_id=session.session_id)).phase == "releasing"
        # Fixed lifetime is still in the future; pending original retirement
        # must nevertheless be retried, without another acquire or extension.
        assert h.runtime._now() < original.lease.expires_at
        h.adapter.unknown_release = False
        assert await h.runtime.maintain_owned_sessions(scope) == []
        retired = await h.repository.get(scope, session_id=session.session_id)
        assert retired.phase == "released" and retired.lease.lease_id == original.lease.lease_id
        assert retired.lease.expires_at == original.lease.expires_at
        assert h.adapter.acquires == 1 and h.adapter.effects == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["observe", "control", "status", "action_status"])
async def test_release_joins_previously_entered_sdk_call_cleanup(harness, monkeypatch, operation):
    h = harness
    context = h.context()
    with h.bind(context):
        session, action = await opened(h, context)
        if operation == "action_status":
            h.adapter.unknown_action = True
            with pytest.raises(ResourceOutcomeUnknown):
                await h.runtime.act(context, action)
        entered, closing, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
        physical = []
        original_release = h.adapter.release
        async def blocked(*args):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                closing.set()
                await finish.wait()
        async def release_original(scope, lease):
            physical.append(lease)
            assert finish.is_set()
            return await original_release(scope, lease)
        monkeypatch.setattr(h.adapter, operation, blocked)
        monkeypatch.setattr(h.adapter, "release", release_original)
        if operation == "observe":
            selected = h.runtime.observe(context, session.session_id)
        elif operation == "control":
            request = ControlRequest(request_id="original-pause", session_id=session.session_id,
                expected_control_epoch=session.control_epoch, operation="pause")
            selected = h.runtime.control(context, session.session_id, request, action=VisualAction.STOP)
        elif operation == "status":
            selected = h.runtime.reconcile_session(context, session.session_id)
        else:
            selected = h.runtime.action_status(context, session.session_id, action.action_id)
        original_call = asyncio.create_task(selected)
        release = None
        try:
            await asyncio.wait_for(entered.wait(), 5)
            release = asyncio.create_task(h.runtime.release(context, session.session_id))
            await asyncio.wait_for(closing.wait(), 5)
            assert not original_call.done() and not release.done() and not physical
            # Repeated observer withdrawal must not interrupt native cleanup.
            original_call.cancel()
            await asyncio.sleep(0)
            assert not original_call.done() and not release.done() and not physical
        finally:
            finish.set()
            original_call.cancel()
            outcomes = await asyncio.gather(*[t for t in (original_call, release) if t is not None], return_exceptions=True)
        assert isinstance(outcomes[0], asyncio.CancelledError), outcomes
        assert outcomes[1] == {"session_id": session.session_id, "released": True}
        assert len(physical) == 1 and not h.runtime.egress._calls


@pytest.mark.asyncio
async def test_second_release_drain_joins_human_reader_between_fence_and_publication(harness, monkeypatch):
    h = harness
    context = h.context()
    with h.bind(context):
        session, _, extension = await setup(h, context)
        publishing, publish, entered = asyncio.Event(), asyncio.Event(), asyncio.Event()
        closing, finish = asyncio.Event(), asyncio.Event()
        original_save, original_release = h.repository.save, h.adapter.release
        async def held_publication(scope, record, guard=None):
            if record.phase == "releasing":
                publishing.set()
                await publish.wait()
            await original_save(scope, record, guard)
        async def held_human():
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                closing.set()
                await finish.wait()
        physical = []
        async def release_original(scope, lease):
            physical.append(lease)
            assert finish.is_set()
            return await original_release(scope, lease)
        monkeypatch.setattr(h.repository, "save", held_publication)
        monkeypatch.setattr(h.adapter, "release", release_original)
        extension.before = held_human
        release = asyncio.create_task(h.runtime.release(context, session.session_id))
        late = None
        try:
            await asyncio.wait_for(publishing.wait(), 5)
            late = asyncio.create_task(h.runtime.human.observe(context, session.session_id, "human"))
            await asyncio.wait_for(entered.wait(), 5)
            publish.set()
            await asyncio.wait_for(closing.wait(), 5)
            assert not release.done() and not late.done() and not physical
        finally:
            publish.set()
            finish.set()
            if late is not None:
                late.cancel()
            outcomes = await asyncio.gather(*[t for t in (release, late) if t is not None], return_exceptions=True)
        assert outcomes[0] == {"session_id": session.session_id, "released": True}
        assert isinstance(outcomes[1], asyncio.CancelledError)
        assert len(physical) == 1 and not h.runtime.egress._calls and extension.effects == 0


@pytest.mark.asyncio
async def test_releasing_human_metadata_is_read_only_without_new_native_reconciliation(harness, monkeypatch):
    h = harness
    context = h.context()
    with h.bind(context):
        session, lease, extension = await setup(h, context)
        request = await input_request(h, context, session, lease)
        receipt = await h.runtime.human.act(context, session.session_id, "human", request)
        h.adapter.unknown_release = True
        with pytest.raises(ResourceOutcomeUnknown):
            await h.runtime.release(context, session.session_id)
        before = await h.repository.get(h.runtime.scope(context), session_id=session.session_id)
        calls = []
        async def unexpected(*args):
            calls.append(args)
            pytest.fail("pending release admitted another worker call")
        monkeypatch.setattr(extension, "status", unexpected)
        monkeypatch.setattr(extension, "revoke", unexpected)
        assert await h.runtime.human.action_status(context, session.session_id, "human", request.action.action_id) == receipt
        with pytest.raises(ResourceConflict, match="releasing"):
            await h.runtime.human.status(context, session.session_id, "human")
        with pytest.raises(ResourceConflict, match="releasing"):
            await h.runtime.human.revoke(context, session.session_id, "human", "late-close", session.control_epoch)
        async with h.runtime.operation(context, VisualAction.HUMAN_VIEW) as check:
            with pytest.raises(ResourceConflict, match="releasing"):
                async with h.runtime.human.transport_delivery(context, session.session_id, "human", check, metadata=True):
                    pytest.fail("pending release registered new transport delivery")
        assert not calls and not h.runtime.egress._calls
        assert await h.repository.get(h.runtime.scope(context), session_id=session.session_id) == before
        assert extension.effects == 1  # The exact already-completed original input.


@pytest.mark.asyncio
async def test_release_wins_cached_handback_reader_before_fence_removal(harness, monkeypatch):
    h = harness
    context = h.context()
    with h.bind(context):
        session, _, _ = await setup(h, context)
        observed = []
        async def unavailable(selected):
            observed.append(selected)
            raise ResourceOutcomeUnknown("original fresh observation lost")
        monkeypatch.setattr(h.adapter, "observe", unavailable)
        request = ControlRequest(request_id="cached-original-handback", session_id=session.session_id,
            expected_control_epoch=session.control_epoch, operation="resume")
        with pytest.raises(ResourceOutcomeUnknown):
            await h.runtime.control(context, session.session_id, request, action=VisualAction.HANDBACK, share_screen=True)
        original_get = h.repository.get
        read, continue_read = asyncio.Event(), asyncio.Event()
        retry = None
        async def held_get(*args, **kwargs):
            value = await original_get(*args, **kwargs)
            if asyncio.current_task() is retry and not read.is_set():
                read.set()
                await continue_read.wait()
            return value
        monkeypatch.setattr(h.repository, "get", held_get)
        retry = asyncio.create_task(h.runtime.control(context, session.session_id, request,
            action=VisualAction.HANDBACK, share_screen=True))
        try:
            await asyncio.wait_for(read.wait(), 5)
            assert (await h.runtime.release(context, session.session_id))["released"] is True
        finally:
            continue_read.set()
            result, = await asyncio.gather(retry, return_exceptions=True)
        assert isinstance(result, ResourceConflict), result
        assert len(observed) == 1 and not h.runtime.egress._calls
        assert h.runtime._key(h.runtime.scope(context), session.session_id) in h.runtime._fences
        assert (await original_get(h.runtime.scope(context), session_id=session.session_id)).phase == "released"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["inline", "borrowed"])
@pytest.mark.parametrize("operation", ["observe", "status", "act", "revoke"])
async def test_release_owns_human_sdk_invocation_and_cancels_native_cleanup_once(
    harness, monkeypatch, mode, operation,
):
    """Actual host/SQLite with a held SDK method, not physical browser proof."""
    h = harness
    context = h.context()
    with h.bind(context):
        session, lease, extension = await setup(h, context)
        request = await input_request(h, context, session, lease) if operation == "act" else None
        entered, closing, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
        cleanup_interruptions, physical = [], []
        async def native():
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                closing.set()
                try:
                    await finish.wait()
                except asyncio.CancelledError as error:
                    cleanup_interruptions.append(error)
                    raise
        async def sdk(*args):
            if mode == "inline":
                return await native()
            identifier = human_operation_id("human", operation,
                request.action.action_id if request is not None else None)
            guard = (h.runtime.egress.human_status_operation if operation in {"status", "revoke"}
                else h.runtime.egress.human_operation)
            async def borrowed():
                async with guard(session, identifier) as check:
                    check()
                    await native()
                    check()
            outcome = await await_owned_operation(borrowed(), cancel_on_caller_cancellation=True)
            return raise_owned_outcome(outcome, operation="controlled original SDK borrower")
        original_release = h.adapter.release
        async def release_original(scope, selected):
            physical.append(selected)
            assert finish.is_set()
            return await original_release(scope, selected)
        monkeypatch.setattr(extension, operation, sdk)
        monkeypatch.setattr(h.adapter, "release", release_original)
        if operation == "act":
            selected = h.runtime.human.act(context, session.session_id, "human", request)
        elif operation == "revoke":
            selected = h.runtime.human.revoke(context, session.session_id, "human", "original-close", session.control_epoch)
        else:
            selected = getattr(h.runtime.human, operation)(context, session.session_id, "human")
        original_call = asyncio.create_task(selected)
        release = None
        try:
            await asyncio.wait_for(entered.wait(), 5)
            release = asyncio.create_task(h.runtime.release(context, session.session_id))
            await asyncio.wait_for(closing.wait(), 5)
            assert not original_call.done() and not release.done() and not physical
            original_call.cancel()
            await asyncio.sleep(0)
            assert not cleanup_interruptions and not release.done() and not physical
        finally:
            finish.set()
            original_call.cancel()
            outcomes = await asyncio.gather(*[t for t in (original_call, release) if t is not None], return_exceptions=True)
        assert isinstance(outcomes[0], asyncio.CancelledError), outcomes
        assert outcomes[1] == {"session_id": session.session_id, "released": True}
        assert not cleanup_interruptions and len(physical) == 1 and extension.effects == 0
        assert not h.runtime.egress._calls


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["inline", "borrowed"])
async def test_human_sdk_expiry_joins_cleanup_without_second_cancellation(harness, monkeypatch, mode):
    from dataclasses import replace
    h, context = harness, harness.context()
    with h.bind(context):
        session, _, extension = await setup(h, context)
        h.runtime.policy = replace(h.runtime.policy, operation_seconds=0.2)
        entered, closing, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
        interrupted = []
        async def native():
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                closing.set()
                try:
                    await finish.wait()
                except asyncio.CancelledError as error:
                    interrupted.append(error)
                    raise
        async def sdk(*args):
            if mode == "inline":
                return await native()
            async def borrowed():
                async with h.runtime.egress.human_operation(session, human_operation_id("human", "observe")):
                    await native()
            outcome = await await_owned_operation(borrowed(), cancel_on_caller_cancellation=True)
            return raise_owned_outcome(outcome, operation="controlled expiring Human borrower")
        monkeypatch.setattr(extension, "observe", sdk)
        task = asyncio.create_task(h.runtime.human.observe(context, session.session_id, "human"))
        observed = False
        try:
            await asyncio.wait_for(entered.wait(), 5)
            await asyncio.wait_for(closing.wait(), 5)
            # Both original parent and borrower deadlines have now elapsed.
            await asyncio.sleep(0.1)
            assert not task.done() and not interrupted
            observed = True
        finally:
            finish.set()
            if not observed:
                task.cancel()
            outcome, = await asyncio.gather(task, return_exceptions=True)
        assert isinstance(outcome, TimeoutError), outcome
        assert not interrupted and not h.runtime.egress._calls


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["inline", "borrowed"])
async def test_human_sdk_cannot_acknowledge_its_own_release(harness, monkeypatch, mode):
    from kestrel_sovereign.execution_custody import execution_cleanup_debts
    h, context = harness, harness.context()
    with h.bind(context):
        session, _, extension = await setup(h, context)
        physical, owners = [], []
        async def release(*args):
            physical.append(args)
            raise AssertionError("a Human SDK invocation cannot acknowledge its own retirement")
        async def native():
            owners.append(asyncio.current_task())
            return await h.runtime.release(context, session.session_id)
        async def sdk(*args):
            if mode == "inline":
                return await native()
            async def borrowed():
                async with h.runtime.egress.human_operation(session, human_operation_id("human", "observe")):
                    return await native()
            outcome = await await_owned_operation(borrowed(), retain_originals=True)
            return raise_owned_outcome(outcome, operation="controlled self-retirement", retain_originals=True)
        monkeypatch.setattr(extension, "observe", sdk)
        monkeypatch.setattr(h.adapter, "release", release)
        with pytest.raises(BaseException) as stopped:
            await h.runtime.human.observe(context, session.session_id, "human")
        assert execution_cleanup_debts(stopped.value), stopped.value
        assert owners and all(task.done() for task in owners)
        assert not physical and not h.runtime.egress._calls


@pytest.mark.asyncio
async def test_human_sdk_owner_cannot_register_itself_as_native_borrower(harness, monkeypatch):
    h, context = harness, harness.context()
    with h.bind(context):
        session, _, extension = await setup(h, context)
        observe = extension.observe
        async def sdk(*args):
            with pytest.raises(OperatorAuthorizationError):
                async with h.runtime.egress.human_operation(session, human_operation_id("human", "observe")):
                    pytest.fail("the SDK owner was accepted as its own native borrower")
            return await observe(*args)
        monkeypatch.setattr(extension, "observe", sdk)
        result = await h.runtime.human.observe(context, session.session_id, "human")
        assert result.control_id == "human" and not h.runtime.egress._calls


@pytest.mark.asyncio
@pytest.mark.parametrize("eager", [False, True])
async def test_human_sdk_factory_accept_then_raise_joins_original_task_before_denial(
    harness, monkeypatch, eager,
):
    """Actual Task admission fault; no physical browser behavior is claimed."""
    from types import CodeType
    from kestrel_sovereign.visual_computer.egress import _ActiveCall
    from tests.unit.test_host_context_native_cleanup import _actual_operation

    if eager and not hasattr(asyncio, "eager_task_factory"):
        pytest.skip("eager native Tasks require Python 3.12 or later")
    invoke_code, = (code for code in _ActiveCall.invoke_sdk.__code__.co_consts
        if isinstance(code, CodeType) and code.co_name == "invoke")
    h, context = harness, harness.context()
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    accepted, entries = [], []
    original = RuntimeError("original SDK Task accepted before factory ACK failed")

    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        if not accepted and getattr(operation, "cr_code", None) is invoke_code:
            if eager:
                kwargs["eager_start"] = True
            accepted.append(asyncio.Task(coroutine, loop=loop, **kwargs))
            raise original
        return (previous(loop, coroutine, **kwargs) if previous
            else asyncio.Task(coroutine, loop=loop, **kwargs))

    with h.bind(context):
        session, lease, extension = await setup(h, context)
        request = await input_request(h, context, session, lease)
        act = extension.act
        async def sdk(*args):
            entries.append(args)
            return await act(*args)
        monkeypatch.setattr(extension, "act", sdk)
        try:
            loop.set_task_factory(factory)
            with pytest.raises(RuntimeError) as caught:
                await h.runtime.human.act(context, session.session_id, "human", request)
            assert caught.value is original
            # Assert before the test's final cleanup: Core must settle the
            # exact accepted Task even when its SDK owner never bound itself.
            assert len(accepted) == 1 and accepted[0].done()
            assert accepted[0].exception() is original
            assert not entries and extension.effects == 0
            assert not h.runtime.egress._calls and h.runtime._inflight == 0
        finally:
            loop.set_task_factory(previous)
            await asyncio.gather(*accepted, return_exceptions=True)
        saved = await h.repository.action(h.runtime.scope(context), session.session_id,
            request.action.action_id)
        assert saved.channel == "human" and saved.control_id == "human"
        assert saved.receipt.outcome == "denied"
        assert saved.receipt.reason_code == "host_not_dispatched"
        assert not await h.repository.has_pending_actions(h.runtime.scope(context), session.session_id)
        replay = await h.runtime.human.act(context, session.session_id, "human", request)
        assert replay.receipt == saved.receipt and not entries and extension.effects == 0
