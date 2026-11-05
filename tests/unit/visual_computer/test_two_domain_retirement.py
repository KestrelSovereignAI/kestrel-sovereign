"""Scripted daemon facts protect native wait ordering; no Docker execution."""
import json

import pytest
from kestrel_sdk.resources import ResourceConflict, ResourceNotFound, ResourceOutcomeUnknown
from kestrel_sovereign.visual_computer.local_worker import _LocalWorkerResources
from kestrel_sovereign.visual_computer.worker_profile import LABEL, LocalWorkerProfile


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_code", [0, 9])
async def test_original_container_wait_precedes_nonforce_removal_and_dependent_cleanup(exit_code):
    profile = LocalWorkerProfile("browser@sha256:"+"a"*64, "broker@sha256:"+"b"*64)
    identity, calls, removed = "c"*64, [], set()
    waited = set()
    class Daemon:
        async def inspect(self, target):
            calls.append(("inspect", target))
            if target in removed:
                raise ResourceNotFound("scripted exact absent")
            role = "browser" if target == "browser-id" else "broker"
            finished = target in waited
            return {"Id": target, "Image": profile.image(role),
                "Config": {"Labels": {LABEL: identity}},
                "State": {"Running": not finished, "Pid": 0 if finished else 123,
                          "ExitCode": exit_code if finished else 0, "Status": "exited" if finished else "running"}}
        async def invoke(self, argv, *, timeout=None):
            calls.append(tuple(argv))
            assert "--force" not in argv
            if argv[0] == "wait":
                assert timeout == 50, "Host must let the browser watchdog and original joins settle"
                waited.add(argv[1])
                return str(exit_code).encode()+b"\n"
            if argv[:2] == ["container", "rm"]:
                assert argv[-1] in waited
                removed.add(argv[-1])
            if argv[:2] in (["network", "inspect"], ["volume", "inspect"]):
                if argv[-1] in removed:
                    raise ResourceNotFound("scripted exact absent")
                return json.dumps([{"Labels": {LABEL: identity}}]).encode()
            if argv[:2] in (["network", "rm"], ["volume", "rm"]):
                assert removed >= {"browser-id", "broker-id"}
                removed.add(argv[-1])
            return b""
    resource = object.__new__(_LocalWorkerResources)
    resource.profile, resource.identity, resource.boundary = profile, identity, Daemon()
    resource.containers = {"browser": "browser-id", "broker": "broker-id"}
    resource._native_instances = {"browser": (1, "original"), "broker": (2, "original")}
    resource._container_waits = {}
    resource.network, resource.volume, resource._closed = "original-net", "original-volume", False
    resource._network_identity = resource._volume_identity = None
    if exit_code == 0:
        await resource._retire()
        assert resource._closed
        assert removed == {"browser-id", "broker-id", "original-net", "original-volume"}
        for cid in ("browser-id", "broker-id"):
            assert calls.index(("kill", "--signal=SIGTERM", cid)) < calls.index(("wait", cid)) < calls.index(("container", "rm", cid))
    else:
        with pytest.raises(ResourceOutcomeUnknown):
            await resource._retire()
        assert not resource._closed and removed == set()
        assert not any(call[:2] in (("network", "rm"), ("volume", "rm")) for call in calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["inspect-disappeared", "wait-disappeared", "different-id"])
async def test_recorded_original_instance_requires_its_exact_native_wait(failure):
    """Scripted daemon loss is UNKNOWN, never a native retirement ACK."""
    profile = LocalWorkerProfile("browser@sha256:"+"a"*64, "broker@sha256:"+"b"*64)
    identity, calls = "c"*64, []
    class Daemon:
        async def inspect(self, target):
            calls.append(("inspect", target))
            if failure == "inspect-disappeared":
                raise ResourceNotFound("scripted retained CID disappeared")
            return {"Id": "replacement-id" if failure == "different-id" else target,
                "Image": profile.image("browser"), "Config": {"Labels": {LABEL: identity}},
                "State": {"Running": True, "Pid": 123, "StartedAt": "original"}}
        async def invoke(self, argv, *, timeout=None):
            calls.append(tuple(argv))
            if argv[0] == "wait":
                raise ResourceNotFound("scripted lost native wait")
            return b""
    resource = object.__new__(_LocalWorkerResources)
    resource.profile, resource.identity, resource.boundary = profile, identity, Daemon()
    resource.containers = {"browser": "original-id"}
    resource._native_instances = {"browser": (123, "original")}
    resource._container_waits = {}
    resource.network = resource.volume = None
    resource._closed = False
    resource._network_identity = resource._volume_identity = None
    with pytest.raises(ResourceConflict if failure == "different-id" else ResourceOutcomeUnknown):
        await resource._retire()
    assert not resource._closed
    assert not any(call[:2] == ("container", "rm") for call in calls)
    assert not any("replacement-id" in call for call in calls if call[0] != "inspect")


# Actual Runtime/Host/SQLite; canonical Adapter remains SCRIPTED. These tests
# make no claim about Docker/native physical retirement or worker registration.
import asyncio
from datetime import UTC, datetime, timedelta
from kestrel_sdk.computer_use import ControlRequest
from kestrel_sovereign._async_ownership import OwnedTaskAdmission, await_owned_task, create_owned_task
from kestrel_sovereign.visual_computer.runtime import VisualAction


async def _release_task_boundary(task, original_lock, provider_entered=None):
    """Passive same-Task/same-Lock observation, never a join/authority ACK."""
    async def observe():
        while True:
            if provider_entered is not None and provider_entered.is_set():
                return "second-provider-entered"
            cursor = task.get_coro()
            for _ in range(32):
                if cursor is None:
                    break
                frame = getattr(cursor, "cr_frame", None)
                if (getattr(cursor, "cr_code", None) is asyncio.Lock.acquire.__code__
                        and frame is not None and frame.f_locals.get("self") is original_lock):
                    return "same-original-lock-queued"
                cursor = getattr(cursor, "cr_await", getattr(cursor, "gi_yieldfrom", None))
            if task.done():
                return "original-task-returned"  # observation only, actually joined below
            await asyncio.sleep(0)
    return await asyncio.wait_for(observe(), 5)


async def _join_release_tasks(admissions):
    outcomes, join_errors = [], []
    for admission in admissions:
        outcome = None
        if admission.task is not None:
            try:
                outcome = await await_owned_task(admission.task)
            except BaseException as error:
                join_errors.append(error)  # unknown join is never an ACK
        outcomes.append(outcome)
    return outcomes, join_errors


def _raise_release_errors(primary, outcomes, extra_errors=()):
    errors = [] if primary is None else [primary]
    for outcome in outcomes:
        if outcome is not None:
            for error in (outcome.error, outcome.cancellation):
                if error is not None and not any(error is old for old in errors):
                    errors.append(error)
    for error in extra_errors:
        if not any(error is old for old in errors):
            errors.append(error)
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise BaseExceptionGroup("original release test operations", errors)


@pytest.mark.asyncio
@pytest.mark.parametrize("ordering", ["foreground-first", "maintenance-first", "two-foreground"])
async def test_release_serializes_original_owners_and_reloads_terminal_row(harness, monkeypatch, ordering):
    h = harness
    context = h.context()
    now = datetime.now(UTC)  # later than this genuine context's issued_at
    clock = [now]
    monkeypatch.setattr(h.runtime.authority, "clock", lambda: clock[0])
    first, second, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
    admissions = (OwnedTaskAdmission(), OwnedTaskAdmission())
    provider_entries = []
    original_release = h.adapter.release
    primary, boundary, queued, final = None, None, None, None
    with h.bind(context):
        session = await h.runtime.admit(context, h.request(context, requested_at=now,
            expires_at=now+timedelta(seconds=30), idle_ttl_seconds=10))
        # The real admitted fixed lease is now expired; original context is
        # still fresh. This also makes the terminal-row skip load-bearing.
        clock[0] = now+timedelta(seconds=31)
        scope = h.runtime.scope(context)
        lock = h.runtime._lock(session.session_id)

        async def gated_original_release(actual_scope, lease):
            row = await h.repository.get(actual_scope, session_id=session.session_id)
            provider_entries.append({"revision": row.revision, "phase": row.phase})
            (first if len(provider_entries) == 1 else second).set()
            await finish.wait()
            return await original_release(actual_scope, lease)

        monkeypatch.setattr(h.adapter, "release", gated_original_release)
        try:
            first_operation = h.runtime.maintain_owned_sessions(scope) if ordering == "maintenance-first" else h.runtime.release(context, session.session_id)
            create_owned_task(first_operation, admission=admissions[0], name="original-first-release-owner")
            await asyncio.wait_for(first.wait(), 5)
            second_operation = h.runtime.maintain_owned_sessions(scope) if ordering == "foreground-first" else h.runtime.release(context, session.session_id)
            create_owned_task(second_operation, admission=admissions[1], name="original-queued-release-owner")
            boundary = await _release_task_boundary(admissions[1].task, lock, second)
            row = await h.repository.get(scope, session_id=session.session_id)
            queued = {"phase": row.phase, "revision": row.revision}
        except BaseException as error:
            primary = error
        finally:
            finish.set()
            outcomes, join_errors = await _join_release_tasks(admissions)
        diagnostic_errors = list(join_errors)
        try:
            final = await h.repository.get(scope, session_id=session.session_id)
        except BaseException as error:
            diagnostic_errors.append(error)
    _raise_release_errors(primary, outcomes, diagnostic_errors)
    assert boundary == "same-original-lock-queued"
    assert len(provider_entries) == 1  # no duplicate provider close after original ACK
    assert queued == provider_entries[0]
    released = {"session_id": session.session_id, "released": True}
    assert outcomes[0].result == ([] if ordering == "maintenance-first" else released)
    assert outcomes[1].result == ([] if ordering == "foreground-first" else released)
    assert final.phase == "released" and final.lease.state == "released"


@pytest.mark.asyncio
async def test_Stop_remains_priority_while_release_waits_original_session_lock(harness):
    h = harness
    context = h.context()
    admissions = (OwnedTaskAdmission(), OwnedTaskAdmission())
    primary, boundary, stopped, final = None, None, None, None
    with h.bind(context):
        session = await h.runtime.admit(context, h.request(context))
        lock = h.runtime._lock(session.session_id)
        await lock.acquire()  # test owns only this timing primitive, no new grant
        try:
            create_owned_task(h.runtime.release(context, session.session_id), admission=admissions[0], name="original-waiting-release")
            boundary = await _release_task_boundary(admissions[0].task, lock)
            request = ControlRequest(request_id="priority-stop-during-release-queue", session_id=session.session_id,
                expected_control_epoch=session.control_epoch, operation="pause")
            create_owned_task(h.runtime.control(context, session.session_id, request, action=VisualAction.STOP),
                admission=admissions[1], name="original-priority-Stop")
            # Shield only this observation deadline; always join the SAME
            # retained Stop Task after unlocking, even if this wait fails.
            stopped = await asyncio.wait_for(asyncio.shield(admissions[1].task), 5)
        except BaseException as error:
            primary = error
        finally:
            lock.release()  # release only the lock acquired by this test
            outcomes, join_errors = await _join_release_tasks(admissions)
        diagnostic_errors = list(join_errors)
        try:
            final = await h.repository.get(h.runtime.scope(context), session_id=session.session_id)
        except BaseException as error:
            diagnostic_errors.append(error)
    _raise_release_errors(primary, outcomes, diagnostic_errors)
    assert boundary == "same-original-lock-queued"
    assert stopped.controller == "none" and stopped.control_epoch == 1
    assert outcomes[0].result == {"session_id": session.session_id, "released": True}
    assert final.phase == "released" and final.session.control_epoch == 1 and final.session.controller == "none"


@pytest.mark.asyncio
async def test_cancelled_release_waiter_has_no_provider_effect_and_fresh_release_remains_possible(harness, monkeypatch):
    h = harness
    context = h.context()
    admission = OwnedTaskAdmission()
    original_release = h.adapter.release
    provider_calls = []
    primary, boundary, outcome, fresh = None, None, None, None
    with h.bind(context):
        session = await h.runtime.admit(context, h.request(context))
        lock = h.runtime._lock(session.session_id)

        async def counted_original_release(scope, lease):
            provider_calls.append(lease.lease_id)
            return await original_release(scope, lease)

        monkeypatch.setattr(h.adapter, "release", counted_original_release)
        await lock.acquire()
        try:
            create_owned_task(h.runtime.release(context, session.session_id), admission=admission, name="original-cancelled-release-waiter")
            boundary = await _release_task_boundary(admission.task, lock)
            admission.task.cancel()  # explicit withdrawal of this original waiter
            outcome = await await_owned_task(admission.task)
        except BaseException as error:
            primary = error
        finally:
            lock.release()
            outcomes, join_errors = await _join_release_tasks((admission,))
            if outcomes[0] is not None:
                outcome = outcomes[0]
        diagnostic_errors = list(join_errors)
        before_fresh = None
        try:
            before_fresh = await h.repository.get(h.runtime.scope(context), session_id=session.session_id)
        except BaseException as error:
            diagnostic_errors.append(error)
        calls_before_fresh = len(provider_calls)
        cancel_observed = outcome is not None and (isinstance(outcome.error, asyncio.CancelledError)
            or isinstance(outcome.cancellation, asyncio.CancelledError))
        unexpected_errors = [] if outcome is None else [error for error in (outcome.error, outcome.cancellation)
            if error is not None and not isinstance(error, asyncio.CancelledError)]
        if primary is None and not diagnostic_errors and cancel_observed and not unexpected_errors:
            fresh = await h.runtime.release(context, session.session_id)
    if primary is not None or diagnostic_errors or unexpected_errors or not cancel_observed:
        _raise_release_errors(primary, (outcome,), diagnostic_errors)
    assert boundary == "same-original-lock-queued"
    assert cancel_observed
    assert calls_before_fresh == 0 and before_fresh.phase == "active" and before_fresh.lease.state == "ready"
    assert fresh == {"session_id": session.session_id, "released": True} and len(provider_calls) == 1
