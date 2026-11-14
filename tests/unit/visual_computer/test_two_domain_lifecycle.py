"""Lifecycle caller regression with real Host/Runtime/driver/SQLite.

Only native construction, live-inspection and transport I/O are controlled
doubles. Session/control values come from the actual installed canonical
BrowserController, never a fabricated ComputerSession/ready DTO. This is not
Docker/TLS/browser/native-retirement conformance.
"""
import asyncio
import hmac
from importlib import util
import json
from types import SimpleNamespace

import pytest
import pytest_asyncio
from kestrel_sdk.computer_use import ComputerSession, ControlRequest
from kestrel_sdk.operator import OperatorAuthorizationError
from kestrel_sdk.resources import ResourceOutcomeUnknown
from kestrel_sovereign.execution_custody import execution_terminal_error
from kestrel_sovereign.visual_computer.local_worker import _LocalWorkerResources
from kestrel_sovereign.visual_computer.worker_profile import DRIVER
from tests.unit.visual_computer.test_local_two_domain_admit import local_setup as local_setup
from tests.shared.installed_worker_requirement import worker_program


def original_installed_controller():
    # Intentionally installed-only: no source fallback or sys.modules overlay.
    path = worker_program("controller.py")
    spec = util.spec_from_file_location("_lifecycle_original_controller", str(path))
    assert spec is not None and spec.loader is not None
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BrowserController


@pytest_asyncio.fixture
async def lifecycle_transport(local_setup, monkeypatch):
    setup = local_setup
    state = {"resources": [], "exchanges": [], "driver_errors": [], "page_closed": 0}
    Controller = original_installed_controller()

    class ControlledPage:
        def __init__(self):
            self.context = SimpleNamespace(close=self.close)
        async def close(self):
            state["page_closed"] += 1
        def __getattr__(self, name):
            raise AssertionError("lifecycle test attempted browser I/O: " + name)

    async def create(resource):
        state["resources"].append(resource)
        # Explicit descriptive test data, not daemon/native process facts.
        resource.containers = {"browser": "a" * 64, "broker": "b" * 64}
        resource.incarnations = {"browser": "c" * 64, "broker": "d" * 64}

    async def live(resource, current):
        current()  # Preserve the SAME original checker.
        if resource._closing or resource._closed:
            raise ResourceOutcomeUnknown("controlled_worker_already_retired")

    class MemoryPriorityTransport:
        def __init__(self, resource, role):
            self.resource, self.role = resource, role
        async def exchange(self, raw, *, admission, timeout_seconds):
            async with admission as current:
                current()
                message = json.loads(raw)
                assert message["schema_version"] == 1
                assert hmac.compare_digest(
                    message["secret"],
                    self.resource._connection.credential.get_secret_value(),
                )
                assert self.role == "priority"
                operation, arguments = message["operation"], message["arguments"]
                borrowed = any(
                    key[-1] == "session"
                    and asyncio.current_task() in call.jobs
                    for key, call in setup.runtime.egress._calls.items()
                    if key[-1] == "session"
                )
                state["exchanges"].append((operation, borrowed))
                if operation == "status":
                    value = self.resource.controller.status(**arguments)
                elif operation == "control":
                    value = await self.resource.controller.control(**arguments)
                else:
                    raise AssertionError("unexpected lifecycle transport operation")
                current()
                return json.dumps(
                    {"schema_version": 1, "id": message["id"], "result": value}
                ).encode() + b"\n"

    async def bootstrap(resource, request, credential):
        assert credential == resource._connection.credential.get_secret_value()
        # The unchanged canonical controller creates/returns its own session.
        resource.controller = Controller(
            ControlledPage(), request.model_dump(mode="json"), driver_id=DRIVER
        )
        resource._tls = {
            role: MemoryPriorityTransport(resource, role)
            for role in ("normal", "priority")
        }

    async def retire(resource):
        # Only a controlled page/context is retired here. This is no native ACK.
        assert await resource.controller.retire() is True
        resource._closed = True

    original_control = setup.driver.control
    async def observe_original_control(session, request):
        # Retain the original exception before Runtime's existing OSError catch.
        # The bound original driver and its borrow guard are never replaced.
        try:
            return await original_control(session, request)
        except BaseException as error:
            state["driver_errors"].append(error)
            raise

    monkeypatch.setattr(_LocalWorkerResources, "create", create)
    monkeypatch.setattr(_LocalWorkerResources, "verify_live", live)
    monkeypatch.setattr(_LocalWorkerResources, "bootstrap", bootstrap)
    monkeypatch.setattr(_LocalWorkerResources, "_retire", retire)
    monkeypatch.setattr(setup.driver, "control", observe_original_control)
    try:
        yield setup, state
    finally:
        # Fixture teardown remains distinct from any original assertion failure.
        # Attempt every retained original closer, preserving companion errors.
        errors = []
        for resource in state["resources"]:
            try:
                session_id = resource.original._record.session_id
                if setup.driver._connections.get(session_id) is resource:
                    # Genuine original Host RELEASE while Feature is still live;
                    # parent local_setup.shutdown then sees a terminal row.
                    async with setup.bound() as context:
                        await setup.runtime.release(context, session_id)
            except BaseException as error:
                errors.append(error)
            finally:
                try:
                    await resource.close()  # Always join the SAME original closer.
                except BaseException as error:
                    errors.append(error)
        if errors:
            raise execution_terminal_error(None, *errors, retain_originals=True)


async def admit_original(setup):
    async with setup.bound() as context:
        session = await setup.runtime.admit(
            context, setup.request(context, required_capabilities=frozenset({"screenshot"}))
        )
    assert setup.runtime.egress._calls == {}
    return session


@pytest.mark.asyncio
async def test_original_lifecycle_pause_reaches_priority_controller(lifecycle_transport):
    setup, state = lifecycle_transport
    session = await admit_original(setup)
    resource, = state["resources"]
    assert state["exchanges"] == [("status", False)]
    assert ComputerSession.model_validate_json(json.dumps(resource.controller.status(
        session.scope.model_dump(mode="json"), session.session_id
    ))) == session

    # No caller/OperatorContext is active. This invokes the unchanged genuine
    # lifecycle boundary used by Feature._withdraw, not a foreground Stop.
    uncertain = await setup.runtime.fence_owned_sessions(session.scope)
    saved = await setup.runtime.repository.get(session.scope, session_id=session.session_id)
    errors = [(type(error).__name__, str(error)) for error in state["driver_errors"]]
    assert not uncertain, (
        "lifecycle priority pause was not delivered; "
        f"original_driver_errors={errors!r}; exchanges={state['exchanges']!r}"
    )
    assert state["exchanges"] == [("status", False), ("control", True)]
    actual = resource.controller.status(session.scope.model_dump(mode="json"), session.session_id)
    assert actual["controller"] == "none" and actual["control_epoch"] == session.control_epoch + 1
    assert actual["state"] == "paused"
    assert saved.phase == "fenced" and saved.pending_control is None
    assert saved.session == ComputerSession.model_validate_json(json.dumps(actual))
    assert saved.lease.state == "ready" and saved.connection is not None
    assert setup.runtime.egress._calls == {}


@pytest.mark.asyncio
async def test_unborrowed_driver_pause_still_refuses(lifecycle_transport):
    setup, state = lifecycle_transport
    session = await admit_original(setup)
    request = ControlRequest(
        request_id="unborrowed-pause", session_id=session.session_id,
        expected_control_epoch=session.control_epoch, operation="pause",
    )
    with pytest.raises(OperatorAuthorizationError, match="original_session_borrow_required"):
        await setup.driver.control(session, request)
    assert state["exchanges"] == [("status", False)]
    resource, = state["resources"]
    actual = resource.controller.status(session.scope.model_dump(mode="json"), session.session_id)
    assert actual["controller"] == "agent" and actual["control_epoch"] == 0
    saved = await setup.runtime.repository.get(session.scope, session_id=session.session_id)
    assert saved.phase == "active" and saved.pending_control is None
    assert setup.runtime.egress._calls == {}


# These guard tests use the same real Host/Runtime/driver/SQLite fixture above;
# only the original controlled transport/SQL interleavings are scheduled here.
from datetime import timedelta
from kestrel_sovereign._async_ownership import OwnedTaskAdmission, create_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain
from kestrel_sovereign.visual_computer.runtime import VisualAction
from tests.unit.visual_computer.test_two_domain_retirement import (
    _join_release_tasks, _raise_release_errors,
)


@pytest.mark.asyncio
async def test_superseding_Stop_during_lifecycle_drain_cannot_be_adopted(lifecycle_transport, monkeypatch):
    setup, state = lifecycle_transport
    session = await admit_original(setup)
    revoke = setup.runtime.egress.revoke_session
    fired, stopped = False, None
    request = ControlRequest(request_id="priority-Stop-during-lifecycle-drain",
        session_id=session.session_id, expected_control_epoch=0, operation="pause")
    async with setup.bound() as context:
        async def superseding_Stop(scope, session_id):
            nonlocal fired, stopped
            await revoke(scope, session_id)
            if not fired and session_id == session.session_id:
                fired = True
                # Genuine current Host STOP, not a manufactured serial/check.
                stopped = await setup.runtime.control(context, session_id, request, action=VisualAction.STOP)
        monkeypatch.setattr(setup.runtime.egress, "revoke_session", superseding_Stop)
        uncertain = await setup.runtime.fence_owned_sessions(session.scope)
    saved = await setup.runtime.repository.get(session.scope, session_id=session.session_id)
    assert fired and stopped.controller == "none" and stopped.control_epoch == 1
    assert uncertain == [session.session_id]
    assert state["exchanges"] == [("status", False), ("control", True)]
    assert saved.session == stopped and saved.pending_control is None
    assert saved.last_control == request
    assert setup.runtime.egress._calls == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["pending-pause", "revision"])
async def test_lifecycle_rereads_changed_original_SQL_custody_before_driver(lifecycle_transport, monkeypatch, changed):
    setup, state = lifecycle_transport
    session = await admit_original(setup)
    get, save = setup.runtime.repository.get, setup.runtime.repository.save
    fired, written = False, None
    async def changed_original_row(*args, **kwargs):
        nonlocal fired, written
        row = await get(*args, **kwargs)
        borrowed = any(key[-1] == "session" and asyncio.current_task() in call.jobs
                       for key, call in setup.runtime.egress._calls.items() if key[-1] == "session")
        if not fired and borrowed and row.session_id == session.session_id and row.pending_control is not None:
            fired = True
            # Persist a real independently revised row through canonical SQL/CAS,
            # never return a fabricated record or replace the original guard.
            if changed == "pending-pause":
                row.pending_control = row.pending_control.model_copy(update={"request_id": "other-original-pause"})
            else:
                row.last_activity_at += timedelta(microseconds=1)
            await save(args[0], row, kwargs["guard"])
            written = row
            return await get(*args, **kwargs)
        return row
    monkeypatch.setattr(setup.runtime.repository, "get", changed_original_row)
    uncertain = await setup.runtime.fence_owned_sessions(session.scope)
    saved = await get(session.scope, session_id=session.session_id)
    resource, = state["resources"]
    actual = resource.controller.status(session.scope.model_dump(mode="json"), session.session_id)
    assert fired and uncertain == [session.session_id]
    assert state["exchanges"] == [("status", False)]
    assert actual["controller"] == "agent" and actual["control_epoch"] == 0
    assert saved == written and saved.pending_control is not None
    assert setup.runtime.egress._calls == {}


@pytest.mark.asyncio
async def test_lifecycle_caller_withdrawal_joins_original_transport_cleanup(lifecycle_transport, monkeypatch):
    setup, state = lifecycle_transport
    session = await admit_original(setup)
    resource, = state["resources"]
    transport = resource._tls["priority"]
    exchange = transport.exchange
    entered, cleanup_entered = asyncio.Event(), asyncio.Event()
    finish, finish_cleanup = asyncio.Event(), asyncio.Event()
    native_tasks = []
    async def held_original_exchange(raw, **kwargs):
        if json.loads(raw)["operation"] == "control":
            native_tasks.append(asyncio.current_task())
            entered.set()
            try:
                await finish.wait()
            finally:
                cleanup_entered.set()
                await finish_cleanup.wait()
        return await exchange(raw, **kwargs)
    monkeypatch.setattr(transport, "exchange", held_original_exchange)
    admission = OwnedTaskAdmission()
    primary, outcomes, join_errors = None, [], []
    retained_before_join = False
    try:
        create_owned_task(setup.runtime.fence_owned_sessions(session.scope),
            admission=admission, name="original-lifecycle-caller-withdrawal")
        await asyncio.wait_for(entered.wait(), 5)
        admission.task.cancel()  # Actual withdrawal of the retained original caller.
        await asyncio.wait_for(cleanup_entered.wait(), 5)
        # Observation only; ownership is acknowledged by actual joins below.
        retained_before_join = not admission.task.done() and bool(setup.runtime.egress._calls)
    except BaseException as error:
        primary = error
    finally:
        finish.set()
        finish_cleanup.set()
        outcomes, join_errors = await _join_release_tasks((admission,))
    outcome = outcomes[0] if outcomes else None
    original_errors = [] if outcome is None else [e for e in (outcome.error, outcome.cancellation) if e is not None]
    cancel_observed = any(isinstance(e, asyncio.CancelledError) for e in original_errors)
    unexpected = [node for error in original_errors for node in _explicit_control_chain(error)
                  if not isinstance(node, (asyncio.CancelledError, BaseExceptionGroup))]
    if primary is not None or join_errors or unexpected or not cancel_observed:
        _raise_release_errors(primary, outcomes, join_errors)
        assert cancel_observed, "original lifecycle caller cancellation was not retained"
    assert retained_before_join
    assert len(native_tasks) == 1
    assert state["exchanges"] == [("status", False)]  # Cancelled before any controlled pause dispatch.
    assert setup.runtime.egress._calls == {}
