"""Measured stopped-signal diagnostic; SCRIPTED daemon, real original closer Task.

The exact supported daemon diagnostic comes from a separately accepted utility.
This module does not run Docker or prove a native worker race/retirement ACK.
It imports unchanged canonical ownership-test helpers; no canonical prefix is
copied, replaced or given a false Source21 origin.
"""
import copy
import json
from types import SimpleNamespace

import pytest
from kestrel_sdk.resources import ResourceNotFound, ResourceOutcomeUnknown
from kestrel_sovereign._async_ownership import OwnedTaskAdmission
from kestrel_sovereign.execution_custody import _explicit_control_chain
from kestrel_sovereign.visual_computer.docker import DockerCommandError
from kestrel_sovereign.visual_computer.local_worker import _LocalWorkerResources
from kestrel_sovereign.visual_computer.worker_profile import LABEL, LocalWorkerProfile
from tests.unit.visual_computer.test_two_domain_retirement import (
    _join_release_tasks, _raise_release_errors,
)

# Actual final inspect State projection from the accepted stopped utility.
# Identity/Image/labels/PID-before-signal below are explicitly SCRIPTED bindings.
MEASURED_FINISHED_STATE = {
    "Status": "exited", "Running": False, "Paused": False, "Restarting": False,
    "OOMKilled": False, "Dead": False, "Pid": 0, "ExitCode": 0, "Error": "",
    "StartedAt": "2026-11-05T07:26:19.806109801Z",
    "FinishedAt": "2026-11-05T07:26:19.862609301Z",
}


async def original_closer_signal_case(kind, tmp_path, *, replacement=None):
    profile = LocalWorkerProfile("browser@sha256:" + "a" * 64, "broker@sha256:" + "b" * 64)
    identity, cid = "c" * 64, "d" * 64
    if kind == "measured-stopped":
        diagnostic = (
            b"Error response from daemon: cannot kill container: " + cid.encode()
            + b": container " + cid.encode() + b" is not running\n"
        )
    else:
        # UNMEASURED negative: this permission diagnostic is scripted test DATA.
        # Even a coincident exited/zero state must not turn arbitrary denial
        # into permission to wait/remove or a successful retirement receipt.
        diagnostic = b"SCRIPTED UNMEASURED permission denied for original container " + cid.encode() + b"\n"
    signal_error = DockerCommandError("docker_operation_outcome_unknown", diagnostics=diagnostic)
    calls, removed = [], set()

    class Daemon:
        def __init__(self):
            self.signal_attempted = False
            self.wait_returned = False
        async def inspect(self, target):
            assert target == cid
            calls.append(("inspect", target))
            if target in removed:
                raise ResourceNotFound("scripted exact original absent")
            if replacement == "missing-after-wait" and self.wait_returned:
                raise ResourceNotFound("SCRIPTED original incarnation unavailable after wait")
            state = copy.deepcopy(MEASURED_FINISHED_STATE)
            if not self.signal_attempted and kind != "normal-stopped" and replacement != "stopped-before":
                state.update(Status="running", Running=True, Pid=123,
                             FinishedAt="0001-01-01T00:00:00Z")
            if (replacement in {"running-before", "stopped-before"}
                    or replacement == "after-signal" and self.signal_attempted
                    or replacement == "after-wait" and self.wait_returned):
                state["StartedAt"] = "2026-11-05T07:26:20.806109801Z"
                if state["Running"]:
                    state["Pid"] = 456
            if replacement == "running-pid-before" and not self.signal_attempted:
                state["Pid"] = 456
            return {"Id": cid, "Image": profile.image("browser"),
                    "Config": {"Labels": {LABEL: identity}}, "State": state}
        async def invoke(self, argv, *, timeout=None):
            calls.append(tuple(argv))
            assert "--force" not in argv
            if argv == ["kill", "--signal=SIGTERM", cid]:
                self.signal_attempted = True  # Natural exit wins this scripted interleave.
                if kind == "normal":
                    return b""
                raise signal_error
            if argv == ["wait", cid]:
                assert self.signal_attempted or kind == "normal-stopped" or replacement == "stopped-before"
                self.wait_returned = True
                return b"0\n"  # SCRIPTED data; not a native container wait ACK.
            if argv == ["container", "rm", cid]:
                assert ("wait", cid) in calls
                removed.add(cid)
                return b""
            raise AssertionError("unexpected scripted retirement command")

    resource = object.__new__(_LocalWorkerResources)
    resource.profile, resource.identity, resource.boundary = profile, identity, Daemon()
    resource.containers = {"browser": cid}
    resource._native_instances = {"browser": (123, MEASURED_FINISHED_STATE["StartedAt"])}
    resource._container_waits = {}
    resource.network = resource.volume = None
    resource._network_identity = resource._volume_identity = None
    resource._pending_creates = set()
    resource._closed = resource._closing = False
    resource._binding = resource
    resource.original = SimpleNamespace(_resources=resource)
    admission = resource._close_admission = resource._original_close_admission = OwnedTaskAdmission()
    errors, tasks, outcomes, join_errors = [], [], [], []
    for _ in range(2):
        try:
            await resource.close()
        except BaseException as error:
            errors.append(error)
        finally:
            tasks.append(admission.task)
            actual, failures = await _join_release_tasks((admission,))
            outcomes.extend(actual)
            join_errors.extend(failures)
    # Every exact original join was attempted before any behavioral assertion.
    _raise_release_errors(None, (), join_errors)
    for outcome in outcomes:
        if outcome is not None and outcome.cancellation is not None:
            _raise_release_errors(None, (outcome,))
    assert tasks[0] is not None and all(task is tasks[0] for task in tasks)
    assert all(outcome is not None for outcome in outcomes)
    retained = any(any(node is signal_error for node in _explicit_control_chain(error)) for error in errors)
    fact = {"scope": "SCRIPTED_DAEMON_REAL_CANONICAL_CLOSER_TASK_ONLY",
            "kind": kind, "replacement": replacement, "calls": calls, "closed": resource._closed,
            "removed": sorted(removed), "original_signal_error_retained": retained,
            "same_original_task": all(task is tasks[0] for task in tasks),
            "actual_canonical_owned_joins_returned": len(outcomes),
            "native_Docker_or_container_wait_proved": False}
    (tmp_path / "signal-case-facts.json").write_text(json.dumps(fact, indent=2))
    assert calls.count(("kill", "--signal=SIGTERM", cid)) <= 1
    return resource, cid, calls, removed, errors, outcomes, retained


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["normal", "normal-stopped"])
async def test_normal_retirement_confirms_the_retained_original_incarnation(tmp_path, kind):
    resource, cid, calls, removed, errors, outcomes, retained = await original_closer_signal_case(
        kind, tmp_path,
    )
    assert not errors and not retained
    assert resource._closed and removed == {cid} and resource._container_waits == {cid: 0}
    assert calls.count(("wait", cid)) == 1
    assert calls.count(("kill", "--signal=SIGTERM", cid)) == (kind == "normal")
    assert all(outcome.error is None for outcome in outcomes)


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", [
    "running-before", "stopped-before", "running-pid-before", "after-signal", "after-wait",
    "missing-after-wait",
])
async def test_normal_retirement_rejects_a_replacement_incarnation(tmp_path, replacement):
    resource, cid, calls, removed, errors, outcomes, retained = await original_closer_signal_case(
        "normal", tmp_path, replacement=replacement,
    )
    assert errors, "replacement exit must never settle the retained original's retirement"
    reason = ("original_worker_disappeared_without_native_wait" if replacement == "missing-after-wait"
              else "original_worker_retirement_incarnation_differs")
    assert all(any(type(node) is ResourceOutcomeUnknown
                   and node.args == (reason,)
                   for node in _explicit_control_chain(error)) for error in errors)
    assert not resource._closed and not removed and not retained
    assert all(outcome.error is not None for outcome in outcomes)
    assert ("container", "rm", cid) not in calls
    assert resource._container_waits == {}, "an unconfirmed incarnation cannot supply an original retirement ACK"
    if replacement.endswith("before"):
        assert ("kill", "--signal=SIGTERM", cid) not in calls
        assert ("wait", cid) not in calls and resource._container_waits == {}
    else:
        assert calls.count(("kill", "--signal=SIGTERM", cid)) == 1
        assert calls.count(("wait", cid)) == 1


@pytest.mark.asyncio
async def test_measured_stopped_signal_drains_original_zero_exit_without_resend(tmp_path):
    resource, cid, calls, removed, errors, outcomes, retained = await original_closer_signal_case(
        "measured-stopped", tmp_path
    )
    # Before fix this is a real behavioral failure after BOTH original joins;
    # preserve the original cached DockerCommandError as the explicit cause.
    try:
        assert not errors, "exact measured stopped-signal refusal must drain original zero exit"
        assert calls.count(("kill", "--signal=SIGTERM", cid)) == 1
        assert resource._closed and removed == {cid}
        assert resource._container_waits == {cid: 0}
        assert calls.count(("wait", cid)) == 1
        assert calls.index(("wait", cid)) < calls.index(("container", "rm", cid))
        assert calls[-1] == ("inspect", cid)  # Exact absence is checked after nonforce removal.
        assert all(outcome.error is None for outcome in outcomes)
    except AssertionError as oracle:
        raise oracle from errors[0] if errors else None


@pytest.mark.asyncio
async def test_unmeasured_permission_signal_keeps_original_error_and_debt(tmp_path):
    resource, cid, calls, removed, errors, outcomes, retained = await original_closer_signal_case(
        "unmeasured-permission", tmp_path
    )
    assert errors and retained
    assert not resource._closed and not removed and resource._container_waits == {}
    assert ("wait", cid) not in calls and ("container", "rm", cid) not in calls
    assert all(outcome.error is not None for outcome in outcomes)


# All cases below are SCRIPTED adversarial interleavings, not daemon measurements.
@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [
    "wrong-reason", "foreign-diagnostic-cid", "extra-diagnostic", "foreign-id",
    "foreign-label", "foreign-image", "changed-start", "running", "nonzero-exit",
    "oom", "dead", "error", "pid", "missing-reinspection", "nonzero-wait",
    "failed-wait", "changed-after-wait",
])
async def test_stopped_signal_does_not_erase_other_original_failures(tmp_path, fault):
    profile = LocalWorkerProfile("browser@sha256:" + "a" * 64, "broker@sha256:" + "b" * 64)
    identity, cid = "c" * 64, "d" * 64
    diagnostic_cid = "e" * 64 if fault == "foreign-diagnostic-cid" else cid
    diagnostic = (b"Error response from daemon: cannot kill container: " + diagnostic_cid.encode()
        + b": container " + diagnostic_cid.encode() + b" is not running\n")
    if fault == "extra-diagnostic":
        diagnostic += b"SCRIPTED extra diagnostic\n"
    signal_error = DockerCommandError(
        "SCRIPTED other reason" if fault == "wrong-reason" else "docker_operation_outcome_unknown",
        diagnostics=diagnostic,
    )
    wait_error = DockerCommandError("SCRIPTED original wait failure", diagnostics=b"SCRIPTED wait failed")
    calls, removed = [], set()
    class Daemon:
        signal_attempted = False
        wait_returned = False
        async def inspect(self, target):
            assert target == cid
            calls.append(("inspect", target))
            state = copy.deepcopy(MEASURED_FINISHED_STATE)
            row = {"Id": cid, "Image": profile.image("browser"),
                   "Config": {"Labels": {LABEL: identity}}, "State": state}
            if not self.signal_attempted:
                state.update(Status="running", Running=True, Pid=123,
                             FinishedAt="0001-01-01T00:00:00Z")
                return row
            if fault == "missing-reinspection":
                raise ResourceNotFound("SCRIPTED original reinspection lost")
            if fault == "foreign-id": row["Id"] = "e" * 64
            elif fault == "foreign-label": row["Config"]["Labels"][LABEL] = "e" * 64
            elif fault == "foreign-image": row["Image"] = profile.image("broker")
            elif fault == "changed-start" or (fault == "changed-after-wait" and self.wait_returned):
                state["StartedAt"] = "2026-11-05T07:26:20.806109801Z"
            elif fault == "running": state["Running"] = True
            elif fault == "nonzero-exit": state["ExitCode"] = 41
            elif fault == "oom": state["OOMKilled"] = True
            elif fault == "dead": state["Dead"] = True
            elif fault == "error": state["Error"] = "SCRIPTED daemon state error"
            elif fault == "pid": state["Pid"] = 123
            return row
        async def invoke(self, argv, *, timeout=None):
            calls.append(tuple(argv))
            assert "--force" not in argv
            if argv == ["kill", "--signal=SIGTERM", cid]:
                self.signal_attempted = True
                raise signal_error
            if argv == ["wait", cid]:
                if fault == "failed-wait": raise wait_error
                self.wait_returned = True
                return b"41\n" if fault == "nonzero-wait" else b"0\n"
            if argv == ["container", "rm", cid]:
                removed.add(cid)
                raise AssertionError("refused original retirement must not remove")
            raise AssertionError("unexpected scripted command")
    resource = object.__new__(_LocalWorkerResources)
    resource.profile, resource.identity, resource.boundary = profile, identity, Daemon()
    resource.containers = {"browser": cid}
    resource._native_instances = {"browser": (123, MEASURED_FINISHED_STATE["StartedAt"])}
    resource._container_waits = {}
    resource.network = resource.volume = None
    resource._network_identity = resource._volume_identity = None
    resource._pending_creates = set()
    resource._closed = resource._closing = False
    resource._binding = resource
    resource.original = SimpleNamespace(_resources=resource)
    admission = resource._close_admission = resource._original_close_admission = OwnedTaskAdmission()
    errors, tasks, outcomes, join_errors = [], [], [], []
    for _ in range(2):
        try:
            await resource.close()
        except BaseException as error:
            errors.append(error)
        finally:
            tasks.append(admission.task)
            actual, failures = await _join_release_tasks((admission,))
            outcomes.extend(actual); join_errors.extend(failures)
    _raise_release_errors(None, (), join_errors)
    for outcome in outcomes:
        if outcome is not None and outcome.cancellation is not None:
            _raise_release_errors(None, (outcome,))
    assert tasks[0] is not None and all(task is tasks[0] for task in tasks)
    assert errors and all(outcome is not None and outcome.error is not None for outcome in outcomes)
    assert all(any(node is signal_error for node in _explicit_control_chain(error)) for error in errors)
    if fault == "failed-wait":
        assert all(any(node is wait_error for node in _explicit_control_chain(error)) for error in errors)
    assert calls.count(("kill", "--signal=SIGTERM", cid)) == 1
    assert not resource._closed and not removed and ("container", "rm", cid) not in calls
    if fault in {"nonzero-wait", "failed-wait", "changed-after-wait"}:
        assert calls.count(("wait", cid)) == 1
    else:
        assert ("wait", cid) not in calls
    assert resource._container_waits == {}
    (tmp_path / "guard-case-facts.json").write_text(json.dumps({
        "fault": fault, "calls": calls, "original_signal_retained": True,
        "same_original_closer_joined_twice": True, "native_Docker_wait_proved": False,
    }, indent=2))
